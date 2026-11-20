from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from n_note_ai_service_layers.adapters.form_definition_loader import FormDefinitionError, FormDefinitionLoader
from n_note_ai_service_layers.contracts.form_schema import SchemaStatus
from n_note_ai_service_layers.services.draft_document import DraftDocument
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from n_note_ai_service_layers.services.report_validation import FormReportValidationFlow
from support import FakeGateway

FORMS_DIR = Path(__file__).parents[1] / "forms"
VALIDATION_TOOL = "report__evaluate_report_against_standard"


def _write_form(root: Path, form_id: str, **overrides) -> Path:
    definition = {
        "schema_id": form_id,
        "version": f"{form_id}/v1",
        "purpose": form_id.upper(),
        "title": "테스트 양식",
        "sections": [{"section_id": "s", "label": "섹션", "fields": [{"field_id": "note", "type": "text", "label": "내용", "required": True}]}],
    }
    definition.update(overrides)
    directory = root / form_id
    directory.mkdir(parents=True)
    (directory / "form.json").write_text(json.dumps(definition, ensure_ascii=False), encoding="utf-8")
    return directory


class FormDefinitionLoaderTests(unittest.TestCase):
    def test_bundled_forms_load_as_active_schemas(self):
        schemas = FormDefinitionLoader(FORMS_DIR).load_all()
        ids = {schema.schema_id for schema in schemas}
        self.assertTrue({"nursing-daily-note", "sbar-handoff"} <= ids)
        self.assertTrue(all(schema.status is SchemaStatus.ACTIVE for schema in schemas))
        daily = next(schema for schema in schemas if schema.schema_id == "nursing-daily-note")
        self.assertEqual(daily.display_title, "간호일지")
        self.assertIn("assessment", {field.field_id for field in daily.fields if field.required})

    def test_enabled_forms_select_a_subset_and_unknown_ids_fail(self):
        with tempfile.TemporaryDirectory() as root:
            _write_form(Path(root), "form-a")
            _write_form(Path(root), "form-b")
            loader = FormDefinitionLoader(Path(root))
            self.assertEqual([schema.schema_id for schema in loader.load_all(("form-b",))], ["form-b"])
            with self.assertRaises(FormDefinitionError):
                loader.load_all(("form-c",))

    def test_previous_versions_load_as_approved_for_existing_drafts(self):
        schemas = FormDefinitionLoader(FORMS_DIR).load_previous()
        daily = [schema for schema in schemas if schema.schema_id == "nursing-daily-note"]
        self.assertEqual([(schema.version, schema.status) for schema in daily], [("nursing-daily-note/v1", SchemaStatus.APPROVED)])
        self.assertEqual(FormDefinitionLoader(FORMS_DIR).load_previous(("sbar-handoff",)), ())

    def test_registry_keeps_previous_version_lookup_but_creates_with_active(self):
        from n_note_ai_service_layers.api import get_services
        registry = get_services().template_registry
        self.assertEqual(registry.get("nursing-daily-note", "nursing-daily-note/v1").status, SchemaStatus.APPROVED)
        self.assertEqual(registry.resolve_active("nursing-daily-note", "NURSING_DAILY_NOTE").version, "nursing-daily-note/v2")
        with self.assertRaises(LookupError):
            registry.resolve_active("nursing-daily-note", "NURSING_DAILY_NOTE", "nursing-daily-note/v1")

    def test_previous_version_must_share_schema_id_and_differ_in_version(self):
        for schema_id, version in (("form-a", "form-a/v1"), ("other", "other/v0")):
            with self.subTest(schema_id=schema_id), tempfile.TemporaryDirectory() as root:
                directory = _write_form(Path(root), "form-a")
                active = json.loads((directory / "form.json").read_text(encoding="utf-8"))
                (directory / "previous").mkdir()
                (directory / "previous" / "old.json").write_text(json.dumps({**active, "schema_id": schema_id, "version": version}), encoding="utf-8")
                with self.assertRaises(FormDefinitionError):
                    FormDefinitionLoader(Path(root)).load_previous()

    def test_pdf_template_must_stay_inside_the_form_directory(self):
        with tempfile.TemporaryDirectory() as root:
            _write_form(Path(root), "form-a", pdf={"template": "../outside.pdf"})
            (Path(root) / "outside.pdf").write_bytes(b"%PDF-1.4")
            with self.assertRaises(FormDefinitionError):
                FormDefinitionLoader(Path(root)).load_all()

    def test_enum_requires_options_and_field_placement_requires_a_location(self):
        with tempfile.TemporaryDirectory() as root:
            _write_form(Path(root), "form-a", sections=[{"section_id": "s", "fields": [{"field_id": "x", "type": "enum"}]}])
            with self.assertRaises(FormDefinitionError):
                FormDefinitionLoader(Path(root)).load_all()
        with tempfile.TemporaryDirectory() as root:
            _write_form(Path(root), "form-a", sections=[{"section_id": "s", "fields": [{"field_id": "x", "type": "text", "pdf": {"font_size": 9}}]}])
            with self.assertRaises(FormDefinitionError):
                FormDefinitionLoader(Path(root)).load_all()


VALIDATION_TOOL_DEFINITION = {
    "name": VALIDATION_TOOL, "description": "evaluate",
    "inputSchema": {"type": "object", "properties": {"report": {"type": "object"}, "validation_standard": {"type": "object"}}, "required": ["report", "validation_standard"], "additionalProperties": True},
}


def _evaluate(_name: str, arguments: dict) -> dict:
    """report MCP의 citation 검증 규칙을 흉내 낸다(citation이 원문과 다르면 uncertain)."""
    text = arguments["report"]["text"]
    assessed = {item["criterion_id"]: item for item in arguments["semantic_assessment"]["criteria"]}
    criteria = []
    for criterion in arguments["validation_standard"]["criteria"]:
        candidate = assessed.get(criterion["criterion_id"])
        valid = candidate is not None and all(text[c["start"]:c["end"]] == c["quote"] for c in candidate.get("citations", []))
        elements = {item["element_id"]: item["status"] for item in (candidate or {}).get("elements", [])}
        criteria.append({
            "criterion_id": criterion["criterion_id"], "description": criterion["description"],
            "finding": {"id": criterion["criterion_id"], "status": candidate["status"] if valid else "uncertain", "reason": None, "citations": [], "warnings": []},
            "elements": [{"id": e["element_id"], "status": elements.get(e["element_id"], "uncertain"), "reason": None, "citations": [], "warnings": []} for e in criterion["required_elements"]],
        })
    return {"structuredContent": {"review_required": True, "warnings": [], "summary": {}, "criteria": criteria}, "isError": False}


def _validation_gateway() -> FakeGateway:
    return FakeGateway(tools=[VALIDATION_TOOL_DEFINITION], handler=_evaluate)



class FormSchemaContractTests(unittest.TestCase):
    """양식 구조 규칙은 계약 한 곳(생성 시점)에서 검사한다."""

    def _schema(self, *fields, **changes):
        from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection
        return FormSchema("f", "v1", "P", (FormSection("s", tuple(fields) or (FormField("a"),)),), **changes)

    def test_structure_rules_are_enforced_on_construction(self):
        from n_note_ai_service_layers.contracts.form_schema import BridgeMapping, FormField, FormSection
        cases = {
            "field IDs must be unique": lambda: self._schema(FormField("a"), FormField("a")),
            "unknown field": lambda: self._schema(FormField("a", conditional_required_when="missing")),
            "bridge mapping": lambda: self._schema(bridge_mapping=(BridgeMapping("missing", "out"),)),
            "enum field requires options": lambda: FormField("a", "enum"),
            "pdf placement requires": lambda: FormField("a", pdf={"page": 1}),
            "literal_error": lambda: FormField("a", "object"),
            "too_short": lambda: FormSection("s", ()),
            "string_too_short": lambda: FormField("  "),
        }
        for message, build in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                build()

    def test_replace_revalidates(self):
        from dataclasses import replace
        from n_note_ai_service_layers.contracts.form_schema import FormField, FormSection
        schema = self._schema(FormField("a"), FormField("b"))
        with self.assertRaisesRegex(ValueError, "unique"):
            replace(schema, sections=(FormSection("s", (FormField("a"), FormField("a"))),))
        self.assertEqual(replace(schema, title=" 제목 ").title, "제목")

    def test_registration_api_uses_the_same_contract(self):
        from fastapi.testclient import TestClient
        from n_note_ai_service_layers.api import app
        field = {"field_id": "x", "type": "enum", "options": ["예", 1]}
        payload = {"schema_id": "contract-test", "version": "v1", "purpose": "CONTRACT_TEST", "sections": [{"section_id": "s", "fields": [field]}]}
        client = TestClient(app)
        bad = client.post("/v1/form-templates", json={**payload, "sections": [{"section_id": "s", "fields": [{"field_id": "x", "type": "enum"}]}]})
        self.assertEqual(bad.status_code, 422)
        good = client.post("/v1/form-templates", json=payload)
        self.assertEqual(good.status_code, 201, good.text)
        self.assertEqual(good.json()["sections"][0]["fields"][0]["options"], ["예", "1"])


class FormReportValidationTests(unittest.TestCase):
    def _schema(self, **changes):
        from dataclasses import replace
        schema = next(item for item in FormDefinitionLoader(FORMS_DIR).load_all() if item.schema_id == "nursing-daily-note")
        return replace(schema, **changes)

    def test_required_fields_are_checked_locally_without_the_gateway(self):
        """필수 필드 완결성은 초안 문서만으로 정해지므로 report MCP를 부르지 않는다."""
        schema = self._schema()
        document = DraftDocument.build(schema, {"assessment": "통증 호소 없음", "plan_and_intervention": None})
        gateway = _validation_gateway()
        result = FormReportValidationFlow(McpGatewayClient(transport=gateway), VALIDATION_TOOL).validate(schema, document)
        self.assertEqual((result.status, result.tool_name), ("ok", None))
        self.assertEqual((gateway.called, gateway.requests), ([], []))
        criterion = result.criteria[0]
        self.assertEqual(criterion.status, "insufficient")
        self.assertEqual({element.field_id: element.status for element in criterion.elements}, {"assessment": "satisfied", "plan_and_intervention": "missing"})
        self.assertEqual(result.summary, {"satisfied": 1, "missing": 1, "insufficient": 1, "uncertain": 0})

    def test_completeness_survives_a_gateway_outage(self):
        schema = self._schema()
        flow = FormReportValidationFlow(McpGatewayClient(transport=_validation_gateway()), VALIDATION_TOOL)
        result = flow.validate(schema, DraftDocument.build(schema, {}), gateway_available=False)
        self.assertEqual(result.status, "ok")
        self.assertEqual({item.status for item in result.criteria}, {"missing"})

    def test_form_specific_criteria_go_to_the_report_tool_and_merge_with_local_results(self):
        extra = {"criterion_id": "custom:pain", "description": "통증 평가가 서술되어야 한다.", "category": "content",
                 "required_elements": [{"element_id": "pain", "description": "통증", "field_id": "assessment"}]}
        schema = self._schema(validation_criteria=(extra,))
        document = DraftDocument.build(schema, {"assessment": "통증 호소 없음", "plan_and_intervention": "관찰함"})
        gateway = _validation_gateway()
        result = FormReportValidationFlow(McpGatewayClient(transport=gateway), VALIDATION_TOOL).validate(schema, document)
        self.assertEqual((result.status, result.tool_name), ("ok", VALIDATION_TOOL))
        standard = gateway.called[-1]["arguments"]["validation_standard"]
        self.assertEqual(standard["schema_version"], "report-validation-standard/v1")
        self.assertEqual(standard["usage"]["mutation_policy"], "no_automatic_report_modification")
        # MCP에는 추가 기준만 보낸다(필수 필드 기준은 로컬 판정).
        self.assertEqual([item["criterion_id"] for item in standard["criteria"]], ["custom:pain"])
        self.assertEqual([item.criterion_id for item in result.criteria][-1], "custom:pain")
        self.assertEqual(result.criteria[-1].status, "satisfied")
        self.assertTrue(all(item.status == "satisfied" for item in result.criteria))

    def test_form_specific_criteria_are_unsupported_when_gateway_lacks_the_tool(self):
        extra = {"criterion_id": "custom:x", "description": "x", "required_elements": []}
        schema = self._schema(validation_criteria=(extra,))
        result = FormReportValidationFlow(McpGatewayClient(transport=_validation_gateway()), "missing__tool").validate(schema, DraftDocument.build(schema, {}))
        self.assertEqual(result.status, "unsupported")
        # 로컬 완결성 결과는 그대로 남는다.
        self.assertTrue(result.criteria)


if __name__ == "__main__":
    unittest.main()
