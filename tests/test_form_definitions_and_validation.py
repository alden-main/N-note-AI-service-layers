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

FORMS_DIR = Path(__file__).parents[1] / "forms"
VALIDATION_TOOL = "report-validation__evaluate_report_against_standard"


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


class _ValidationGateway:
    """report-validation MCP의 citation 검증 규칙을 흉내 내는 Gateway다."""

    def __init__(self):
        self.arguments = None

    def __call__(self, url, body, headers, timeout):
        message = json.loads(body)
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": VALIDATION_TOOL, "description": "evaluate", "inputSchema": {"type": "object", "properties": {"report": {"type": "object"}, "validation_standard": {"type": "object"}}, "required": ["report", "validation_standard"], "additionalProperties": True}}]}
        else:
            self.arguments = message["params"]["arguments"]
            text = self.arguments["report"]["text"]
            assessed = {item["criterion_id"]: item for item in self.arguments["semantic_assessment"]["criteria"]}
            criteria = []
            for criterion in self.arguments["validation_standard"]["criteria"]:
                candidate = assessed.get(criterion["criterion_id"])
                valid = candidate is not None and all(text[c["start"]:c["end"]] == c["quote"] for c in candidate.get("citations", []))
                elements = {item["element_id"]: item["status"] for item in (candidate or {}).get("elements", [])}
                criteria.append({
                    "criterion_id": criterion["criterion_id"], "description": criterion["description"],
                    "finding": {"id": criterion["criterion_id"], "status": candidate["status"] if valid else "uncertain", "reason": None, "citations": [], "warnings": []},
                    "elements": [{"id": e["element_id"], "status": elements.get(e["element_id"], "uncertain"), "reason": None, "citations": [], "warnings": []} for e in criterion["required_elements"]],
                })
            result = {"structuredContent": {"review_required": True, "warnings": [], "summary": {}, "criteria": criteria}, "isError": False}
        return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}, ensure_ascii=False).encode()


class FormReportValidationTests(unittest.TestCase):
    def test_required_fields_become_criteria_and_citations_match_the_report(self):
        schema = next(item for item in FormDefinitionLoader(FORMS_DIR).load_all() if item.schema_id == "nursing-daily-note")
        document = DraftDocument.build(schema, {"assessment": "통증 호소 없음", "plan_and_intervention": None})
        gateway = _ValidationGateway()
        result = FormReportValidationFlow(McpGatewayClient(transport=gateway), VALIDATION_TOOL).validate(schema, document)
        self.assertEqual(result.status, "ok")
        standard = gateway.arguments["validation_standard"]
        self.assertEqual(standard["schema_version"], "report-validation-standard/v1")
        self.assertEqual(standard["usage"]["mutation_policy"], "no_automatic_report_modification")
        criterion = result.criteria[0]
        self.assertEqual(criterion.status, "insufficient")
        self.assertEqual({element.field_id: element.status for element in criterion.elements}, {"assessment": "satisfied", "plan_and_intervention": "missing"})

    def test_validation_is_reported_as_unsupported_when_gateway_lacks_the_tool(self):
        schema = next(item for item in FormDefinitionLoader(FORMS_DIR).load_all() if item.schema_id == "nursing-daily-note")
        result = FormReportValidationFlow(McpGatewayClient(transport=_ValidationGateway()), "missing__tool").validate(schema, DraftDocument.build(schema, {}))
        self.assertEqual(result.status, "unsupported")


if __name__ == "__main__":
    unittest.main()
