"""양식 필수 항목을 보고서 검증 MCP 기준으로 변환해 초안 완결성을 검증한다."""

from __future__ import annotations

from typing import Any

from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.contracts.report_validation import (
    ReportValidationCriterion,
    ReportValidationElement,
    ReportValidationResult,
)
from n_note_ai_service_layers.services.draft_document import DraftDocument, ordered_fields, ordered_sections
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient, McpGatewayError

STANDARD_SCHEMA_VERSION = "report-validation-standard/v1"
RESULT_POLICY = "candidate_only_human_review_required"
MUTATION_POLICY = "no_automatic_report_modification"
_STATUSES = {"satisfied", "missing", "insufficient", "uncertain"}


class FormReportValidationFlow:
    """Gateway의 report-validation 도구로 섹션별 필수 항목 작성 여부를 검증한다.

    입력: Gateway client와 사용할 도구 이름이다.
    출력: 섹션·추가 기준별 ``satisfied/missing/insufficient/uncertain`` 결과이며 초안을 수정하지 않는다.
    """

    def __init__(self, gateway: McpGatewayClient, tool_name: str | None = None) -> None:
        from n_note_ai_service_layers.settings import report_validation_tool_name
        self.gateway = gateway
        self.tool_name = tool_name if tool_name is not None else report_validation_tool_name()

    def validate(self, schema: FormSchema, document: DraftDocument, request_id: str | None = None, catalog: tuple[dict[str, Any], ...] | None = None) -> ReportValidationResult:
        standard = self.standard_for(schema)
        if not self.tool_name or not standard["criteria"]:
            return ReportValidationResult("skipped")
        if catalog is None:
            try:
                catalog = self.gateway.discover(request_id)
            except McpGatewayError:
                return ReportValidationResult("unavailable", tool_name=self.tool_name)
        if not any(tool.get("name") == self.tool_name for tool in catalog):
            return ReportValidationResult("unsupported", tool_name=self.tool_name)
        arguments = {
            "report": {"text": document.text},
            "validation_standard": standard,
            "semantic_assessment": self.assessment_for(standard, document),
        }
        try:
            result = self.gateway.call_tool(self.tool_name, arguments, request_id)
        except McpGatewayError:
            return ReportValidationResult("unavailable", tool_name=self.tool_name)
        return self._parse(result.get("structuredContent"), standard)

    @staticmethod
    def standard_for(schema: FormSchema) -> dict[str, Any]:
        """양식 섹션의 필수 필드와 양식별 추가 기준을 검증 기준 JSON으로 만든다."""
        criteria: list[dict[str, Any]] = []
        for section in ordered_sections(schema):
            required = [field for field in ordered_fields(section) if field.required]
            if not required:
                continue
            criteria.append({
                "criterion_id": f"section:{section.section_id}",
                "description": f"{section.label or section.section_id}의 필수 항목이 작성되어야 한다.",
                "category": "form_completeness",
                "required_elements": [
                    {"element_id": field.field_id, "description": field.label or field.field_id, "field_id": field.field_id}
                    for field in required
                ],
            })
        criteria.extend(dict(item) for item in schema.validation_criteria)
        return {
            "schema_version": STANDARD_SCHEMA_VERSION,
            "standard_id": f"n-note-form:{schema.schema_id}",
            "standard_version": schema.version,
            "status": "active",
            "usage": {"result_policy": RESULT_POLICY, "mutation_policy": MUTATION_POLICY},
            "criteria": criteria,
        }

    @staticmethod
    def assessment_for(standard: dict[str, Any], document: DraftDocument) -> dict[str, Any]:
        """필드와 연결된 요소만 초안 원문 범위 citation으로 결정적으로 평가한다.

        필드와 연결되지 않은 양식별 기준은 평가를 제출하지 않아 MCP가 ``uncertain``으로 남기며,
        사람이 직접 확인해야 한다.
        """
        assessed: list[dict[str, Any]] = []
        for criterion in standard["criteria"]:
            elements = []
            for element in criterion.get("required_elements", []):
                span = document.span_for(element.get("field_id")) if isinstance(element, dict) and element.get("field_id") else None
                if span is None:
                    continue
                quote = document.text[span.start:span.end]
                if quote.strip():
                    elements.append({"element_id": element["element_id"], "status": "satisfied", "citations": [{"start": span.start, "end": span.end, "quote": quote}]})
                else:
                    elements.append({"element_id": element["element_id"], "status": "missing", "reason": "초안에 값이 없습니다."})
            if not elements or len(elements) != len(criterion.get("required_elements", [])):
                continue
            satisfied = [item for item in elements if item["status"] == "satisfied"]
            citations = [citation for item in satisfied for citation in item["citations"]]
            if len(satisfied) == len(elements):
                status = "satisfied"
            elif satisfied:
                status = "insufficient"
            else:
                status = "missing"
            entry: dict[str, Any] = {"criterion_id": criterion["criterion_id"], "status": status, "elements": elements}
            if citations:
                entry["citations"] = citations
            assessed.append(entry)
        return {"criteria": assessed}

    def _parse(self, payload: Any, standard: dict[str, Any]) -> ReportValidationResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("criteria"), list):
            return ReportValidationResult("unsupported", tool_name=self.tool_name)
        field_ids = {
            element["element_id"]: element.get("field_id")
            for criterion in standard["criteria"]
            for element in criterion.get("required_elements", [])
            if isinstance(element, dict) and "element_id" in element
        }
        criteria: list[ReportValidationCriterion] = []
        for item in payload["criteria"]:
            finding = item.get("finding") if isinstance(item, dict) else None
            if not isinstance(finding, dict) or finding.get("status") not in _STATUSES or not isinstance(item.get("elements"), list):
                return ReportValidationResult("unsupported", tool_name=self.tool_name)
            elements: list[ReportValidationElement] = []
            for element in item["elements"]:
                if not isinstance(element, dict) or element.get("status") not in _STATUSES or not isinstance(element.get("id"), str):
                    return ReportValidationResult("unsupported", tool_name=self.tool_name)
                elements.append(ReportValidationElement(element["id"], element["status"], field_ids.get(element["id"]), tuple(str(w) for w in element.get("warnings", []))))
            criteria.append(ReportValidationCriterion(
                str(item.get("criterion_id")),
                str(item.get("description") or ""),
                finding["status"],
                tuple(elements),
                tuple(str(w) for w in finding.get("warnings", [])),
            ))
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else None
        return ReportValidationResult("ok", tuple(criteria), summary, self.tool_name)
