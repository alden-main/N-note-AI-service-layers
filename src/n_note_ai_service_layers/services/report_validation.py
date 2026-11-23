"""초안의 양식 완결성을 검증한다: 필수 필드는 로컬에서, 양식별 추가 기준은 보고서 검증 MCP로."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.contracts.gateway_payloads import ReportEvaluation
from n_note_ai_service_layers.contracts.report_validation import (
    ReportValidationCriterion,
    ReportValidationElement,
    ReportValidationResult,
    ReportValidationStatus,
    ValidationItemStatus,
)
from n_note_ai_service_layers.services.draft_document import DraftDocument, ordered_fields, ordered_sections
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient, McpGatewayError
from n_note_ai_service_layers.settings import DEFAULT_REPORT_VALIDATION_TOOL

STANDARD_SCHEMA_VERSION = "report-validation-standard/v1"
RESULT_POLICY = "candidate_only_human_review_required"
MUTATION_POLICY = "no_automatic_report_modification"


class FormReportValidationFlow:
    """양식 완결성을 검증한다.

    - 섹션별 필수 필드: 초안 값이 있는지 로컬에서 판정한다. 결과는 초안 문서만으로 정해지므로 MCP를 거치지 않는다
      (Gateway 장애와 무관하게 항상 결과가 있다).
    - 양식별 추가 기준(``validation_criteria``): 필드만으로 판정할 수 없으므로 Gateway의 report 도구에 맡긴다.

    출력: 기준별 ``satisfied/missing/insufficient/uncertain``이며 초안을 수정하지 않는다.
    """

    def __init__(self, gateway: McpGatewayClient, tool_name: str | None = DEFAULT_REPORT_VALIDATION_TOOL) -> None:
        """``tool_name``이 ``None``이면 양식별 추가 기준 검증을 생략한다(필수 항목 판정은 계속)."""
        self.gateway = gateway
        self.tool_name = tool_name

    def validate(
        self, schema: FormSchema, document: DraftDocument, request_id: str | None = None,
        catalog: tuple[dict[str, Any], ...] | None = None, gateway_available: bool = True,
    ) -> ReportValidationResult:
        """``gateway_available=False``(workflow가 catalog 조회에 실패)면 추가 기준은 ``unavailable``로 둔다."""
        local = self.completeness(schema, document)
        extra = [dict(item) for item in schema.validation_criteria]
        if not extra:
            return ReportValidationResult("ok", local, _summary(local)) if local else ReportValidationResult("skipped")
        status, remote = self._remote(extra, schema, document, request_id, catalog, gateway_available)
        criteria = (*local, *remote)
        return ReportValidationResult(status, criteria, _summary(criteria), self.tool_name)

    @staticmethod
    def completeness(schema: FormSchema, document: DraftDocument) -> tuple[ReportValidationCriterion, ...]:
        """섹션마다 필수 필드가 모두 채워졌는지다(모두 ``satisfied``, 일부 ``insufficient``, 전혀 없으면 ``missing``)."""
        criteria: list[ReportValidationCriterion] = []
        for section in ordered_sections(schema):
            required = [field for field in ordered_fields(section) if field.required]
            if not required:
                continue
            elements = tuple(
                ReportValidationElement(field.field_id, "satisfied" if _filled(document, field.field_id) else "missing", field.field_id)
                for field in required
            )
            filled = sum(element.status == "satisfied" for element in elements)
            status: ValidationItemStatus = "satisfied" if filled == len(elements) else "insufficient" if filled else "missing"
            criteria.append(ReportValidationCriterion(
                f"section:{section.section_id}", f"{section.label or section.section_id}의 필수 항목이 작성되어야 한다.", status, elements,
            ))
        return tuple(criteria)

    def _remote(
        self, criteria: list[dict[str, Any]], schema: FormSchema, document: DraftDocument, request_id: str | None,
        catalog: tuple[dict[str, Any], ...] | None, gateway_available: bool,
    ) -> tuple[ReportValidationStatus, tuple[ReportValidationCriterion, ...]]:
        """양식별 추가 기준을 report 도구로 검증한다. 반환: (상태, 기준 결과)."""
        if not self.tool_name:
            return "skipped", ()
        if not gateway_available:
            return "unavailable", ()
        if catalog is None:
            try:
                catalog = self.gateway.discover(request_id)
            except McpGatewayError:
                return "unavailable", ()
        if not any(tool.get("name") == self.tool_name for tool in catalog):
            return "unsupported", ()
        standard = self.standard_for(schema, criteria)
        arguments = {
            "report": {"text": document.text},
            "validation_standard": standard,
            "semantic_assessment": self.assessment_for(standard, document),
        }
        try:
            result = self.gateway.call_tool(self.tool_name, arguments, request_id)
        except McpGatewayError:
            return "unavailable", ()
        parsed = self._parse(result.get("structuredContent"), standard)
        return ("ok", parsed) if parsed is not None else ("unsupported", ())

    @staticmethod
    def standard_for(schema: FormSchema, criteria: list[dict[str, Any]]) -> dict[str, Any]:
        """report 도구에 보낼 검증 기준 JSON이다(양식별 추가 기준만 담는다)."""
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
                field_id = element.get("field_id") if isinstance(element, dict) else None
                span = document.span_for(field_id) if isinstance(field_id, str) and field_id else None
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

    @staticmethod
    def _parse(payload: Any, standard: dict[str, Any]) -> tuple[ReportValidationCriterion, ...] | None:
        """report 도구 응답을 기준 결과로 바꾼다. 형식이 다르면 ``None``이다."""
        try:
            evaluation = ReportEvaluation.model_validate(payload)
        except ValidationError:
            return None
        field_ids = {
            element["element_id"]: element.get("field_id")
            for criterion in standard["criteria"]
            for element in criterion.get("required_elements", [])
            if isinstance(element, dict) and "element_id" in element
        }
        return tuple(
            ReportValidationCriterion(
                str(item.criterion_id),
                str(item.description or ""),
                item.finding.status,
                tuple(
                    ReportValidationElement(element.id, element.status, field_ids.get(element.id), tuple(str(w) for w in element.warnings))
                    for element in item.elements
                ),
                tuple(str(w) for w in item.finding.warnings),
            )
            for item in evaluation.criteria
        )


def _filled(document: DraftDocument, field_id: str) -> bool:
    span = document.span_for(field_id)
    return span is not None and bool(document.text[span.start:span.end].strip())


def _summary(criteria: tuple[ReportValidationCriterion, ...]) -> dict[str, int]:
    """상태별 개수다(기준과 그 요소를 함께 센다). report MCP의 summary와 같은 형식이다."""
    counts = {status: 0 for status in ("satisfied", "missing", "insufficient", "uncertain")}
    for criterion in criteria:
        counts[criterion.status] += 1
        for element in criterion.elements:
            counts[element.status] += 1
    return counts
