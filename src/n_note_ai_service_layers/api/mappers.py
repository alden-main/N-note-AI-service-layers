"""내부 불변 계약을 API 응답 JSON 형식으로 변환한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.contracts.draft_review import DraftCandidate, DraftReviewRecord, FinalizedNursingNote
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.report_validation import ReportValidationResult
from n_note_ai_service_layers.api.schemas import TemplateRegistrationRequest
from n_note_ai_service_layers.services.draft_document import ordered_fields, ordered_sections


def template_response(schema: FormSchema) -> dict[str, Any]:
    """내부 불변 template 계약을 API 응답 형식으로 변환한다."""
    return {
        "schema_id": schema.schema_id,
        "version": schema.version,
        "purpose": schema.purpose,
        "title": schema.display_title,
        "status": schema.status.value,
        "approved_by": schema.approved_by,
        "approved_at": schema.approved_at,
        "pdf_template": Path(schema.pdf_template).name if schema.pdf_template else None,
        "is_default": schema.schema_id == settings.default_form_id(),
        "sections": [
            {
                "section_id": section.section_id,
                "label": section.label,
                "display_order": section.display_order,
                "fields": [
                    {
                        "field_id": field.field_id,
                        "type": field.type,
                        "label": field.label,
                        "required": field.required,
                        "llm_allowed": field.llm_allowed,
                        "fact_id": field.fact_id,
                        "conditional_required_when": field.conditional_required_when,
                        "display_order": field.display_order,
                        "description": field.description,
                        "options": list(field.options),
                    }
                    for field in section.fields
                ],
            }
            for section in schema.sections
        ],
    }


def request_schema(request: TemplateRegistrationRequest) -> FormSchema:
    """template 등록 요청을 항상 draft 상태의 내부 계약으로 변환한다."""
    return FormSchema(
        request.schema_id,
        request.version,
        request.purpose,
        tuple(
            FormSection(
                section.section_id,
                tuple(
                    FormField(
                        field.field_id,
                        field.type,
                        field.required,
                        field.llm_allowed,
                        field.fact_id,
                        field.conditional_required_when,
                        field.label,
                        field.display_order,
                    )
                    for field in section.fields
                ),
                section.label,
                section.display_order,
            )
            for section in request.sections
        ),
        status=SchemaStatus.DRAFT,
    )


def candidate_id(phase: str, tool_name: str, index: int) -> str:
    return f"{phase}:{tool_name}:{index}"


def gateway_verification_response(result) -> dict[str, Any]:
    return {
        "phase": result.phase,
        "status": result.status,
        "tool_names": list(result.tool_names),
        "auto_apply": result.auto_apply,
        "occurrences": [
            {
                "candidate_id": candidate_id(result.phase, item.tool_name, index),
                "tool_name": item.tool_name, "start_offset": item.start_offset, "end_offset": item.end_offset,
                "input_text": item.input_text, "normalized_text": item.normalized_text,
                "field_id": item.field_id, "category": item.category,
                "review_status": item.review_status,
                "suggestions": [{"text": suggestion.text, "score": suggestion.score, "reason": suggestion.reason} for suggestion in item.suggestions],
            }
            for index, item in enumerate(result.occurrences)
        ],
    }


def candidate_ids(*results) -> tuple[str, ...]:
    """workflow 응답의 후보 ID와 revision 저장소의 후보 ID를 같은 규칙으로 만든다."""
    return tuple(
        candidate_id(result.phase, item.tool_name, index)
        for result in results
        for index, item in enumerate(result.occurrences)
    )


def draft_candidates(*results) -> tuple[DraftCandidate, ...]:
    """Gateway 후보를 사용자가 승인·적용할 수 있는 초안 검토 후보로 변환한다."""
    return tuple(
        DraftCandidate(
            candidate_id(result.phase, item.tool_name, index), result.phase, item.tool_name,
            item.start_offset, item.end_offset, item.input_text,
            tuple(suggestion.text for suggestion in item.suggestions), item.field_id, item.category,
        )
        for result in results
        for index, item in enumerate(result.occurrences)
    )


def numeric_verification_response(findings) -> dict[str, Any]:
    """출처에 없는 초안 수치 목록이다. 비어 있지 않으면 확정 시 수치 확인이 필요하다."""
    return {
        "status": "needs_review" if findings else "ok",
        "findings": [
            {"finding_id": item.finding_id, "field_id": item.field_id, "start_offset": item.start_offset,
             "end_offset": item.end_offset, "text": item.text, "reason": item.reason}
            for item in findings
        ],
    }


def report_validation_response(result: ReportValidationResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "tool_name": result.tool_name,
        "review_required": result.review_required,
        "summary": result.summary,
        "criteria": [
            {
                "criterion_id": criterion.criterion_id,
                "description": criterion.description,
                "status": criterion.status,
                "warnings": list(criterion.warnings),
                "elements": [
                    {"element_id": element.element_id, "field_id": element.field_id, "status": element.status, "warnings": list(element.warnings)}
                    for element in criterion.elements
                ],
            }
            for criterion in result.criteria
        ],
    }


def sections_response(schema: FormSchema, fields: dict[str, Any], evidence_refs: dict[str, Any] | None = None, candidates: tuple[DraftCandidate, ...] = ()) -> list[dict[str, Any]]:
    """평면 필드를 양식 섹션·표시 순서대로 묶고, 필드별 적용 가능한 후보 ID를 함께 반환한다."""
    return [
        {
            "section_id": section.section_id,
            "label": section.label,
            "display_order": section.display_order,
            "fields": [
                {
                    "field_id": field.field_id,
                    "label": field.label,
                    "type": field.type,
                    "required": field.required,
                    "display_order": field.display_order,
                    "value": fields.get(field.field_id),
                    "evidence_refs": list((evidence_refs or {}).get(field.field_id, ())),
                    "candidate_ids": [item.candidate_id for item in candidates if item.field_id == field.field_id and not item.stale],
                }
                for field in ordered_fields(section)
            ],
        }
        for section in ordered_sections(schema)
    ]


def draft_review_response(record: DraftReviewRecord, schema: FormSchema, numeric_findings) -> dict[str, Any]:
    """초안 검토 레코드를 사용자 수정 API의 응답 형식으로 변환한다."""
    latest = record.revisions[-1]
    return {
        "draft_id": record.draft_id,
        "schema_id": record.schema_id,
        "schema_version": record.schema_version,
        "form_title": schema.display_title,
        "transcript": record.transcript,
        "candidate_ids": list(record.candidate_ids),
        "revision": latest.revision,
        "fields": latest.fields,
        "sections": sections_response(schema, latest.fields, candidates=record.candidates),
        "candidates": [
            {
                "candidate_id": item.candidate_id, "phase": item.phase, "tool_name": item.tool_name,
                "field_id": item.field_id, "category": item.category,
                "start_offset": item.start_offset, "end_offset": item.end_offset,
                "input_text": item.input_text, "suggestions": list(item.suggestions),
                "decision": latest.candidate_decisions.get(item.candidate_id, "pending"),
                "applicable": bool(item.field_id) and not item.stale,
                "stale": item.stale,
            }
            for item in record.candidates
        ],
        "candidate_decisions": latest.candidate_decisions,
        "applied_suggestions": {key: value for revision in record.revisions for key, value in revision.applied_suggestions.items()},
        "llm_applied_term_hints": [dict(item) for item in record.llm_applied_term_hints],
        "edited_by": latest.edited_by,
        "created_at": latest.created_at,
        "finalized_note_id": record.finalized_note_id,
        "numeric_verification": numeric_verification_response(numeric_findings),
    }


def note_response(note: FinalizedNursingNote, pdf_status: str) -> dict[str, Any]:
    return {**note.as_dict(), "pdf_status": pdf_status, "pdf_url": f"/v1/nursing-notes/{note.note_id}/pdf" if pdf_status == "ready" else None}


def workflow_result_response(result, review: DraftReviewRecord, schema: FormSchema, trace_id: str) -> dict[str, Any]:
    """초안 생성 작업의 결과다. 검토 화면은 ``draft_id``로 초안 API를 이어서 사용한다."""
    return {
        "trace_id": trace_id, "auto_apply": False, "redacted": True, "redaction_count": result.redaction_count,
        "draft_id": review.draft_id,
        "revision": review.latest_revision,
        "form": {"schema_id": schema.schema_id, "version": schema.version, "title": schema.display_title},
        "transcript": {"text": result.transcript.text, "provider": result.transcript.provider, "model": result.transcript.model, "language": result.transcript.language},
        "draft": {
            "proposed_fields": result.draft.proposed_fields,
            "evidence_refs": {key: list(value) for key, value in result.draft.evidence_refs.items()},
            "needs_confirmation": result.draft.needs_confirmation,
            "model_version": result.draft.model_version,
            "sections": sections_response(schema, result.draft.proposed_fields, result.draft.evidence_refs, review.candidates),
        },
        "transcript_verification": gateway_verification_response(result.transcript_gateway_verification),
        "draft_verification": gateway_verification_response(result.draft_gateway_verification),
        # LLM이 용어 힌트로 이미 초안에 반영한 교정이다. 후보 승인 절차를 거치지 않았으므로 확인이 필요하다.
        "llm_applied_term_hints": [dict(item) for item in result.llm_applied_term_hints],
        "report_validation": report_validation_response(result.report_validation),
        "numeric_verification": numeric_verification_response(result.numeric_findings),
    }
