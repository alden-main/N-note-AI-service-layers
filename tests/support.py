"""여러 테스트가 공유하는 중립 양식 스키마다(특정 업무 양식에 묶이지 않는다)."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus

SCORE_FIELD = "assessment.pain_score"
DERIVED_FIELD = "summary.derived_level"


def sample_form_schema(status: SchemaStatus = SchemaStatus.DRAFT) -> FormSchema:
    """LLM이 채우는 정수 필드 2개와 LLM에 노출하지 않는 파생 필드 1개를 가진 양식이다."""
    fields = (
        FormField(SCORE_FIELD, "integer", required=True, fact_id=SCORE_FIELD),
        FormField("assessment.mobility_score", "integer", fact_id="assessment.mobility_score"),
        FormField(DERIVED_FIELD, "string", llm_allowed=False),
    )
    return FormSchema("sample-note", "sample-note/v1", "SAMPLE_NOTE", (FormSection("assessment", fields),), status=status)
