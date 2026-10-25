"""욕창 위험도 POC용 양식 스키마를 생성하는 정책 모듈이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.form_schema import BridgeMapping, FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.policies.pressure_injury import BRADEN_FACT_IDS


def pressure_injury_poc_schema(status: SchemaStatus = SchemaStatus.DRAFT, version: str = "pressure-injury-poc/v1") -> FormSchema:
    """욕창 위험도 재평가에 사용할 POC 양식 스키마를 생성한다.

    입력: ``status``는 스키마 상태이고 ``version``은 버전 식별자이다.
    출력: Braden 사실, 계산 필드, 브리지 매핑과 신선도 기준을 포함한 ``FormSchema``이다.
    """
    fields = tuple(FormField(fact_id, "integer", required=True, fact_id=fact_id) for fact_id in BRADEN_FACT_IDS)
    fields += (
        FormField("risk.total_score", "integer", llm_allowed=False),
        FormField("risk.level", "string", llm_allowed=False),
        FormField("prevention.interventions", "array", llm_allowed=False),
    )
    mappings = tuple(BridgeMapping(field.field_id, field.field_id) for field in fields)
    return FormSchema("pressure-injury", version, "PRESSURE_INJURY_REASSESSMENT", (FormSection("braden", fields),), ("pressure-injury-poc/v1",), mappings, status, freshness_windows_minutes={fact_id: 1440 for fact_id in BRADEN_FACT_IDS})
