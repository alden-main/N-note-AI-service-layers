"""원천 임상 사실을 활성 스키마 기준의 안전한 문맥 스냅샷으로 조립한다."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from n_note_ai_service_layers.contracts.clinical_context import Availability, ClinicalContextQuery, ClinicalContextSnapshot, ClinicalFact
from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus


class ClinicalContextAssembler:
    """사실의 충돌·신선도·누락 상태를 반영해 임상 문맥을 확정한다.

    입력: 조회 요청, 정규화된 임상 사실 목록 및 활성 양식 스키마이다.
    출력: 스키마가 요구한 사실만 포함하는 ``ClinicalContextSnapshot``이다.
    """
    def build(self, query: ClinicalContextQuery, facts: list[ClinicalFact], schema: FormSchema) -> ClinicalContextSnapshot:
        """활성 스키마에 맞춰 사실을 선택하고 가용 상태를 판정한다.

        입력: ``query``는 기준 시각과 목적을, ``facts``는 정규화된 관찰값을,
        ``schema``는 요구 사실과 신선도 기준을 제공한다.
        출력: 누락·충돌·오래된 사실을 표시한 ``ClinicalContextSnapshot``이며,
        활성 스키마가 요청과 맞지 않으면 예외를 발생시킨다.
        """
        if schema.status is not SchemaStatus.ACTIVE or schema.schema_id != query.schema_id or schema.purpose != query.purpose:
            raise PermissionError("an active schema matching the request is required")
        if query.schema_version and query.schema_version != schema.version:
            raise ValueError("requested schema version does not match active schema")
        grouped: dict[str, list[ClinicalFact]] = defaultdict(list)
        for fact in facts:
            if fact.fact_id in schema.requested_fact_ids and (fact.observed_at is None or fact.observed_at <= query.as_of):
                grouped[fact.fact_id].append(fact)
        selected: list[ClinicalFact] = []
        for fact_id in schema.requested_fact_ids:
            candidates = grouped.get(fact_id, [])
            if not candidates:
                selected.append(ClinicalFact(fact_id, None, None, None, None, Availability.UNAVAILABLE, "no observation"))
                continue
            candidates.sort(key=lambda item: item.observed_at or query.as_of, reverse=True)
            latest = candidates[0]
            same_time = [item for item in candidates if item.observed_at == latest.observed_at]
            if len({repr(item.value) for item in same_time}) > 1:
                selected.append(ClinicalFact(fact_id, None, None, latest.observed_at, latest.source, Availability.CONFLICT, "conflicting observations"))
                continue
            minutes = schema.freshness_windows_minutes.get(fact_id)
            if minutes is not None and latest.observed_at and latest.observed_at < query.as_of - timedelta(minutes=minutes):
                selected.append(ClinicalFact(fact_id, None, latest.display, latest.observed_at, latest.source, Availability.STALE, "outside freshness window"))
            else:
                selected.append(latest)
        return ClinicalContextSnapshot(schema.schema_id, schema.version, tuple(sorted(selected, key=lambda f: f.fact_id)), query.trace_id)
