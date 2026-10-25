"""원천 임상 데이터를 서비스 표준 사실 값으로 정규화하는 어댑터이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.clinical_context import Availability, ClinicalFact, SourceRecord


class CodeMappingNormalizer:
    """병원 코드를 명시적 매핑으로 표준 임상 사실로 변환한다.

    입력: 사실 ID별 코드 매핑과 ``SourceRecord`` 목록.
    출력: 알 수 없는 코드와 빈 값의 상태를 보존한 ``ClinicalFact`` 목록.
    """
    def __init__(self, code_maps: dict[str, dict[object, object]] | None = None) -> None:
        self.code_maps = code_maps or {}

    def normalize(self, records: list[SourceRecord]) -> list[ClinicalFact]:
        """원천 레코드를 표준 임상 사실 목록으로 변환한다.

        입력: ``records``는 원천 값·관찰 시각·출처를 담은 레코드 목록이다.
        출력: 매핑 성공 여부에 따라 사용 가능 또는 사용 불가 상태가 지정된 사실 목록이다.
        """
        facts: list[ClinicalFact] = []
        for record in records:
            mapping = self.code_maps.get(record.fact_id)
            if mapping is not None and record.value not in mapping:
                facts.append(ClinicalFact(record.fact_id, None, None, record.observed_at, record.source, Availability.UNAVAILABLE, "unmapped source code"))
                continue
            value = mapping[record.value] if mapping is not None else record.value
            if value is None:
                facts.append(ClinicalFact(record.fact_id, None, None, record.observed_at, record.source, Availability.UNAVAILABLE, "null source value"))
            else:
                facts.append(ClinicalFact(record.fact_id, value, str(value), record.observed_at, record.source, Availability.AVAILABLE))
        return facts
