"""EMR 연동 전 전사문만으로 초안을 만드는 양식에 사용하는 빈 임상 저장소이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery, SourceRecord


class TranscriptOnlyRepository:
    """임상 사실을 제공하지 않는 저장소다.

    입력: 조회 요청과 요청 사실 ID이다.
    출력: 항상 빈 목록이며, 사실 ID를 요구하는 양식의 사실은 ``unavailable``로 표시된다.
    """

    def fetch(self, query: ClinicalContextQuery, requested_fact_ids: set[str]) -> list[SourceRecord]:
        return []
