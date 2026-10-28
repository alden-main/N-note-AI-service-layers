"""초안 수치 대조 검증 결과 계약이다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericFinding:
    """출처에서 찾을 수 없는 초안 수치다. offset은 그 필드 값(표시 문자열) 기준이다."""

    finding_id: str
    field_id: str
    start_offset: int
    end_offset: int
    text: str
    reason: str = "not_in_source"
