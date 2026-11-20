"""초안 수치 대조 검증 결과 계약이다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# not_in_source: 초안의 수치가 출처(전사문·사실)에 없다. field_id가 있고 offset은 그 필드 값(표시 문자열) 기준이다.
# missing_from_draft: 전사문에서 구술한 수치가 초안 어디에도 없다. field_id는 None이고 offset은 전사문(비식별화본) 기준이다.
NumericFindingReason = Literal["not_in_source", "missing_from_draft"]


@dataclass(frozen=True)
class NumericFinding:
    """간호사가 확인해야 하는 수치다(``reason``으로 종류를 구분한다)."""

    finding_id: str
    field_id: str | None
    start_offset: int
    end_offset: int
    text: str
    reason: NumericFindingReason = "not_in_source"

    def describe(self) -> str:
        """확정본·평가 결과에 남기는 한 줄 설명이다."""
        if self.reason == "missing_from_draft":
            return f"transcript: {self.text} (missing_from_draft)"
        return f"{self.field_id}: {self.text}"
