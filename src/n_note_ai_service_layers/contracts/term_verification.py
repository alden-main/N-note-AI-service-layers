"""전사문과 초안의 검토 전용 용어 검증 계약이다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TermVerificationPhase = Literal["transcript", "draft"]
TermCategory = Literal["standard_korean", "medical_term"]
ReviewStatus = Literal["pending", "accepted", "rejected"]


@dataclass(frozen=True)
class GatewaySuggestion:
    """Gateway가 제공한 검토 전용 표준어 또는 의학용어 제안이다."""

    text: str
    score: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class GatewayTermOccurrence:
    """원문 범위 검증을 통과한 Gateway 후보이며 자동 적용되지 않는다.

    ``field_id``가 있으면 offset은 해당 초안 필드 값 기준이고, 없으면 전사문 기준이다.
    """

    phase: TermVerificationPhase
    tool_name: str
    start_offset: int
    end_offset: int
    input_text: str
    normalized_text: str
    suggestions: tuple[GatewaySuggestion, ...]
    review_status: ReviewStatus = "pending"
    field_id: str | None = None
    category: TermCategory | None = None


@dataclass(frozen=True)
class GatewayTermVerificationResult:
    """Gateway 후보 검토 단계 결과다. 오류와 계약 불일치를 후보로 추측하지 않는다."""

    phase: TermVerificationPhase
    status: Literal["ok", "unavailable", "unsupported"]
    occurrences: tuple[GatewayTermOccurrence, ...] = ()
    tool_names: tuple[str, ...] = ()
    auto_apply: bool = False
