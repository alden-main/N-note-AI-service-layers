"""보고서 검증 MCP 결과를 간호 초안 검토 화면에 전달하는 계약이다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ValidationItemStatus = Literal["satisfied", "missing", "insufficient", "uncertain"]


@dataclass(frozen=True)
class ReportValidationElement:
    """양식 기준의 세부 요소(주로 필수 필드) 검증 결과다."""

    element_id: str
    status: ValidationItemStatus
    field_id: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportValidationCriterion:
    """양식 섹션 또는 양식별 추가 기준 하나의 검증 결과다."""

    criterion_id: str
    description: str
    status: ValidationItemStatus
    elements: tuple[ReportValidationElement, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportValidationResult:
    """Gateway 보고서 검증 단계 결과다. 결과는 사람 검토 전용이며 초안을 바꾸지 않는다."""

    status: Literal["ok", "unavailable", "unsupported", "skipped"]
    criteria: tuple[ReportValidationCriterion, ...] = ()
    summary: dict[str, int] | None = None
    tool_name: str | None = None
    review_required: bool = True
