"""보고서 검증 MCP의 입력·출력과 내부 중간표현 계약을 정의한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Status = Literal["satisfied", "missing", "insufficient", "uncertain"]


class InputError(ValueError):
    """호출자가 수정할 수 있는 입력 계약 위반을 나타낸다."""


@dataclass(frozen=True)
class ElementSpec:
    """기준 충족에 필요한 세부 작성 요소의 정의다."""

    element_id: str
    description: str | None = None


@dataclass(frozen=True)
class CriterionSpec:
    """검증 기준 하나와 요구되는 세부 요소 목록이다."""

    criterion_id: str
    description: str
    required_elements: tuple[ElementSpec, ...]
    category: str | None = None


@dataclass(frozen=True)
class ValidationStandard:
    """schema별 원본 기준을 변환한 공통의 읽기 전용 검증 기준이다."""

    standard_id: str
    standard_version: str
    status: str
    result_policy: str
    mutation_policy: str
    criteria: tuple[CriterionSpec, ...]


@dataclass(frozen=True)
class Citation:
    """보고서 원문에서 확인 가능한 근거 구간이다."""

    start: int
    end: int
    quote: str


@dataclass(frozen=True)
class CandidateElement:
    """외부 의미 분석기가 제출한 세부 요소 후보다."""

    element_id: str
    status: Status
    reason: str | None = None
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class CandidateCriterion:
    """외부 의미 분석기가 제출한 기준 단위 후보다."""

    criterion_id: str
    status: Status
    reason: str | None = None
    citations: tuple[Citation, ...] = ()
    elements: tuple[CandidateElement, ...] = ()


@dataclass(frozen=True)
class SemanticAssessment:
    """신뢰하지 않는 외부 의미 분석 결과의 전송 단위다."""

    criteria: tuple[CandidateCriterion, ...] = ()


@dataclass(frozen=True)
class Finding:
    """기준 또는 세부 요소에 대한 정규화된 검증 결과다."""

    item_id: str
    status: Status
    reason: str | None
    citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """MCP 응답에 사용할 JSON 호환 사전으로 변환한다."""
        return {
            "id": self.item_id,
            "status": self.status,
            "reason": self.reason,
            "citations": [citation.__dict__ for citation in self.citations],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CriterionResult:
    """기준 전체 결과와 그 기준에 속한 요소별 결과를 묶는다."""

    criterion_id: str
    description: str
    finding: Finding
    elements: tuple[Finding, ...]

    def as_dict(self) -> dict[str, Any]:
        """MCP 응답에 사용할 JSON 호환 사전으로 변환한다."""
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "finding": self.finding.as_dict(),
            "elements": [element.as_dict() for element in self.elements],
        }


@dataclass(frozen=True)
class EvaluationResult:
    """사람 검토를 전제로 한 도구의 최종 읽기 전용 결과다."""

    standard_id: str
    standard_version: str
    standard_status: str
    result_policy: str
    mutation_policy: str
    criteria: tuple[CriterionResult, ...]
    review_required: bool = True
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """요약을 포함한 외부 응답 형식으로 변환한다."""
        counts = {status: 0 for status in ("satisfied", "missing", "insufficient", "uncertain")}
        for result in self.criteria:
            counts[result.finding.status] += 1
            for element in result.elements:
                counts[element.status] += 1
        return {
            "standard": {
                "standard_id": self.standard_id,
                "standard_version": self.standard_version,
                "status": self.standard_status,
                "result_policy": self.result_policy,
                "mutation_policy": self.mutation_policy,
            },
            "review_required": self.review_required,
            "warnings": list(self.warnings),
            "summary": counts,
            "criteria": [criterion.as_dict() for criterion in self.criteria],
        }


@dataclass(frozen=True)
class ToolError:
    """공통 Gateway/MCP 오류 계약을 표현한다."""

    code: str
    message: str
    retryable: bool = False

    def as_dict(self) -> dict[str, Any]:
        """민감정보를 포함하지 않는 표준 오류 사전을 반환한다."""
        return {"error": {"code": self.code, "message": self.message, "retryable": self.retryable}}
