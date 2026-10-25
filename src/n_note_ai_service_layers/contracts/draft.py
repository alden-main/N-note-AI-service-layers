"""AI 초안과 결정 규칙 검증 결과를 표현하는 불변 계약이다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DraftResult:
    """AI가 제안한 양식 필드와 근거 정보를 표현한다.

    입력: 제안 필드, 필드별 근거 참조, 확인 필요 여부와 모델·프롬프트 버전이다.
    출력: 검증기에 전달되는 불변 초안 결과이다.
    """
    proposed_fields: dict[str, Any]
    evidence_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    needs_confirmation: bool = False
    model_version: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True)
class ValidationIssue:
    """검증 과정에서 발견한 필드 단위 문제를 표현한다.

    입력: 대상 필드 ID, 문제 코드와 설명 메시지이다.
    출력: ``ValidationResult``에 포함되는 불변 문제 객체이다.
    """
    field_id: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """초안과 결정 규칙을 합친 최종 검증 상태를 표현한다.

    입력: 상태, 확정 필드, 문제 목록 및 확인 필요 여부이다.
    출력: 핸드오프 변환 또는 사용자 검토에 사용할 불변 결과이다.
    """
    status: str
    fields: dict[str, Any]
    issues: tuple[ValidationIssue, ...] = ()
    needs_confirmation: bool = False
