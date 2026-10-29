"""전사문을 외부 처리 경계로 안전하게 전달하기 위한 비식별화 계약이다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RedactedText:
    """원문을 외부 처리용 보호 토큰으로 치환한 결과를 표현한다.

    입력: 비식별 전사문과 치환된 항목 수이다.
    출력: Gateway·LLM에 전달 가능한 텍스트이며 원문과 복원 매핑은 포함하지 않는다.
    """
    text: str
    replacement_count: int


@dataclass(frozen=True)
class EntitySpan:
    """NER이 찾은 개체의 원문 범위다. ``label``은 보호 토큰 종류(예: ``PERSON``)다."""
    start: int
    end: int
    label: str
    score: float


class NamedEntityRecognizer(Protocol):
    """전사문에서 개인 식별 개체(사람 이름 등)의 범위를 찾는 로컬 모델 인터페이스이다.

    입력: 규칙 기반 치환을 마친 전사문이다.
    출력: 원문 offset 기준 ``EntitySpan`` 목록이다. 외부로 원문을 보내는 구현은 허용하지 않는다.
    """
    def detect(self, text: str) -> list[EntitySpan]: ...
