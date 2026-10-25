"""전사문을 외부 처리 경계로 안전하게 전달하기 위한 비식별화 계약이다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedactedText:
    """원문을 외부 처리용 보호 토큰으로 치환한 결과를 표현한다.

    입력: 비식별 전사문과 치환된 항목 수이다.
    출력: Gateway·LLM에 전달 가능한 텍스트이며 원문과 복원 매핑은 포함하지 않는다.
    """
    text: str
    replacement_count: int
