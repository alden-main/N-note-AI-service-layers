"""Kiwi 형태소 분석기를 도메인 서비스와 분리하는 Adapter다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MorphToken:
    """원문 코드 포인트 위치를 보존한 형태소 분석 단위다."""

    source_text: str
    analysis_form: str
    tag: str
    start: int
    end: int


class MorphologyAnalyzer:
    """Kiwi 직접 호출을 한 곳에 한정하는 형태소 분석 Adapter다."""

    def __init__(self) -> None:
        from kiwipiepy import Kiwi

        self._kiwi = Kiwi()

    def analyze(self, text: str) -> list[MorphToken]:
        """Kiwi 토큰을 유니코드 코드 포인트 오프셋 기반 도메인 모델로 변환한다."""
        tokens = self._kiwi.tokenize(text)
        result = [
            MorphToken(
                source_text=text[token.start : token.start + token.len],
                analysis_form=token.form,
                tag=token.tag,
                start=token.start,
                end=token.start + token.len,
            )
            for token in tokens
        ]
        if any(token.start < 0 or token.end < token.start or token.end > len(text) for token in result):
            raise ValueError("형태소 분석기가 유효하지 않은 원문 위치를 반환했습니다.")
        return result
