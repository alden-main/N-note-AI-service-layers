"""용어 검색 전에 입력 표기를 결정적으로 정규화한다."""

from __future__ import annotations

import re
import unicodedata


class TextNormalizer:
    """전각·반각, 대소문자와 공백 차이를 검색 키에서 제거한다."""

    def normalize(self, text: str) -> str:
        """입력 문자열을 비교 가능한 검색 키로 반환한다."""
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return re.sub(r"\s+", "", normalized)
