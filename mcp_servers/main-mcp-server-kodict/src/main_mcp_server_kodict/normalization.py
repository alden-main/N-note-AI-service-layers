"""원본 표시값을 보존하면서 검색 키만 정규화하는 모듈이다."""

import re


def normalize_surface(value: str) -> str:
    """정확 조회용으로 공백과 어휘의 괄호 의미 번호만 제거한다."""
    return re.sub(r"\s+", "", re.sub(r"\(\d+\)", "", value)).strip()

