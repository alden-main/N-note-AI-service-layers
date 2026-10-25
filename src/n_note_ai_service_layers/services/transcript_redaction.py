"""전사문에서 명시적인 개인정보만 보호 토큰으로 치환한다."""

from __future__ import annotations

import re

from n_note_ai_service_layers.contracts.privacy import RedactedText


class TranscriptRedactor:
    """낮은 오탐을 우선해 명시적 식별자만 비식별화한다.

    입력: STT가 생성한 원문 전사 문자열이다.
    출력: 외부 Gateway·LLM용 보호 토큰 치환본과 치환 수이다.
    """
    _RULES = (
        ("RESIDENT_ID", re.compile(r"(?<!\d)\d{6}-?[1-4]\d{6}(?!\d)")),
        ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ("PHONE", re.compile(r"(?<!\d)(?:01[016789]|0\d{1,2})[- ]?\d{3,4}[- ]?\d{4}(?!\d)")),
    )
    _LABELED_RULES = (
        ("LABELED_ID", re.compile(r"(?P<label>(?:환자번호|등록번호|병록번호|MRN)\s*[:#：]\s*)(?P<value>[A-Za-z0-9-]{4,})")),
        ("LABELED_PERSON", re.compile(r"(?P<label>(?:환자명|보호자명|성명|이름)\s*[:：]\s*)(?P<value>[가-힣]{2,4})")),
        ("DATE_OF_BIRTH", re.compile(r"(?P<label>생년월일\s*[:：]\s*)(?P<value>\d{4}[./-]\d{1,2}[./-]\d{1,2})")),
    )

    def redact(self, text: str) -> RedactedText:
        """명시적 식별자만 순차적으로 치환하고 임상 용어는 보존한다.

        입력: 원문 전사 문자열이다.
        출력: ``[[TYPE_n]]`` 보호 토큰을 포함한 비식별 문자열이다.
        """
        counts: dict[str, int] = {}
        redacted = text
        for kind, pattern in self._RULES:
            def replace(match: re.Match[str]) -> str:
                counts[kind] = counts.get(kind, 0) + 1
                return f"[[{kind}_{counts[kind]}]]"
            redacted = pattern.sub(replace, redacted)
        for kind, pattern in self._LABELED_RULES:
            def replace_labeled(match: re.Match[str]) -> str:
                counts[kind] = counts.get(kind, 0) + 1
                return f"{match.group('label')}[[{kind}_{counts[kind]}]]"
            redacted = pattern.sub(replace_labeled, redacted)
        return RedactedText(redacted, sum(counts.values()))
