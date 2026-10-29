"""전사문에서 명시적인 개인정보만 보호 토큰으로 치환한다."""

from __future__ import annotations

import re
from collections.abc import Iterable

from n_note_ai_service_layers.contracts.privacy import NamedEntityRecognizer, RedactedText

# 인구 비중이 높은 성씨만 둔다. 드문 성씨(어·은·모 등)는 일반 단어(어머님·모든)와 겹쳐 오탐이 크다.
_SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제탁국"
_NAME = rf"[{_SURNAMES}][가-힣]{{1,2}}"
# '님'·'씨' 앞에 오지만 사람 이름이 아닌 호칭·직함이다.
_HONORIFIC_STOPWORDS = frozenset({
    "선생", "선배", "주치의", "교수", "원장", "과장", "부장", "실장", "팀장", "주임", "대리", "반장", "회장", "사장",
    "고객", "기사", "여사", "도우미", "이모", "고모", "이모부", "고모부", "조무사", "수간호", "보호", "환자", "간병",
    "장모", "장인", "사모", "부모", "조카", "아드", "도련", "형", "누", "동생",
})
# 한글로 읽은 숫자다(공일공 일이삼사 오육칠팔).
_SPOKEN_DIGIT = "[공영일이삼사오육칠팔구]"
_PROTECTED_TOKEN = re.compile(r"\[\[[A-Z_]+_\d+\]\]")


class TranscriptRedactor:
    """낮은 오탐을 우선해 명시적 식별자만 비식별화한다.

    입력: STT가 생성한 원문 전사 문자열과, 요청에서 이미 알고 있는 환자·내원 식별자이다.
    출력: 외부 Gateway·LLM용 보호 토큰 치환본과 치환 수이다.

    규칙은 문맥 없이 말한 이름("영희가 열이 나요")을 잡지 못한다. ``recognizer``(로컬 NER,
    ``N_NOTE_NER_MODEL``)를 주면 규칙 치환 뒤 남은 사람 이름을 한 번 더 가린다.
    NER이 실패하면 예외를 그대로 올려, 덜 가린 전사문이 외부로 나가지 않게 한다.
    """
    _RULES = (
        ("RESIDENT_ID", re.compile(r"(?<!\d)\d{6}-?[1-4]\d{6}(?!\d)")),
        ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ("PHONE", re.compile(r"(?<!\d)(?:01[016789]|0\d{1,2})[- ]?\d{3,4}[- ]?\d{4}(?!\d)")),
        ("PHONE", re.compile(rf"(?<![가-힣])공\s?일\s?[공영일육칠팔구](?:[\s-]?{_SPOKEN_DIGIT}){{7,8}}")),
    )
    _LABELED_RULES = (
        ("LABELED_ID", re.compile(r"(?P<label>(?:환자번호|등록번호|병록번호|MRN)\s*[:#：]\s*)(?P<value>[A-Za-z0-9-]{4,})")),
        ("LABELED_PERSON", re.compile(r"(?P<label>(?:환자명|보호자명|성명|이름)\s*[:：]\s*)(?P<value>[가-힣]{2,4})")),
        ("DATE_OF_BIRTH", re.compile(r"(?P<label>생년월일\s*[:：]\s*)(?P<value>\d{4}[./-]\d{1,2}[./-]\d{1,2})")),
        # 구술에는 콜론이 없다: "환자 성함은 홍길동이고", "이름은 김영희입니다".
        ("LABELED_PERSON", re.compile(rf"(?P<label>(?:성함|이름|환자명|보호자명)\s?(?:은|는|이요|:)?\s+)(?P<value>{_NAME})")),
        ("LABELED_ID", re.compile(r"(?P<label>(?:환자번호|등록번호|병록번호)\s?(?:은|는)?\s+)(?P<value>[A-Za-z0-9-]{4,})")),
    )
    _HONORIFIC = re.compile(rf"(?<![가-힣])(?P<value>{_NAME})(?P<suffix>\s?(?:님|씨))")

    def __init__(self, recognizer: NamedEntityRecognizer | None = None) -> None:
        self.recognizer = recognizer

    def redact(self, text: str, known_identifiers: Iterable[str] = ()) -> RedactedText:
        """명시적 식별자만 순차적으로 치환하고 임상 용어는 보존한다.

        입력: 원문 전사 문자열과 그대로 가려야 할 식별자(환자 ID·내원 ID 등)이다.
        출력: ``[[TYPE_n]]`` 보호 토큰을 포함한 비식별 문자열이다.
        """
        counts: dict[str, int] = {}

        def token(kind: str) -> str:
            counts[kind] = counts.get(kind, 0) + 1
            return f"[[{kind}_{counts[kind]}]]"

        redacted = text
        # 짧은 값은 일반 수치와 겹치므로 4자 이상인 식별자만 가린다.
        for identifier in sorted({item.strip() for item in known_identifiers if item and len(item.strip()) >= 4}, key=len, reverse=True):
            redacted = re.sub(rf"(?<![0-9A-Za-z]){re.escape(identifier)}(?![0-9A-Za-z])", lambda _: token("KNOWN_ID"), redacted)
        for kind, pattern in self._RULES:
            redacted = pattern.sub(lambda _, kind=kind: token(kind), redacted)
        for kind, pattern in self._LABELED_RULES:
            redacted = pattern.sub(lambda match, kind=kind: f"{match.group('label')}{token(kind)}", redacted)

        def honorific(match: re.Match[str]) -> str:
            value = match.group("value")
            if value in _HONORIFIC_STOPWORDS or value[1:] in _HONORIFIC_STOPWORDS:
                return match.group(0)
            return f"{token('PERSON')}{match.group('suffix')}"
        redacted = self._HONORIFIC.sub(honorific, redacted)
        if self.recognizer is not None:
            redacted = self._redact_entities(redacted, token)
        return RedactedText(redacted, sum(counts.values()))

    def _redact_entities(self, text: str, token) -> str:
        """NER 범위를 뒤에서부터 치환한다. 보호 토큰과 겹치거나 호칭·직함인 범위는 건너뛴다."""
        protected = [(match.start(), match.end()) for match in _PROTECTED_TOKEN.finditer(text)]
        spans = sorted({(span.start, span.end) for span in self.recognizer.detect(text)}, reverse=True)
        replacements: list[tuple[int, int]] = []
        for start, end in spans:
            value = text[start:end].strip()
            if not (0 <= start < end <= len(text)) or len(value) < 2 or value in _HONORIFIC_STOPWORDS:
                continue
            if any(start < p_end and p_start < end for p_start, p_end in protected):
                continue
            if any(start < r_end and r_start < end for r_start, r_end in replacements):
                continue
            replacements.append((start, end))
        # 토큰 번호가 문장 앞쪽부터 매겨지도록 앞에서부터 번호를 받고, 치환은 뒤에서부터 한다.
        tokens = {span: token("PERSON") for span in sorted(replacements)}
        for start, end in replacements:
            text = text[:start] + tokens[(start, end)] + text[end:]
        return text
