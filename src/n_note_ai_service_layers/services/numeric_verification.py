"""초안의 수치가 전사문(또는 LLM에 준 사실)에 실제로 있는지 대조하는 결정적 검증이다.

LLM은 '혈압 백에 육십'을 160/100으로 쓰거나, 구술에 없는 기록 시각을 기준 시각으로 채우는 등
수치를 바꿀 수 있다. 수치 오류는 용어 사전으로 잡히지 않으므로 별도로 대조한다.

- 출처 수치: 아라비아 숫자, 한자어 수사(삼십팔 점 이, 백이십, 칠오일), 고유어 수사(일곱 시, 한 시간)를
  모두 숫자로 읽는다. '오후/저녁/밤 N시'는 N+12도 허용한다. 출처 쪽은 넓게 읽어 오탐을 줄인다.
- 초안 수치: 아라비아 숫자만 대조한다. 한 자리 정수(0~9)는 점수·횟수 표기와 구분하기 어려워 제외한다.
- 날짜·일시 항목의 날짜는 한 덩어리로 보고, 구술 기준일(reference_time) 전후(그제~내일)이거나
  연·월·일이 모두 출처에 있으면 허용한다.

결과는 간호사 확인 대상(``NumericFinding``)이며, 초안을 자동 수정하지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.contracts.numeric_verification import NumericFinding
from n_note_ai_service_layers.services.draft_document import format_value

SINO_DIGITS = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
SINO_UNITS = {"십": 10, "백": 100, "천": 1000}
SINO_CHARS = "".join(SINO_DIGITS) + "".join(SINO_UNITS) + "만"
NATIVE_TENS = (("스물", 20), ("스무", 20), ("서른", 30), ("마흔", 40), ("쉰", 50), ("열", 10))
NATIVE_ONES = (("하나", 1), ("한", 1), ("둘", 2), ("두", 2), ("셋", 3), ("세", 3), ("넷", 4), ("네", 4), ("다섯", 5), ("여섯", 6), ("일곱", 7), ("여덟", 8), ("아홉", 9))
PM_WORDS = ("오후", "저녁", "밤")
PRIVACY_TOKEN = re.compile(r"\[\[[A-Z]+(?:_[A-Z]+)*_\d+\]\]")
ARABIC_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?")
DATE_VALUE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
WORD = re.compile(r"[^\s,.;:!?()\[\]/]+")


def _number(text: str) -> Decimal | None:
    try:
        return Decimal(text).normalize()
    except InvalidOperation:
        return None


def _sino(run: str) -> tuple[int | None, int, bool]:
    """한자어 수사 문자열을 읽는다. 반환: (값, 가장 작은 자리, 자리 단위 포함 여부).

    단위가 없으면 '칠오일'처럼 숫자를 한 자리씩 읽은 것으로 본다.
    """
    if not any(char in SINO_UNITS or char == "만" for char in run):
        return int("".join(str(SINO_DIGITS[char]) for char in run)), 1, False
    total, section, digit, last_unit, smallest = 0, 0, None, 10 ** 5, 1
    for char in run:
        if char in SINO_DIGITS:
            if digit is not None:
                return None, 1, True
            digit = SINO_DIGITS[char]
            smallest = 1
        elif char in SINO_UNITS:
            unit = SINO_UNITS[char]
            if unit >= last_unit:
                return None, 1, True
            section += (1 if digit is None else digit) * unit
            digit, last_unit, smallest = None, unit, unit
        else:  # 만
            total += ((section + (digit or 0)) or 1) * 10000
            section, digit, last_unit, smallest = 0, None, 10 ** 5, 10000
    return total + section + (digit or 0), smallest if digit is None else 1, True


def _native(word: str) -> tuple[int, int] | None:
    """고유어 수사 접두를 읽는다. 반환: (값, 소비한 글자 수)."""
    value, used = 0, 0
    for prefix, number in NATIVE_TENS:
        if word.startswith(prefix):
            value, used = number, len(prefix)
            break
    for prefix, number in NATIVE_ONES:
        if word.startswith(prefix, used):
            return value + number, used + len(prefix)
    return (value, used) if used else None


@dataclass(frozen=True)
class _Spoken:
    value: int
    smallest: int
    has_unit: bool
    whole_word: bool
    rest: str


def _spoken(word: str) -> _Spoken | None:
    """단어 앞부분의 수사를 읽는다. 고유어(일곱, 열두)를 한자어(일)보다 먼저 본다."""
    native = _native(word)
    if native and (native[1] > 1 or len(word) == 1 or word[native[1]:native[1] + 1] in {"시", "번", "개", "명", "살", "달", "시간", "방울", "알"}):
        value, used = native
        return _Spoken(value, 1, True, used == len(word), word[used:])
    length = 0
    while length < len(word) and word[length] in SINO_CHARS:
        length += 1
    if not length:
        return None
    value, smallest, has_unit = _sino(word[:length])
    if value is None:
        return None
    return _Spoken(value, smallest, has_unit, length == len(word), word[length:])


def source_numbers(texts: Iterable[str]) -> set[Decimal]:
    """출처 문장에서 숫자로 읽을 수 있는 모든 값을 모은다."""
    numbers: set[Decimal] = {Decimal(0)}
    for raw in texts:
        text = PRIVACY_TOKEN.sub(" ", raw)
        for match in ARABIC_NUMBER.finditer(text):
            value = _number(match.group())
            if value is not None:
                numbers.update({value, Decimal(int(value))})
        words = WORD.findall(text)
        index = 0
        while index < len(words):
            spoken = _spoken(words[index])
            if spoken is None:
                index += 1
                continue
            value = spoken.value
            consumed = 1
            if not spoken.has_unit:
                # '칠오일'은 751이면서 각 숫자일 수도 있다(오일=5, 1). 넓게 허용한다.
                numbers.update(Decimal(int(digit)) for digit in str(value))
            # '구십 이 회', '육십 이세요'처럼 띄어 읽은 낮은 자리 숫자를 합친다.
            if spoken.has_unit and spoken.whole_word and spoken.smallest > 1 and index + 1 < len(words):
                following = _spoken(words[index + 1])
                if following and not following.has_unit and following.value < spoken.smallest and len(str(following.value)) == 1:
                    value += following.value
                    consumed = 2
            numbers.add(Decimal(value))
            if spoken.has_unit and spoken.smallest == 1 and spoken.rest and consumed == 1:
                # '육십이었고'의 '이'는 서술격 조사일 수 있으므로 60도 허용한다.
                numbers.add(Decimal(value - value % 10))
            fraction = _fraction(words, index + consumed - 1)
            if fraction is not None:
                numbers.add(_number(f"{value}.{fraction}"))
            if _is_hour(words, index + consumed - 1) and 1 <= value <= 11 and any(word in PM_WORDS for word in words[max(0, index - 2):index]):
                numbers.add(Decimal(value + 12))
            index += consumed
    return numbers


def _fraction(words: list[str], index: int) -> str | None:
    """'삼십팔 점 이 (도)'의 소수 부분을 읽는다. '육 점이고'처럼 조사가 붙으면 소수가 아니다."""
    current = _spoken(words[index])
    rest = current.rest if current else ""
    if not rest and index + 1 < len(words) and words[index + 1].startswith("점"):
        rest, index = words[index + 1], index + 1
    if rest == "점" and index + 1 < len(words):
        candidate = words[index + 1]
    elif rest.startswith("점") and len(rest) > 1:
        candidate = rest[1:]
    else:
        return None
    if candidate and all(char in SINO_DIGITS for char in candidate):
        return "".join(str(SINO_DIGITS[char]) for char in candidate)
    return None


def _is_hour(words: list[str], index: int) -> bool:
    spoken = _spoken(words[index])
    rest = spoken.rest if spoken else ""
    return rest.startswith("시") or (not rest and index + 1 < len(words) and words[index + 1].startswith("시"))


class NumericVerifier:
    """초안 필드의 수치를 출처 수치와 대조한다."""

    def check(self, schema: FormSchema, fields: dict[str, Any], sources: Iterable[str], reference_time: datetime | None = None) -> tuple[NumericFinding, ...]:
        allowed = source_numbers(sources)
        allowed_dates = self._allowed_dates(reference_time)
        findings: list[NumericFinding] = []
        for field in schema.fields:
            value = fields.get(field.field_id)
            if value is None:
                continue
            text = format_value(value)
            masked = PRIVACY_TOKEN.sub(lambda match: " " * len(match.group()), text)
            skip: set[tuple[int, int]] = set()
            if field.type in {"date", "datetime"}:
                # 날짜는 한 덩어리로 판단한다: 기준일 전후이거나 연·월·일이 모두 출처에 있으면 허용한다.
                for match in DATE_VALUE.finditer(masked):
                    skip.update({match.span(1), match.span(2), match.span(3)})
                    try:
                        written = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    except ValueError:
                        written = None
                    spoken = all(Decimal(int(match.group(index))) in allowed for index in (1, 2, 3))
                    if written not in allowed_dates and not spoken:
                        findings.append(NumericFinding(f"numeric:{field.field_id}:{match.start()}", field.field_id, match.start(), match.end(), match.group()))
            for match in ARABIC_NUMBER.finditer(masked):
                number = _number(match.group())
                if number is None or match.span() in skip:
                    continue
                if number == number.to_integral_value() and 0 <= number <= 9:
                    continue
                if number not in allowed:
                    findings.append(NumericFinding(f"numeric:{field.field_id}:{match.start()}", field.field_id, match.start(), match.end(), match.group()))
        return tuple(findings)

    @staticmethod
    def _allowed_dates(reference_time: datetime | None) -> set[date]:
        if reference_time is None:
            return set()
        today = reference_time.date()
        return {today + timedelta(days=offset) for offset in (-2, -1, 0, 1)}
