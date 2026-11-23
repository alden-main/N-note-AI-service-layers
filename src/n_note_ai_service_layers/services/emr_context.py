"""bridge가 보낸 EMR JSON을 LLM 참고 문맥으로 정리한다: 검증 → 식별 정보 제거 → 평탄화 → 값 비식별화.

EMR은 양식 필드에 매칭하지 않는다. 모든 항목을 경로(``bloodPressureList[1]``)와 값 문자열로 펼쳐 LLM에 참고 문맥으로 주고,
LLM이 양식에 필요한 내용을 골라 쓴다. 근거는 ``emr:<경로>``로 남긴다.

- ``parse_emr``: 요청 단계(API)에서 크기·깊이·항목 수를 검사한다. 원본은 저장하지 않고 작업 메모리로만 넘긴다.
- ``prepare_emr``: 작업 단계(비식별화와 같은 단계)에서 실행한다. 식별 정보 키(환자·의사 이름, ID, 연락처 등)의 값은
  보내지 않고, 그 값(특히 이름)은 남은 EMR 값과 전사문 비식별화에 알려진 식별자로 넘긴다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor
from n_note_ai_service_layers.settings import DEFAULT_EMR_IDENTIFIER_KEYS

# 식별 정보 키다(대소문자·구분 기호를 뺀 정규화 이름과 비교한다). 값은 LLM에 보내지 않는다.
# ``N_NOTE_EMR_IDENTIFIER_KEYS``로 바꿀 수 있다. 약 이름(medName)처럼 임상 정보인 *Name은 넣지 않는다(정확히 같은 키만 뺀다).
DEFAULT_IDENTIFIER_KEYS = DEFAULT_EMR_IDENTIFIER_KEYS
# 이 정규화 이름이 들어간 식별 정보 키의 값은 사람 이름으로 보고 길이와 관계없이 가린다.
_NAME_KEY_MARKERS = ("name", "성명", "이름", "환자명", "보호자", "담당의", "간호사명", "의사명")
_KEY_SEPARATORS = re.compile(r"[\s_\-.]+")

DEFAULT_MAX_BYTES = 64 * 1024
DEFAULT_MAX_ENTRIES = 500
DEFAULT_MAX_DEPTH = 6

Scalar = str | int | float | bool | None


class EmrInputError(ValueError):
    """EMR 입력이 JSON 객체가 아니거나 상한(크기·깊이·항목 수)을 넘었다. 메시지에 값은 넣지 않는다."""


@dataclass(frozen=True)
class EmrEntry:
    """평탄화한 EMR 항목 하나다. ``key``는 원본 경로(``bloodSugarList[1]``), ``value``는 비식별화한 값 문자열이다."""

    key: str
    value: str


@dataclass(frozen=True)
class EmrContext:
    """LLM 참고 문맥으로 쓸 EMR이다. 식별 정보 키의 값은 들어 있지 않다."""

    entries: tuple[EmrEntry, ...] = ()
    # 보내지 않은 식별 정보 키 수와 값에서 가린 식별자 수다(로그·추적용, 값은 남기지 않는다).
    removed_identifier_keys: int = 0
    replacements: int = 0

    def payload(self) -> list[dict[str, str]]:
        return [{"key": entry.key, "value": entry.value} for entry in self.entries]

    def source_texts(self) -> tuple[str, ...]:
        """수치 대조 출처다. EMR에서 그대로 옮긴 수치를 '출처 없는 수치'로 보지 않는다."""
        return tuple(f"{entry.key}: {entry.value}" for entry in self.entries)


@dataclass(frozen=True)
class EmrIdentifiers:
    """EMR 식별 정보 키에서 꺼낸 값이다. 전사문과 EMR 값의 비식별화에 알려진 식별자로 쓴다."""

    names: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()


def parse_emr(raw: str | bytes | dict[str, Any] | None, max_bytes: int = DEFAULT_MAX_BYTES, max_entries: int = DEFAULT_MAX_ENTRIES,
              max_depth: int = DEFAULT_MAX_DEPTH) -> dict[str, Any] | None:
    """EMR JSON을 읽고 상한을 검사한다. 비었으면 ``None``(EMR 없음)이다."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        data: Any = raw
        size = len(json.dumps(raw, ensure_ascii=False).encode("utf-8"))
    else:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not text.strip():
            return None
        size = len(text.encode("utf-8"))
        if size > max_bytes:
            raise EmrInputError(f"emr exceeds {max_bytes} bytes")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EmrInputError("emr must be a JSON object") from exc
    if size > max_bytes:
        raise EmrInputError(f"emr exceeds {max_bytes} bytes")
    if not isinstance(data, dict):
        raise EmrInputError("emr must be a JSON object")
    leaves = 0
    for depth, _value in _walk_leaves(data, 1):
        leaves += 1
        if depth > max_depth:
            raise EmrInputError(f"emr nesting exceeds depth {max_depth}")
        if leaves > max_entries:
            raise EmrInputError(f"emr has more than {max_entries} values")
    return data or None


def _walk_leaves(value: Any, depth: int) -> Iterator[tuple[int, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EmrInputError("emr keys must be strings")
            yield from _walk_leaves(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_leaves(item, depth + 1)
    elif value is None or isinstance(value, str | int | float | bool):
        yield depth, value
    else:  # json.loads가 만들지 않는 값(사전으로 직접 받은 경우)
        raise EmrInputError("emr values must be JSON scalars, arrays or objects")


def normalized_key(key: str) -> str:
    return _KEY_SEPARATORS.sub("", key).casefold()


def split_identifiers(data: dict[str, Any], identifier_keys: Iterable[str] = DEFAULT_IDENTIFIER_KEYS) -> tuple[dict[str, Any], EmrIdentifiers, int]:
    """식별 정보 키를 (중첩·배열 안까지) 빼고 그 값을 모은다. 반환: (남은 EMR, 꺼낸 값, 뺀 키 수)."""
    denied = {normalized_key(key) for key in identifier_keys}
    names: list[str] = []
    identifiers: list[str] = []
    removed = 0

    def collect(key: str, value: Any) -> None:
        for _depth, leaf in _walk_leaves(value, 0):
            if leaf is None or isinstance(leaf, bool):
                continue
            text = str(leaf).strip()
            if text:
                (names if any(marker in key for marker in _NAME_KEY_MARKERS) else identifiers).append(text)

    def strip(value: Any) -> Any:
        nonlocal removed
        if isinstance(value, dict):
            kept: dict[str, Any] = {}
            for key, item in value.items():
                normal = normalized_key(key)
                if normal in denied:
                    removed += 1
                    collect(normal, item)
                else:
                    kept[key] = strip(item)
            return kept
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(data), EmrIdentifiers(tuple(dict.fromkeys(names)), tuple(dict.fromkeys(identifiers))), removed


def flatten(data: dict[str, Any]) -> list[tuple[str, str]]:
    """중첩 EMR을 (경로, 값) 목록으로 펼친다. 값이 없는(null·빈 문자열) 항목은 뺀다.

    스칼라만 가진 객체 배열(측정 기록 목록)은 원소 하나를 한 항목으로 묶는다(``sbp=114, dbp=101``). 측정값과 측정 시각이
    떨어지지 않게 하고 항목 수를 줄이기 위해서다.
    """
    entries: list[tuple[str, str]] = []

    def visit(path: str, value: Any) -> None:
        if isinstance(value, dict):
            if value and all(not isinstance(item, dict | list) for item in value.values()) and "[" in path.rsplit(".", 1)[-1]:
                record = _record(value)
                if record:
                    entries.append((path, record))
                return
            for key, item in value.items():
                visit(f"{path}.{key}" if path else key, item)
        elif isinstance(value, list):
            if value and all(not isinstance(item, dict | list) for item in value):
                values = [_scalar(item) for item in value if _scalar(item)]
                if values:
                    entries.append((path, ", ".join(values)))
                return
            for index, item in enumerate(value):
                visit(f"{path}[{index}]", item)
        else:
            text = _scalar(value)
            if text:
                entries.append((path, text))

    visit("", data)
    return entries


def _record(value: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_scalar(item)}" for key, item in value.items() if _scalar(item))


def _scalar(value: Scalar) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def prepare_emr(
    data: dict[str, Any] | None, redactor: TranscriptRedactor, known_identifiers: Iterable[str] = (),
    identifier_keys: Iterable[str] = DEFAULT_IDENTIFIER_KEYS,
) -> tuple[EmrContext | None, EmrIdentifiers]:
    """EMR을 LLM 참고 문맥으로 만든다. 식별 정보 키는 빼고, 남은 문자열 값은 전사문과 같은 규칙(NER 포함)으로 가린다.

    반환한 ``EmrIdentifiers``는 전사문 비식별화에도 쓴다(EMR의 환자 이름이 구술에 나오면 가린다).
    비식별화가 실패하면 예외를 그대로 올린다(덜 가린 EMR이 외부로 나가지 않게).
    """
    if not data:
        return None, EmrIdentifiers()
    remaining, found, removed = split_identifiers(data, identifier_keys)
    known = (*known_identifiers, *found.identifiers)
    entries: list[EmrEntry] = []
    replacements = 0
    for key, value in flatten(remaining):
        redacted = redactor.redact(value, known, found.names)
        replacements += redacted.replacement_count
        entries.append(EmrEntry(key, redacted.text))
    return EmrContext(tuple(entries), removed, replacements), found
