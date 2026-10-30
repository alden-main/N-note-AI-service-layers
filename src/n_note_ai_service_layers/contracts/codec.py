"""불변 dataclass 계약을 저장소·API용 JSON 사전과 상호 변환한다(pydantic ``TypeAdapter``)."""

from __future__ import annotations

from datetime import datetime
from functools import cache
from typing import Annotated, Any, TypeVar

from pydantic import PlainSerializer, TypeAdapter

T = TypeVar("T")

# pydantic 기본 직렬화는 UTC를 ``Z``로 쓰지만, 기존 저장본·API 응답과 같게 ``isoformat()``(``+00:00``)을 유지한다.
IsoDatetime = Annotated[datetime, PlainSerializer(datetime.isoformat, when_used="json")]


@cache
def _adapter(cls: type[T]) -> TypeAdapter[T]:
    return TypeAdapter(cls)


def to_json_dict(value: Any) -> dict[str, Any]:
    """중첩 dataclass·tuple·Enum·datetime을 JSON 호환 사전으로 바꾼다."""
    return _adapter(type(value)).dump_python(value, mode="json")


def from_json_dict(cls: type[T], raw: dict[str, Any]) -> T:
    """``to_json_dict`` 결과(또는 같은 형식의 사전)를 ``cls`` 인스턴스로 되살린다."""
    return _adapter(cls).validate_python(raw)
