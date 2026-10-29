"""외부 schema 라이브러리 없이 쓰는 작은 JSON Schema 검사기이다."""

from __future__ import annotations

from typing import Any


def matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Gateway 도구 입력과 LLM 초안 출력 schema에 쓰는 결정적 JSON Schema 부분 집합 검사다.

    ``anyOf``, ``type``, ``enum``, ``properties``, ``required``, ``additionalProperties: false``, ``items``만 본다.
    """
    if "anyOf" in schema:
        return isinstance(schema["anyOf"], list) and any(matches_schema(value, item) for item in schema["anyOf"] if isinstance(item, dict))
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool), "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool), "null": value is None,
    }
    if isinstance(expected, str) and not type_matches.get(expected, True):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list) or any(key not in value for key in required):
            return False
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            return False
        return all(not isinstance(properties.get(key), dict) or matches_schema(item, properties[key]) for key, item in value.items())
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(matches_schema(item, schema["items"]) for item in value)
    return True
