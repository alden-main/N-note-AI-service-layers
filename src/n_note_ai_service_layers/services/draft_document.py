"""초안 필드를 양식 순서의 문서 텍스트로 만들고 필드별 원문 범위를 보존한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection


def ordered_sections(schema: FormSchema) -> tuple[FormSection, ...]:
    """표시 순서가 있으면 그 순서로, 없으면 선언 순서로 섹션을 정렬한다."""
    indexed = list(enumerate(schema.sections))
    indexed.sort(key=lambda item: (item[1].display_order is None, item[1].display_order or 0, item[0]))
    return tuple(section for _, section in indexed)


def ordered_fields(section: FormSection) -> tuple[FormField, ...]:
    """표시 순서가 있으면 그 순서로, 없으면 선언 순서로 필드를 정렬한다."""
    indexed = list(enumerate(section.fields))
    indexed.sort(key=lambda item: (item[1].display_order is None, item[1].display_order or 0, item[0]))
    return tuple(field for _, field in indexed)


def format_value(value: Any) -> str:
    """필드 값을 화면·PDF·검토에 쓰는 문자열로 변환한다."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, (list, tuple)):
        return "\n".join(format_value(item) for item in value if item is not None)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {format_value(item)}" for key, item in value.items() if item is not None)
    return str(value)


def is_filled(value: Any) -> bool:
    return bool(format_value(value).strip())


@dataclass(frozen=True)
class FieldSpan:
    """문서 텍스트 안에서 하나의 필드 값이 차지하는 범위다."""

    field_id: str
    section_id: str
    start: int
    end: int


@dataclass(frozen=True)
class DraftDocument:
    """양식 순서로 렌더링한 초안 텍스트와 필드 값 범위다.

    입력: 활성 양식과 필드 값이다.
    출력: Gateway 검토·보고서 검증·PDF가 공통으로 사용하는 텍스트와 범위 목록이다.
    """

    text: str
    spans: tuple[FieldSpan, ...]

    @classmethod
    def build(cls, schema: FormSchema, fields: dict[str, Any]) -> "DraftDocument":
        parts: list[str] = []
        spans: list[FieldSpan] = []
        length = 0

        def append(chunk: str) -> None:
            nonlocal length
            parts.append(chunk)
            length += len(chunk)

        for section in ordered_sections(schema):
            append(f"[{section.label or section.section_id}]\n")
            for field in ordered_fields(section):
                append(f"{field.label or field.field_id}: ")
                value = format_value(fields.get(field.field_id))
                spans.append(FieldSpan(field.field_id, section.section_id, length, length + len(value)))
                append(value + "\n")
            append("\n")
        return cls("".join(parts), tuple(spans))

    def span_for(self, field_id: str) -> FieldSpan | None:
        return next((span for span in self.spans if span.field_id == field_id), None)

    def locate(self, start: int, end: int) -> tuple[FieldSpan, int, int] | None:
        """문서 범위를 하나의 필드 값 내부 범위로 변환한다. 경계를 넘으면 ``None``이다."""
        for span in self.spans:
            if span.start <= start and end <= span.end and start < end:
                return span, start - span.start, end - span.start
        return None
