"""양식 스키마의 승인 가능 상태를 검사하는 규칙이다.

구조 규칙(필드 ID 중복·참조 등)은 ``contracts.form_schema``가 생성 시점에 검사한다.
"""

from __future__ import annotations

from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus


class SchemaValidationError(ValueError):
    """스키마가 승인할 수 없는 상태일 때 발생한다."""


def ensure_approvable(schema: FormSchema) -> None:
    if schema.status not in (SchemaStatus.DRAFT, SchemaStatus.REVIEW, SchemaStatus.APPROVED):
        raise SchemaValidationError("only draft, review, or approved schemas can be approved")
