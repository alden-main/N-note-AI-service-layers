"""양식 스키마의 구조와 승인 가능 상태를 검증하는 규칙이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus


class SchemaValidationError(ValueError):
    """스키마 구조 또는 승인 조건이 유효하지 않을 때 발생한다."""
    pass


def validate_schema_structure(schema: FormSchema) -> None:
    """스키마 식별자, 필드, 조건식 및 브리지 매핑의 구조를 검사한다.

    입력: ``schema``는 검사할 양식 스키마이다.
    출력: 반환값은 없으며, 구조가 유효하지 않으면 ``SchemaValidationError``를 발생시킨다.
    """
    if not schema.schema_id or not schema.version or not schema.purpose:
        raise SchemaValidationError("schema_id, version, and purpose are required")
    field_ids = [field.field_id for field in schema.fields]
    if len(field_ids) != len(set(field_ids)):
        raise SchemaValidationError("field IDs must be unique")
    if any(not field.field_id or not field.type for field in schema.fields):
        raise SchemaValidationError("every field requires an ID and type")
    if any(field.display_order is not None and field.display_order < 0 for field in schema.fields):
        raise SchemaValidationError("field display order must not be negative")
    if any(section.display_order is not None and section.display_order < 0 for section in schema.sections):
        raise SchemaValidationError("section display order must not be negative")
    known = set(field_ids)
    for field in schema.fields:
        if field.conditional_required_when and field.conditional_required_when not in known:
            raise SchemaValidationError("conditional rule references an unknown field")
    for mapping in schema.bridge_mapping:
        if mapping.field_id not in known or not mapping.output_key:
            raise SchemaValidationError("bridge mapping must reference a schema field")


def ensure_approvable(schema: FormSchema) -> None:
    if schema.status not in (SchemaStatus.DRAFT, SchemaStatus.REVIEW, SchemaStatus.APPROVED):
        raise SchemaValidationError("only draft, review, or approved schemas can be approved")
    validate_schema_structure(schema)
