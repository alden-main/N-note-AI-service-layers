"""양식 디렉터리의 ``form.json``과 PDF 양식 파일을 활성 양식 스키마로 읽는다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from n_note_ai_service_layers.contracts.form_schema import (
    FormField,
    FormSchema,
    FormSection,
    PdfPlacement,
    SchemaStatus,
)
from n_note_ai_service_layers.services.schema_validation import SchemaValidationError, validate_schema_structure

FORM_DEFINITION_FILE = "form.json"
SUPPORTED_FIELD_TYPES = {"string", "text", "integer", "number", "boolean", "array", "date", "datetime", "enum"}
META_KEYS = {"form_title", "patient_id", "encounter_id", "finalized_by", "finalized_at", "note_id"}


class FormDefinitionError(ValueError):
    """양식 정의 파일이 계약을 만족하지 않을 때 발생한다."""


class FormDefinitionLoader:
    """운영자가 배포한 양식 정의를 승인·활성 상태의 ``FormSchema``로 변환한다.

    입력: ``<forms_dir>/<form>/form.json``과 선택 PDF 양식 파일이다.
    출력: 양식 디렉터리 경로를 승인 출처로 기록한 활성 ``FormSchema`` 목록이다.
    """

    def __init__(self, forms_dir: Path) -> None:
        self.forms_dir = forms_dir

    def load_all(self, enabled_ids: tuple[str, ...] = ()) -> tuple[FormSchema, ...]:
        """양식 디렉터리의 모든 정의를 읽고, 활성화 목록이 있으면 그 양식만 반환한다.

        입력: ``enabled_ids``는 사용할 ``schema_id`` 목록이며 비어 있으면 전체를 사용한다.
        출력: 정의 파일 이름 순서의 활성 스키마 튜플이다. 목록에 있는 ID가 없으면 오류다.
        """
        if not self.forms_dir.is_dir():
            if enabled_ids:
                raise FormDefinitionError(f"forms directory not found: {self.forms_dir}")
            return ()
        schemas = [self.load(path) for path in sorted(self.forms_dir.glob(f"*/{FORM_DEFINITION_FILE}"))]
        ids = [schema.schema_id for schema in schemas]
        if len(ids) != len(set(ids)):
            raise FormDefinitionError("schema_id must be unique across form definitions")
        if enabled_ids:
            missing = set(enabled_ids) - set(ids)
            if missing:
                raise FormDefinitionError("enabled forms are not defined: " + ", ".join(sorted(missing)))
            schemas = [schema for schema in schemas if schema.schema_id in enabled_ids]
        return tuple(schemas)

    def load(self, path: Path) -> FormSchema:
        """양식 정의 파일 하나를 검증해 활성 스키마로 변환한다."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FormDefinitionError(f"cannot read form definition: {path}") from exc
        if not isinstance(raw, dict):
            raise FormDefinitionError(f"form definition must be an object: {path}")
        pdf_template = None
        pdf_config = raw.get("pdf") or {}
        if not isinstance(pdf_config, dict):
            raise FormDefinitionError(f"pdf must be an object: {path}")
        template_name = pdf_config.get("template")
        meta = pdf_config.get("meta") or {}
        if not isinstance(meta, dict) or set(meta) - META_KEYS:
            raise FormDefinitionError(f"pdf.meta keys must be among {sorted(META_KEYS)}: {path}")
        pdf_meta = tuple((key, _placement(value, path)) for key, value in meta.items())
        if template_name:
            template_path = (path.parent / template_name).resolve()
            if path.parent.resolve() not in template_path.parents or not template_path.is_file():
                raise FormDefinitionError(f"PDF template must be an existing file in the form directory: {path}")
            pdf_template = str(template_path)
        sections = tuple(self._section(item, path) for item in _list(raw.get("sections"), "sections", path))
        schema = FormSchema(
            _text(raw.get("schema_id"), "schema_id", path),
            _text(raw.get("version"), "version", path),
            _text(raw.get("purpose"), "purpose", path),
            sections,
            status=SchemaStatus.ACTIVE,
            approved_by=f"form-definition:{path.parent.name}",
            approved_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
            title=_optional_text(raw.get("title"), "title", path),
            instructions=_optional_text(raw.get("instructions"), "instructions", path),
            pdf_template=pdf_template,
            pdf_meta=pdf_meta,
            validation_criteria=tuple(_list(raw.get("validation_criteria", []), "validation_criteria", path)),
        )
        try:
            validate_schema_structure(schema)
        except SchemaValidationError as exc:
            raise FormDefinitionError(f"{path}: {exc}") from exc
        return schema

    def _section(self, raw: Any, path: Path) -> FormSection:
        if not isinstance(raw, dict):
            raise FormDefinitionError(f"section must be an object: {path}")
        fields = tuple(self._field(item, path) for item in _list(raw.get("fields"), "fields", path))
        return FormSection(
            _text(raw.get("section_id"), "section_id", path),
            fields,
            _optional_text(raw.get("label"), "label", path),
            _optional_int(raw.get("display_order"), path),
        )

    @staticmethod
    def _field(raw: Any, path: Path) -> FormField:
        if not isinstance(raw, dict):
            raise FormDefinitionError(f"field must be an object: {path}")
        field_type = raw.get("type", "text")
        if field_type not in SUPPORTED_FIELD_TYPES:
            raise FormDefinitionError(f"unsupported field type {field_type!r}: {path}")
        options = tuple(str(item) for item in _list(raw.get("options", []), "options", path))
        if field_type == "enum" and not options:
            raise FormDefinitionError(f"enum field requires options: {path}")
        placement = _placement(raw.get("pdf"), path) if raw.get("pdf") is not None else None
        return FormField(
            _text(raw.get("field_id"), "field_id", path),
            field_type,
            bool(raw.get("required", False)),
            bool(raw.get("llm_allowed", True)),
            _optional_text(raw.get("fact_id"), "fact_id", path),
            _optional_text(raw.get("conditional_required_when"), "conditional_required_when", path),
            _optional_text(raw.get("label"), "label", path),
            _optional_int(raw.get("display_order"), path),
            _optional_text(raw.get("description"), "description", path),
            options,
            placement,
        )


def _placement(pdf: Any, path: Path) -> PdfPlacement:
    """AcroForm 필드명 또는 쪽·좌표·크기로 지정한 PDF 배치 정보를 읽는다."""
    if not isinstance(pdf, dict):
        raise FormDefinitionError(f"pdf placement must be an object: {path}")
    try:
        placement = PdfPlacement(
            field_name=pdf.get("field_name"),
            page=int(pdf["page"]) if "page" in pdf else None,
            x=float(pdf["x"]) if "x" in pdf else None,
            y=float(pdf["y"]) if "y" in pdf else None,
            width=float(pdf["width"]) if "width" in pdf else None,
            height=float(pdf["height"]) if "height" in pdf else None,
            font_size=float(pdf.get("font_size", 10.0)),
        )
    except (TypeError, ValueError) as exc:
        raise FormDefinitionError(f"invalid pdf placement: {path}") from exc
    if not placement.field_name and not placement.has_box:
        raise FormDefinitionError(f"pdf placement requires field_name or page/x/y/width/height: {path}")
    return placement


def _list(value: Any, name: str, path: Path) -> list[Any]:
    if not isinstance(value, list):
        raise FormDefinitionError(f"{name} must be a list: {path}")
    return value


def _text(value: Any, name: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormDefinitionError(f"{name} must be a non-empty string: {path}")
    return value.strip()


def _optional_text(value: Any, name: str, path: Path) -> str | None:
    if value is None:
        return None
    return _text(value, name, path)


def _optional_int(value: Any, path: Path) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise FormDefinitionError(f"display_order must be an integer: {path}")
    return value
