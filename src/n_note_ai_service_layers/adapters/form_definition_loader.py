"""양식 디렉터리의 ``form.json``과 PDF 양식 파일을 활성 양식 스키마로 읽는다."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, StrictInt, StringConstraints, ValidationError, model_validator

from n_note_ai_service_layers.contracts.codec import from_json_dict
from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus
from n_note_ai_service_layers.services.schema_validation import SchemaValidationError, validate_schema_structure

FORM_DEFINITION_FILE = "form.json"
FieldType = Literal["string", "text", "integer", "number", "boolean", "array", "date", "datetime", "enum"]
MetaKey = Literal["form_title", "patient_id", "encounter_id", "finalized_by", "finalized_at", "note_id"]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Placement(BaseModel):
    """AcroForm 필드명 또는 쪽·좌표·크기로 지정한 PDF 배치 정보다."""

    field_name: str | None = None
    page: int | None = None
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    font_size: float = 10.0

    @model_validator(mode="after")
    def _located(self) -> _Placement:
        if not self.field_name and None in (self.page, self.x, self.y, self.width, self.height):
            raise ValueError("pdf placement requires field_name or page/x/y/width/height")
        return self


class _Field(BaseModel):
    field_id: Text
    type: FieldType = "text"
    required: bool = False
    llm_allowed: bool = True
    fact_id: Text | None = None
    conditional_required_when: Text | None = None
    label: Text | None = None
    display_order: StrictInt | None = None
    description: Text | None = None
    options: list[Annotated[str, BeforeValidator(str)]] = []
    pdf: _Placement | None = None

    @model_validator(mode="after")
    def _enum_has_options(self) -> _Field:
        if self.type == "enum" and not self.options:
            raise ValueError("enum field requires options")
        return self


class _Section(BaseModel):
    section_id: Text
    fields: list[_Field]
    label: Text | None = None
    display_order: StrictInt | None = None


class _PdfConfig(BaseModel):
    template: str | None = None
    meta: dict[MetaKey, _Placement] | None = None


class _FormDefinition(BaseModel):
    """``form.json``의 계약이다."""

    schema_id: Text
    version: Text
    purpose: Text
    sections: list[_Section]
    title: Text | None = None
    instructions: Text | None = None
    pdf: _PdfConfig | None = None
    validation_criteria: list[dict[str, Any]] = []


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
            definition = _FormDefinition.model_validate_json(path.read_bytes())
        except OSError as exc:
            raise FormDefinitionError(f"cannot read form definition: {path}") from exc
        except ValidationError as exc:
            raise FormDefinitionError(f"invalid form definition {path}: {exc}") from exc
        pdf = definition.pdf or _PdfConfig()
        pdf_template = None
        if pdf.template:
            template_path = (path.parent / pdf.template).resolve()
            if path.parent.resolve() not in template_path.parents or not template_path.is_file():
                raise FormDefinitionError(f"PDF template must be an existing file in the form directory: {path}")
            pdf_template = str(template_path)
        schema = from_json_dict(FormSchema, {
            **definition.model_dump(exclude={"pdf"}),
            "status": SchemaStatus.ACTIVE,
            "approved_by": f"form-definition:{path.parent.name}",
            "approved_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
            "pdf_template": pdf_template,
            "pdf_meta": [(key, placement.model_dump()) for key, placement in (pdf.meta or {}).items()],
        })
        try:
            validate_schema_structure(schema)
        except SchemaValidationError as exc:
            raise FormDefinitionError(f"{path}: {exc}") from exc
        return schema
