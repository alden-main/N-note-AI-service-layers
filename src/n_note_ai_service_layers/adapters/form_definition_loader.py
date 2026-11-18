"""양식 디렉터리의 ``form.json``과 PDF 양식 파일을 활성 양식 스키마로 읽는다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from n_note_ai_service_layers.contracts.codec import from_json_dict
from n_note_ai_service_layers.contracts.form_schema import FormSchema, PdfPlacement, SchemaStatus

# 파일에 있어도 무시하고 배포 사실로 정하는 키다(``pdf``는 위에서 따로 읽는다).
_DEPLOYMENT_KEYS = frozenset({"pdf", "status", "approved_by", "approved_at", "pdf_template", "pdf_meta"})

FORM_DEFINITION_FILE = "form.json"
MetaKey = Literal["form_title", "patient_id", "encounter_id", "finalized_by", "finalized_at", "note_id"]


class _PdfConfig(BaseModel):
    """``form.json``의 ``pdf`` 블록이다. 나머지 키는 ``FormSchema`` 계약을 그대로 따른다."""

    template: str | None = None
    meta: dict[MetaKey, PdfPlacement] | None = None


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
            raw = json.loads(path.read_bytes())
        except OSError as exc:
            raise FormDefinitionError(f"cannot read form definition: {path}") from exc
        except json.JSONDecodeError as exc:
            raise FormDefinitionError(f"invalid form definition {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise FormDefinitionError(f"invalid form definition {path}: top level must be an object")
        try:
            pdf = _PdfConfig.model_validate(raw.get("pdf") or {})
            pdf_template = self._pdf_template(path, pdf.template)
            # 승인·활성 상태는 파일이 아니라 배포 사실(양식 디렉터리)로 정한다.
            return from_json_dict(FormSchema, {
                **{key: value for key, value in raw.items() if key not in _DEPLOYMENT_KEYS},
                "status": SchemaStatus.ACTIVE,
                "approved_by": f"form-definition:{path.parent.name}",
                "approved_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
                "pdf_template": pdf_template,
                "pdf_meta": [(key, placement) for key, placement in (pdf.meta or {}).items()],
            })
        except ValidationError as exc:
            raise FormDefinitionError(f"invalid form definition {path}: {exc}") from exc

    @staticmethod
    def _pdf_template(path: Path, template: str | None) -> str | None:
        """PDF 양식 파일 경로다. 양식 디렉터리 밖이거나 없는 파일은 거부한다."""
        if not template:
            return None
        template_path = (path.parent / template).resolve()
        if path.parent.resolve() not in template_path.parents or not template_path.is_file():
            raise FormDefinitionError(f"PDF template must be an existing file in the form directory: {path}")
        return str(template_path)
