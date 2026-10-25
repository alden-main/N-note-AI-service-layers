"""양식 스키마의 버전 등록, 검토, 승인과 활성화를 관리하는 서비스이다."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from n_note_ai_service_layers.contracts.form_schema import (
    ChangeLogEntry, FormField, FormSchema, SchemaCandidate, SchemaReview, SchemaStatus,
)
from n_note_ai_service_layers.services.schema_validation import ensure_approvable, validate_schema_structure


class FormSchemaRegistry:
    """스키마 버전을 보관하고 업무 목적별 활성 버전을 관리한다.

    입력: ``FormSchema`` 등록 및 식별자·버전 기반 조회 요청이다.
    출력: 등록·조회·활성화된 불변 ``FormSchema``이며, 버전 및 활성화 규칙을 강제한다.
    """
    def __init__(self) -> None:
        self._schemas: dict[tuple[str, str], FormSchema] = {}

    def add(self, schema: FormSchema) -> FormSchema:
        """새 스키마 버전을 등록하고 활성 버전 중복을 방지한다.

        입력: ``schema``는 등록할 양식 스키마이다.
        출력: 동일한 ``schema``이며, 기존 버전 또는 활성 버전 충돌 시 ``ValueError``를 발생시킨다.
        """
        validate_schema_structure(schema)
        key = (schema.schema_id, schema.version)
        if key in self._schemas:
            raise ValueError("schema version already exists and is immutable")
        if schema.status is SchemaStatus.ACTIVE and any(
            item.schema_id == schema.schema_id and item.status is SchemaStatus.ACTIVE
            for item in self._schemas.values()
        ):
            raise ValueError("only one active version is allowed")
        self._schemas[key] = schema
        return schema

    def get(self, schema_id: str, version: str) -> FormSchema:
        return self._schemas[(schema_id, version)]

    def resolve_active(self, schema_id: str, purpose: str, version: str | None = None) -> FormSchema:
        """업무 목적과 선택 버전에 일치하는 유일한 활성 스키마를 찾는다.

        입력: 스키마 ID, 업무 목적, 선택적인 요청 버전이다.
        출력: 일치하는 활성 ``FormSchema``이며 없거나 여러 개이면 ``LookupError``를 발생시킨다.
        """
        matches = [s for s in self._schemas.values() if s.schema_id == schema_id and s.status is SchemaStatus.ACTIVE]
        if len(matches) != 1 or matches[0].purpose != purpose:
            raise LookupError("no active schema for this purpose")
        schema = matches[0]
        if version and schema.version != version:
            raise LookupError("requested schema version is not active")
        return schema

    def list_versions(self, schema_id: str | None = None) -> tuple[FormSchema, ...]:
        """등록된 template 버전을 표시 순서와 무관한 안정된 순서로 반환한다.

        입력: 선택적인 template 식별자이다.
        출력: 해당 template 또는 전체의 불변 스키마 버전 튜플이다.
        """
        schemas = (schema for schema in self._schemas.values() if schema_id is None or schema.schema_id == schema_id)
        return tuple(sorted(schemas, key=lambda schema: (schema.schema_id, schema.version)))

    def _replace(self, schema: FormSchema) -> None:
        self._schemas[(schema.schema_id, schema.version)] = schema

    def activate(self, schema_id: str, version: str) -> FormSchema:
        """승인된 스키마 버전을 활성화하고 기존 활성 버전을 승인 상태로 전환한다.

        입력: 활성화할 ``schema_id``와 ``version``이다.
        출력: 활성 상태의 ``FormSchema``이며, 승인되지 않은 버전은 ``ValueError``를 발생시킨다.
        """
        target = self.get(schema_id, version)
        if target.status is not SchemaStatus.APPROVED:
            raise ValueError("only approved schemas can become active")
        for item in list(self._schemas.values()):
            if item.schema_id == schema_id and item.status is SchemaStatus.ACTIVE:
                self._replace(replace(item, status=SchemaStatus.APPROVED))
        target = replace(target, status=SchemaStatus.ACTIVE)
        self._replace(target)
        return target


class SchemaReviewService:
    """스키마 후보를 초안으로 만들고 검토·수정·승인 이력을 관리한다.

    입력: 스키마를 보관할 ``FormSchemaRegistry``이다.
    출력: 검토 상태가 반영된 스키마와 ``SchemaReview`` 이력이다.
    """
    def __init__(self, registry: FormSchemaRegistry) -> None:
        self.registry = registry
        self.reviews: dict[tuple[str, str], SchemaReview] = {}

    def create_draft(self, candidate: SchemaCandidate, schema_id: str, version: str, purpose: str) -> FormSchema:
        schema = FormSchema(schema_id, version, purpose, candidate.sections, status=SchemaStatus.DRAFT)
        return self.registry.add(schema)

    def mark_section_reviewed(self, schema: FormSchema, section_id: str, reviewer: str, when: datetime) -> None:
        if section_id not in {section.section_id for section in schema.sections}:
            raise ValueError("unknown section")
        current = self.reviews.get((schema.schema_id, schema.version), SchemaReview(schema.schema_id, schema.version))
        sections = tuple(sorted(set(current.reviewed_sections) | {section_id}))
        self.reviews[(schema.schema_id, schema.version)] = replace(current, reviewed_sections=sections, reviewed_by=reviewer, reviewed_at=when)

    def revise_field(self, schema: FormSchema, field_id: str, replacement: FormField, reason: str, reviewer: str, when: datetime) -> FormSchema:
        """활성화되지 않은 스키마의 한 필드를 바꾸고 변경 이력을 남긴다.

        입력: 대상 ``schema``, 필드 ID, 대체 필드, 변경 사유·검토자·시각이다.
        출력: 검토 상태로 갱신된 ``FormSchema``이며, 활성 스키마 또는 없는 필드는 ``ValueError``를 발생시킨다.
        """
        if schema.status is SchemaStatus.ACTIVE:
            raise ValueError("active schemas are immutable; create a new draft version")
        before = next((field for field in schema.fields if field.field_id == field_id), None)
        if before is None:
            raise ValueError("unknown field")
        sections = tuple(replace(section, fields=tuple(replacement if f.field_id == field_id else f for f in section.fields)) for section in schema.sections)
        revised = replace(schema, sections=sections, status=SchemaStatus.REVIEW)
        self.registry._replace(revised)
        current = self.reviews.get((schema.schema_id, schema.version), SchemaReview(schema.schema_id, schema.version))
        entry = ChangeLogEntry(field_id, before, replacement, reason, reviewer, when)
        self.reviews[(schema.schema_id, schema.version)] = replace(current, change_log=current.change_log + (entry,))
        return revised

    def approve_all(self, schema: FormSchema, reviewer: str, when: datetime) -> FormSchema:
        if schema.status is SchemaStatus.ACTIVE:
            raise ValueError("active schemas cannot be modified")
        ensure_approvable(schema)
        approved = replace(schema, status=SchemaStatus.APPROVED, approved_by=reviewer, approved_at=when)
        self.registry._replace(approved)
        current = self.reviews.get((schema.schema_id, schema.version), SchemaReview(schema.schema_id, schema.version))
        self.reviews[(schema.schema_id, schema.version)] = replace(current, reviewed_by=reviewer, reviewed_at=when)
        return approved
