"""양식 스키마의 버전 등록, 검토, 승인과 활성화를 관리하는 서비스이다."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime

from n_note_ai_service_layers.adapters.document_store import DocumentStore, VersionConflict
from n_note_ai_service_layers.contracts.codec import from_json_dict, to_json_dict
from n_note_ai_service_layers.contracts.form_schema import (
    ChangeLogEntry, FormField, FormSchema, SchemaCandidate, SchemaReview, SchemaStatus,
)
from n_note_ai_service_layers.observability import get_logger, log_event

TEMPLATE_COLLECTION = "form_templates"
_logger = get_logger("templates")
from n_note_ai_service_layers.services.schema_validation import ensure_approvable, validate_schema_structure


def _store_key(schema_id: str, version: str) -> str:
    return f"{schema_id}\x1f{version}"


class FormSchemaRegistry:
    """스키마 버전을 보관하고 업무 목적별 활성 버전을 관리한다.

    입력: ``FormSchema`` 등록 및 식별자·버전 기반 조회 요청이다.
    출력: 등록·조회·활성화된 불변 ``FormSchema``이며, 버전 및 활성화 규칙을 강제한다.

    저장소가 있으면 두 계층으로 동작한다. 배포된 양식 정의(``add_base``)는 기동할 때마다 같은 값으로
    읽는 기본 계층이고, API 등록·승인·활성화로 바뀐 버전은 저장소에 기록되어 기본 계층보다 우선한다.
    다른 worker가 바꾼 상태는 조회 시 저장소 변경을 감지해 다시 읽는다.
    """
    def __init__(self, store: DocumentStore | None = None) -> None:
        self._schemas: dict[tuple[str, str], FormSchema] = {}
        self._base: dict[tuple[str, str], FormSchema] = {}
        self._store = store
        self._fingerprint: tuple[int, float] | None = None
        self._lock = threading.RLock()

    def _sync(self) -> None:
        if self._store is None:
            return
        fingerprint = self._store.fingerprint(TEMPLATE_COLLECTION)
        if fingerprint == self._fingerprint:
            return
        with self._lock:
            merged = dict(self._base)
            for document in self._store.list(TEMPLATE_COLLECTION):
                schema = from_json_dict(FormSchema, document.payload)
                merged[(schema.schema_id, schema.version)] = schema
            self._schemas = merged
            self._fingerprint = fingerprint

    def _has_other_active(self, schema: FormSchema) -> bool:
        return any(
            item.schema_id == schema.schema_id and item.version != schema.version and item.status is SchemaStatus.ACTIVE
            for item in self._schemas.values()
        )

    def add_base(self, schema: FormSchema) -> FormSchema:
        """배포된 양식 정의를 기본 계층으로 등록한다. 저장소에 같은 버전이 있으면 저장소 상태가 우선한다.

        저장소 없이 쓰면 ``add``와 같다. 저장소에서 같은 양식의 다른 버전이 이미 활성이면, 새로 배포된
        버전은 승인 상태로 등록해 운영자가 API로 활성화하게 한다.
        """
        if self._store is None:
            return self.add(schema)
        validate_schema_structure(schema)
        self._sync()
        key = (schema.schema_id, schema.version)
        with self._lock:
            if self._store.get(TEMPLATE_COLLECTION, _store_key(*key)) is not None:
                return self._schemas[key]
            if key in self._base:
                raise ValueError("schema version already exists and is immutable")
            if schema.status is SchemaStatus.ACTIVE and self._has_other_active(schema):
                log_event(_logger, "deployed_form_not_activated", schema_id=schema.schema_id, version=schema.version)
                schema = replace(schema, status=SchemaStatus.APPROVED)
            self._base[key] = schema
            self._schemas[key] = schema
        return schema

    def add(self, schema: FormSchema) -> FormSchema:
        """새 스키마 버전을 등록하고 활성 버전 중복을 방지한다.

        입력: ``schema``는 등록할 양식 스키마이다.
        출력: 동일한 ``schema``이며, 기존 버전 또는 활성 버전 충돌 시 ``ValueError``를 발생시킨다.
        """
        validate_schema_structure(schema)
        self._sync()
        key = (schema.schema_id, schema.version)
        with self._lock:
            if key in self._schemas:
                raise ValueError("schema version already exists and is immutable")
            if schema.status is SchemaStatus.ACTIVE and self._has_other_active(schema):
                raise ValueError("only one active version is allowed")
            if self._store is not None:
                try:
                    self._store.put(TEMPLATE_COLLECTION, _store_key(*key), to_json_dict(schema), expected_version=0)
                except VersionConflict as exc:
                    raise ValueError("schema version already exists and is immutable") from exc
                self._fingerprint = None
            self._schemas[key] = schema
        return schema

    def get(self, schema_id: str, version: str) -> FormSchema:
        self._sync()
        return self._schemas[(schema_id, version)]

    def resolve_active(self, schema_id: str, purpose: str, version: str | None = None) -> FormSchema:
        """업무 목적과 선택 버전에 일치하는 유일한 활성 스키마를 찾는다.

        입력: 스키마 ID, 업무 목적, 선택적인 요청 버전이다.
        출력: 일치하는 활성 ``FormSchema``이며 없거나 여러 개이면 ``LookupError``를 발생시킨다.
        """
        self._sync()
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
        self._sync()
        schemas = (schema for schema in self._schemas.values() if schema_id is None or schema.schema_id == schema_id)
        return tuple(sorted(schemas, key=lambda schema: (schema.schema_id, schema.version)))

    def _replace(self, schema: FormSchema) -> None:
        with self._lock:
            if self._store is not None:
                self._store.put(TEMPLATE_COLLECTION, _store_key(schema.schema_id, schema.version), to_json_dict(schema))
                self._fingerprint = None
            self._schemas[(schema.schema_id, schema.version)] = schema

    def activate(self, schema_id: str, version: str) -> FormSchema:
        """승인된 스키마 버전을 활성화하고 기존 활성 버전을 승인 상태로 전환한다.

        입력: 활성화할 ``schema_id``와 ``version``이다.
        출력: 활성 상태의 ``FormSchema``이며, 승인되지 않은 버전은 ``ValueError``를 발생시킨다.
        """
        target = self.get(schema_id, version)
        if target.status is not SchemaStatus.APPROVED:
            raise ValueError("only approved schemas can become active")
        for item in list(self.list_versions(schema_id)):
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
