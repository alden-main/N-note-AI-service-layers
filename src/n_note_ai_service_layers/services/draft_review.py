"""확정 전 간호 초안의 후보 결정과 직접 편집 revision, 최종 확정을 관리한다."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from n_note_ai_service_layers.adapters.document_store import DocumentStore, InMemoryDocumentStore, VersionConflict
from n_note_ai_service_layers.contracts.codec import from_json_dict, to_json_dict

from n_note_ai_service_layers.contracts.draft_review import (
    CandidateDecision,
    DraftCandidate,
    DraftReviewRecord,
    DraftRevision,
    FinalizedNursingNote,
)
from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.contracts.numeric_verification import NumericFinding
from n_note_ai_service_layers.services.audit import AuditTrail
from n_note_ai_service_layers.services.draft_document import is_filled
from n_note_ai_service_layers.services.numeric_verification import NumericVerifier


class DraftAlreadyFinalized(ValueError):
    """확정된 초안을 다시 수정하거나 확정하려 할 때 발생한다."""


class NumericReviewRequired(ValueError):
    """출처에 없는 수치가 남아 있는데 확정자가 수치 확인을 표시하지 않았을 때 발생한다."""

    def __init__(self, findings: tuple[NumericFinding, ...]) -> None:
        super().__init__("numeric values not found in the transcript need confirmation")
        self.findings = findings


class DraftRevisionConflict(ValueError):
    """다른 사용자가 먼저 초안을 수정해, 요청이 기준으로 삼은 revision이 최신이 아니다."""


DRAFT_COLLECTION = "drafts"


class DraftReviewStore:
    """확정 전 초안 revision을 문서 저장소(SQLite 또는 메모리)에 보관한다.

    입력: 초안 생성·수정·확정 요청과 활성 template이다.
    출력: 사용자 편집과 후보 결정을 순서대로 담은 불변 revision과 확정본이다.
    동시 수정은 저장소 버전으로 거부하고(``DraftRevisionConflict``), 보관 기간이 지난 초안은 삭제한다.
    """
    def __init__(self, store: DocumentStore | None = None, audit: AuditTrail | None = None, retention_hours: float | None = None) -> None:
        self.store = store or InMemoryDocumentStore()
        self.audit = audit or AuditTrail()
        self.retention_hours = retention_hours

    def clear(self) -> None:
        self.store.clear(DRAFT_COLLECTION)

    def _load(self, draft_id: str) -> tuple[DraftReviewRecord, int]:
        stored = self.store.get(DRAFT_COLLECTION, draft_id)
        if stored is None:
            raise LookupError("draft not found")
        return from_json_dict(DraftReviewRecord, stored.payload), stored.version

    def _save(self, record: DraftReviewRecord, expected_version: int) -> None:
        try:
            self.store.put(DRAFT_COLLECTION, record.draft_id, to_json_dict(record), expected_version=expected_version)
        except VersionConflict as exc:
            raise DraftRevisionConflict("draft was modified by another request; reload and retry") from exc

    @staticmethod
    def _check_expected(record: DraftReviewRecord, expected_revision: int | None) -> None:
        if expected_revision is not None and record.latest_revision != expected_revision:
            raise DraftRevisionConflict(f"draft is at revision {record.latest_revision}, not {expected_revision}; reload and retry")

    def create(
        self,
        schema: FormSchema,
        transcript: str,
        fields: dict[str, Any],
        candidate_ids: tuple[str, ...],
        trace_id: str,
        candidates: tuple[DraftCandidate, ...] = (),
        patient_id: str | None = None,
        encounter_id: str | None = None,
        numeric_sources: tuple[str, ...] = (),
        reference_time: datetime | None = None,
        llm_applied_term_hints: tuple[dict[str, str], ...] = (),
    ) -> DraftReviewRecord:
        """초안 생성 결과를 첫 revision으로 저장한다.

        입력: 활성 template, 원문, 허용 초안 필드, 후보 ID·상세, 추적 ID와 환자 문맥이다.
        출력: 모든 후보가 pending 상태인 revision 1 레코드이다.
        """
        self._validate_fields(schema, fields)
        draft_id = str(uuid4())
        now = datetime.now(timezone.utc)
        initial = DraftRevision(draft_id, 1, dict(fields), {candidate_id: "pending" for candidate_id in candidate_ids}, "system", now)
        record = DraftReviewRecord(
            draft_id, schema.schema_id, schema.version, transcript, candidate_ids, (initial,), candidates, patient_id, encounter_id,
            numeric_sources=numeric_sources or (transcript,), reference_time=reference_time,
            llm_applied_term_hints=llm_applied_term_hints, created_at=now,
        )
        if self.retention_hours:
            # 전사문·환자 ID가 담긴 초안을 무기한 보관하지 않는다.
            self.store.purge_older_than(DRAFT_COLLECTION, time.time() - self.retention_hours * 3600)
        self._save(record, expected_version=0)
        self.audit.record(trace_id, "draft_created", schema.schema_id, schema.version, subject_id=draft_id)
        return record

    def get(self, draft_id: str) -> DraftReviewRecord:
        """초안 ID에 해당하는 검토 레코드를 반환한다."""
        return self._load(draft_id)[0]

    def revise(
        self,
        draft_id: str,
        schema: FormSchema,
        edited_fields: dict[str, Any],
        decisions: dict[str, CandidateDecision],
        edited_by: str,
        trace_id: str,
        selected_suggestions: dict[str, str] | None = None,
        expected_revision: int | None = None,
    ) -> DraftReviewRecord:
        """후보 결정과 직접 편집을 다음 revision으로 추가한다.

        승인한 초안 후보는 선택한 제안어(없으면 첫 제안어)로 해당 필드 값의 원문 범위를 치환한다.
        직접 편집한 필드와 같은 필드의 후보를 함께 승인하거나, 원문 범위가 바뀐 후보를 승인하면 거부한다.

        입력: 초안 ID, 활성 template, 변경할 필드·후보 결정·선택 제안어, 편집자와 추적 ID,
        선택적인 기준 revision(다르면 ``DraftRevisionConflict``)이다.
        출력: 변경 후 전체 필드와 누적 결정을 가진 새 revision 레코드이다.
        """
        record, version = self._load(draft_id)
        self._check_expected(record, expected_revision)
        if record.finalized_note_id:
            raise DraftAlreadyFinalized("draft is already finalized")
        if (record.schema_id, record.schema_version) != (schema.schema_id, schema.version):
            raise ValueError("draft template does not match active template")
        if not edited_by:
            raise ValueError("edited_by is required")
        selected_suggestions = selected_suggestions or {}
        if set(decisions) - set(record.candidate_ids) or set(selected_suggestions) - set(record.candidate_ids):
            raise ValueError("unknown candidate decision")
        if any(value not in ("pending", "accepted", "rejected") for value in decisions.values()):
            raise ValueError("invalid candidate decision")
        self._validate_fields(schema, edited_fields)
        previous = record.revisions[-1]
        applied_before = {key for revision in record.revisions for key in revision.applied_suggestions}
        if any(candidate_id in applied_before and decision != "accepted" for candidate_id, decision in decisions.items()):
            raise ValueError("an applied candidate cannot be reverted; edit the field directly")

        fields = dict(previous.fields)
        candidates = {item.candidate_id: item for item in record.candidates}
        to_apply = [
            candidates[candidate_id]
            for candidate_id, decision in decisions.items()
            if decision == "accepted" and candidate_id not in applied_before
            and candidate_id in candidates and candidates[candidate_id].field_id
        ]
        if any(candidate.field_id in edited_fields for candidate in to_apply):
            raise ValueError("a field cannot be edited directly and corrected by a candidate in the same revision")
        applied: dict[str, str] = {}
        # 같은 필드 안에서는 뒤쪽 범위부터 치환해 앞쪽 후보의 offset을 보존한다.
        for planned in sorted(to_apply, key=lambda item: (item.field_id, -item.start_offset)):
            candidate = candidates[planned.candidate_id]
            if candidate.stale:
                raise ValueError(f"candidate {candidate.candidate_id} no longer matches the draft field")
            replacement = selected_suggestions.get(candidate.candidate_id, candidate.suggestions[0] if candidate.suggestions else None)
            if replacement is None or replacement not in candidate.suggestions:
                raise ValueError(f"candidate {candidate.candidate_id} requires one of its suggestions")
            value = fields.get(candidate.field_id)
            if not isinstance(value, str) or value[candidate.start_offset:candidate.end_offset] != candidate.input_text:
                raise ValueError(f"candidate {candidate.candidate_id} no longer matches the draft field")
            fields[candidate.field_id] = value[:candidate.start_offset] + replacement + value[candidate.end_offset:]
            applied[candidate.candidate_id] = replacement
            candidates = self._shift(candidates, candidate, len(replacement) - len(candidate.input_text))
        for field_id in edited_fields:
            candidates = {key: replace(item, stale=True) if item.field_id == field_id and key not in applied_before else item for key, item in candidates.items()}
        fields.update(edited_fields)

        revision = DraftRevision(draft_id, previous.revision + 1, fields, {**previous.candidate_decisions, **decisions}, edited_by, datetime.now(timezone.utc), applied)
        updated = replace(record, revisions=record.revisions + (revision,), candidates=tuple(candidates[item.candidate_id] for item in record.candidates))
        self._save(updated, version)
        self.audit.record(trace_id, "draft_revised", schema.schema_id, schema.version, actor=edited_by, subject_id=draft_id)
        return updated

    def numeric_findings(self, record: DraftReviewRecord, schema: FormSchema) -> tuple[NumericFinding, ...]:
        """최신 revision의 필드 값 중 출처(전사문·사실)에 없는 수치다."""
        return NumericVerifier().check(schema, record.revisions[-1].fields, record.numeric_sources, record.reference_time)

    def finalize(
        self, draft_id: str, schema: FormSchema, finalized_by: str, trace_id: str, numeric_review_confirmed: bool = False,
        expected_revision: int | None = None,
    ) -> FinalizedNursingNote:
        """최신 revision을 확정본으로 만든다.

        필수 필드가 비어 있거나, 출처에 없는 수치가 남아 있는데 ``numeric_review_confirmed``가 아니면
        확정하지 않는다. 확인한 수치는 확정본에 남긴다.

        입력: 초안 ID, 초안과 같은 template, 확정자와 추적 ID, 수치 확인 여부이다.
        출력: 이후 수정할 수 없는 ``FinalizedNursingNote``이다.
        """
        record, version = self._load(draft_id)
        if record.finalized_note_id:
            raise DraftAlreadyFinalized("draft is already finalized")
        self._check_expected(record, expected_revision)
        if (record.schema_id, record.schema_version) != (schema.schema_id, schema.version):
            raise ValueError("draft template does not match active template")
        if not finalized_by:
            raise ValueError("finalized_by is required")
        latest = record.revisions[-1]
        missing = [field.field_id for field in schema.fields if field.required and not is_filled(latest.fields.get(field.field_id))]
        if missing:
            raise ValueError("required fields are empty: " + ", ".join(missing))
        numeric = self.numeric_findings(record, schema)
        if numeric and not numeric_review_confirmed:
            raise NumericReviewRequired(numeric)
        applied = {key: value for revision in record.revisions for key, value in revision.applied_suggestions.items()}
        note = FinalizedNursingNote(
            str(uuid4()), draft_id, latest.revision, schema.schema_id, schema.version, schema.display_title,
            record.patient_id, record.encounter_id, dict(latest.fields), dict(latest.candidate_decisions), applied,
            finalized_by, datetime.now(timezone.utc), trace_id,
            tuple(f"{item.field_id}: {item.text}" for item in numeric),
            record.llm_applied_term_hints,
        )
        # 같은 초안을 두 요청이 동시에 확정하면 저장소 버전 검사로 하나만 성공한다.
        self._save(replace(record, finalized_note_id=note.note_id), version)
        self.audit.record(trace_id, "draft_finalized", schema.schema_id, schema.version, actor=finalized_by, subject_id=note.note_id)
        return note

    @staticmethod
    def _shift(candidates: dict[str, DraftCandidate], applied: DraftCandidate, delta: int) -> dict[str, DraftCandidate]:
        """적용된 후보 뒤쪽 후보의 offset을 이동하고, 겹치는 후보는 stale로 표시한다."""
        shifted: dict[str, DraftCandidate] = {}
        for key, item in candidates.items():
            if key == applied.candidate_id or item.field_id != applied.field_id:
                shifted[key] = item
            elif item.start_offset >= applied.end_offset:
                shifted[key] = replace(item, start_offset=item.start_offset + delta, end_offset=item.end_offset + delta)
            elif item.end_offset <= applied.start_offset:
                shifted[key] = item
            else:
                shifted[key] = replace(item, stale=True)
        return shifted

    @staticmethod
    def _validate_fields(schema: FormSchema, fields: dict[str, Any]) -> None:
        if set(fields) - {field.field_id for field in schema.fields}:
            raise ValueError("draft contains fields outside the template")
