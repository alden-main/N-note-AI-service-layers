"""간호 초안의 사용자 검토·수정 이력과 확정본을 표현하는 불변 계약이다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


CandidateDecision = Literal["pending", "accepted", "rejected"]


@dataclass(frozen=True)
class DraftCandidate:
    """초안 검토 화면에 표시하는 Gateway 용어 후보다.

    ``field_id``가 있는 초안 후보만 필드 값에 적용할 수 있으며 offset은 그 필드 값 기준이다.
    필드가 직접 편집되거나 겹치는 후보가 적용되면 ``stale``이 되어 더 이상 적용할 수 없다.
    """
    candidate_id: str
    phase: str
    tool_name: str
    start_offset: int
    end_offset: int
    input_text: str
    suggestions: tuple[str, ...]
    field_id: str | None = None
    category: str | None = None
    stale: bool = False


@dataclass(frozen=True)
class DraftRevision:
    """사용자 선택과 직접 편집 후의 한 초안 버전을 표현한다.

    입력: 초안 ID·순번, 전체 필드 값, 후보 결정, 편집자와 생성 시각, 이번 revision에서 적용한 제안어이다.
    출력: 수정 가능한 초안의 변경 이력에 추가할 불변 revision 객체이다.
    """
    draft_id: str
    revision: int
    fields: dict[str, Any]
    candidate_decisions: dict[str, CandidateDecision]
    edited_by: str
    created_at: datetime
    applied_suggestions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftReviewRecord:
    """확정 전 간호 초안과 그 revision 목록을 표현한다.

    입력: 초안 식별자, template 식별자·버전, 원문, 후보 ID와 revision 목록, 후보 상세, 환자 문맥이다.
    출력: 사용자 검토 API가 조회·수정하는 메모리 기반 검토 레코드이다.
    """
    draft_id: str
    schema_id: str
    schema_version: str
    transcript: str
    candidate_ids: tuple[str, ...]
    revisions: tuple[DraftRevision, ...]
    candidates: tuple[DraftCandidate, ...] = ()
    patient_id: str | None = None
    encounter_id: str | None = None
    finalized_note_id: str | None = None

    def candidate(self, candidate_id: str) -> DraftCandidate | None:
        return next((item for item in self.candidates if item.candidate_id == candidate_id), None)


@dataclass(frozen=True)
class FinalizedNursingNote:
    """사용자가 확정한 간호기록이다. PDF는 이 확정본만 렌더링한다."""
    note_id: str
    draft_id: str
    revision: int
    schema_id: str
    schema_version: str
    form_title: str
    patient_id: str | None
    encounter_id: str | None
    fields: dict[str, Any]
    candidate_decisions: dict[str, CandidateDecision]
    applied_suggestions: dict[str, str]
    finalized_by: str
    finalized_at: datetime
    trace_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "draft_id": self.draft_id,
            "revision": self.revision,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "form_title": self.form_title,
            "patient_id": self.patient_id,
            "encounter_id": self.encounter_id,
            "fields": self.fields,
            "candidate_decisions": self.candidate_decisions,
            "applied_suggestions": self.applied_suggestions,
            "finalized_by": self.finalized_by,
            "finalized_at": self.finalized_at.isoformat(),
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FinalizedNursingNote":
        return cls(
            raw["note_id"], raw["draft_id"], int(raw["revision"]), raw["schema_id"], raw["schema_version"],
            raw["form_title"], raw.get("patient_id"), raw.get("encounter_id"), dict(raw["fields"]),
            dict(raw["candidate_decisions"]), dict(raw.get("applied_suggestions", {})), raw["finalized_by"],
            datetime.fromisoformat(raw["finalized_at"]), raw["trace_id"],
        )
