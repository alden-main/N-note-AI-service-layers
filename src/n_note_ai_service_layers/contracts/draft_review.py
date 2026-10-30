"""간호 초안의 사용자 검토·수정 이력과 확정본을 표현하는 불변 계약이다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from n_note_ai_service_layers.contracts.codec import IsoDatetime


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
    created_at: IsoDatetime
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
    # 수치 대조 출처(전사문·사실)와 구술 기준 시각이다. revision마다 최신 필드 값으로 다시 대조한다.
    numeric_sources: tuple[str, ...] = ()
    reference_time: IsoDatetime | None = None
    # LLM이 전사문 용어 힌트를 받아 초안에 반영한 교정({"source_text", "applied_text"})이다.
    # 후보 승인 절차를 거치지 않은 교정이므로 검토 화면과 확정본에 남긴다.
    llm_applied_term_hints: tuple[dict[str, str], ...] = ()
    created_at: IsoDatetime | None = None

    def candidate(self, candidate_id: str) -> DraftCandidate | None:
        return next((item for item in self.candidates if item.candidate_id == candidate_id), None)

    @property
    def latest_revision(self) -> int:
        return self.revisions[-1].revision


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
    finalized_at: IsoDatetime
    trace_id: str
    # 출처에 없지만 확정자가 확인하고 남긴 수치("필드 ID: 값")다.
    confirmed_numeric_findings: tuple[str, ...] = ()
    llm_applied_term_hints: tuple[dict[str, str], ...] = ()
