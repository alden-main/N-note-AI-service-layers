"""HTTP API의 요청·응답 모델이다."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class BridgeHandoffRequest(BaseModel):
    """브리지 전송을 요청하는 검토 완료 초안의 API 입력 모델이다.

    입력: 스키마 식별자·버전, 전송 필드, 검증 결과 및 간호사 검토 상태이다.
    출력: 전송 대기 항목을 만들 때 ``HandoffEnvelope``의 기반 데이터가 된다.
    """
    schema_id: str
    schema_version: str
    fields: dict[str, Any]
    validation: dict[str, Any]
    nurse_review_status: str


class BridgeAcknowledgement(BaseModel):
    """브리지 수신 시스템이 전달 상태를 회신하는 API 입력 모델이다.

    입력: 처리 상태와 선택적인 브리지 참조·메시지이다.
    출력: 해당 핸드오프의 상태를 갱신하는 확인 객체이다.
    """
    status: str = Field(description="received, accepted, rejected, or failed")
    bridge_reference: str | None = None
    message: str | None = None


class HandoffEnvelope(BridgeHandoffRequest):
    """저장된 핸드오프의 식별자와 전송 상태를 포함하는 API 응답 모델이다.

    입력: 요청 모델의 필드와 서버가 부여한 식별자·생성 시각·확인 정보이다.
    출력: 생성, 조회, 확인 API가 반환하는 핸드오프 상태이다.
    """
    handoff_id: str
    delivery_status: str
    created_at: datetime
    acknowledgement: BridgeAcknowledgement | None = None


class TranscriptionResponse(BaseModel):
    """음성 파일을 전사한 결과다. 원본 음성 바이트는 응답에 포함하지 않는다."""

    text: str
    provider: str
    model: str
    language: str | None = None
    trace_id: str
    redacted: bool
    redaction_count: int


class TemplateFieldRequest(BaseModel):
    """운영자가 등록할 template 필드의 API 입력 모델이다."""

    field_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    label: str | None = None
    required: bool = False
    llm_allowed: bool = True
    fact_id: str | None = None
    conditional_required_when: str | None = None
    display_order: int | None = Field(default=None, ge=0)


class TemplateSectionRequest(BaseModel):
    """운영자가 등록할 template 섹션의 API 입력 모델이다."""

    section_id: str = Field(min_length=1)
    label: str | None = None
    display_order: int | None = Field(default=None, ge=0)
    fields: list[TemplateFieldRequest] = Field(min_length=1)


class TemplateRegistrationRequest(BaseModel):
    """새 양식 template 버전을 초안 상태로 등록하는 API 입력 모델이다."""

    schema_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    sections: list[TemplateSectionRequest] = Field(min_length=1)


class TemplateActionRequest(BaseModel):
    """template 검토·승인 API에 필요한 작업자와 시각 입력 모델이다.

    인증 사용자 헤더가 있으면 ``operator_id``는 생략할 수 있고, 값이 다르면 거부한다.
    """

    operator_id: str | None = Field(default=None, min_length=1)
    occurred_at: datetime | None = None


class DraftRevisionRequest(BaseModel):
    """후보 결정과 직접 편집을 다음 초안 revision으로 남기는 API 입력 모델이다.

    ``selected_suggestions``는 승인한 초안 후보에 적용할 제안어이며, 생략하면 첫 제안어를 적용한다.
    ``expected_revision``을 주면 그 사이 다른 사용자가 수정한 경우 409로 거부한다.
    """

    edited_by: str | None = Field(default=None, min_length=1)
    edited_fields: dict[str, Any] = Field(default_factory=dict)
    candidate_decisions: dict[str, Literal["pending", "accepted", "rejected"]] = Field(default_factory=dict)
    selected_suggestions: dict[str, str] = Field(default_factory=dict)
    expected_revision: int | None = Field(default=None, ge=1)


class DraftFinalizeRequest(BaseModel):
    """최신 초안 revision을 확정하고 PDF를 생성하는 API 입력 모델이다."""

    finalized_by: str | None = Field(default=None, min_length=1)
    # 전사문에 없는 초안 수치(numeric_verification.findings)를 확인했다는 표시다. 남아 있으면 필수다.
    numeric_review_confirmed: bool = False
    # 사용자가 화면에서 본 revision이다. 그 뒤 다른 수정이 있었으면 409로 거부한다.
    expected_revision: int | None = Field(default=None, ge=1)


class JobError(BaseModel):
    """실패한 작업의 오류 코드와 설명이다.

    ``code``: ``stt_unavailable``, ``redaction_failed``, ``draft_generation_unavailable``,
    ``internal_error``, ``interrupted``(서버 중단 등으로 갱신이 멈춤), ``server_shutdown``, ``queue_full``.
    """
    code: str
    message: str


class JobLinks(BaseModel):
    self: str
    events: str
    draft: str | None = None


class WorkflowJob(BaseModel):
    """초안 생성 비동기 작업의 상태다.

    ``status``: ``queued`` → ``running`` → ``succeeded`` | ``failed``.
    ``stage``: 실행 중인 단계(``stt``, ``redaction``, ``transcript_term_review``, ``draft_generation``,
    ``draft_review``, ``saving_draft``)이고, ``completed_stages``는 끝난 단계다.
    ``result``는 ``succeeded``일 때만 있으며 초안 생성 결과(``draft_id`` 포함)다.
    """
    job_id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str | None = None
    completed_stages: list[str] = Field(default_factory=list)
    trace_id: str
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: JobError | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime
    links: JobLinks
