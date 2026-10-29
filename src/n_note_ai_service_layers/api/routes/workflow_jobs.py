"""음성 → 간호기록 초안 생성을 비동기 작업으로 접수하고, 진행 상태·결과를 조회·구독하는 API이다.

흐름: ``POST /v1/nursing-note-workflows``(202, 작업 ID) → ``GET /v1/nursing-note-jobs/{job_id}`` 폴링
또는 ``GET /v1/nursing-note-jobs/{job_id}/events``(SSE) 구독 → ``succeeded``면 ``result.draft_id``로 검토 시작.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.api import mappers
from n_note_ai_service_layers.api.deps import get_nursing_note_workflow, get_workflow_context_flow, read_validated_audio
from n_note_ai_service_layers.api.schemas import WorkflowJob
from n_note_ai_service_layers.api.state import get_services
from n_note_ai_service_layers.contracts.audio import AudioInput
from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery
from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.jobs import TERMINAL_STATUSES, JobFailure, JobQueueFull, Progress
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow, WorkflowStageError

router = APIRouter(tags=["nursing-note"])
JOB_KIND = "nursing_note_workflow"
QUEUE_FULL_RETRY_SECONDS = 5
_SSE_POLL_SECONDS = 0.5
_SSE_KEEPALIVE_SECONDS = 15.0
# 실패 단계별 오류 코드다. 본문·예외 메시지는 노출하지 않는다.
_STAGE_FAILURES = {
    "stt": ("stt_unavailable", "speech transcription is unavailable"),
    "redaction": ("redaction_failed", "transcript de-identification failed; nothing was sent to external services"),
    "draft_generation": ("draft_generation_unavailable", "draft generation is unavailable"),
}


def resolve_workflow_form(schema_id: str | None, purpose: str | None) -> tuple[str, str]:
    """요청 양식이 없으면 기본 양식(N_NOTE_DEFAULT_FORM)을, 목적이 없으면 활성 양식의 목적을 사용한다."""
    schema_id = schema_id or settings.default_form_id()
    if not schema_id:
        raise HTTPException(status_code=422, detail="schema_id is required when N_NOTE_DEFAULT_FORM is not set")
    if purpose:
        return schema_id, purpose
    active = [item for item in get_services().template_registry.list_versions(schema_id) if item.status is SchemaStatus.ACTIVE]
    if len(active) != 1:
        raise HTTPException(status_code=422, detail="workflow schema or context is not available")
    return schema_id, active[0].purpose


def job_response(job: dict[str, Any]) -> dict[str, Any]:
    links = {"self": f"/v1/nursing-note-jobs/{job['job_id']}", "events": f"/v1/nursing-note-jobs/{job['job_id']}/events"}
    if job.get("status") == "succeeded" and job.get("result"):
        links["draft"] = f"/v1/nursing-note-drafts/{job['result']['draft_id']}"
    return {**job, "links": links}


def _workflow_job(
    workflow: NursingNoteWorkflow, snapshot, audio: AudioInput, schema: FormSchema, patient_id: str, encounter_id: str,
):
    """작업 스레드에서 실행할 함수다: 초안 생성 → 초안 검토 레코드 저장 → 결과 사전."""
    def work(progress: Progress) -> dict[str, Any]:
        try:
            result = workflow.generate(snapshot, audio, schema, known_identifiers=(patient_id, encounter_id), progress=progress)
        except WorkflowStageError as exc:
            code, message = _STAGE_FAILURES.get(exc.stage, ("workflow_failed", "nursing note workflow failed"))
            raise JobFailure(code, message) from exc
        progress("saving_draft")
        review = get_services().draft_review_store.create(
            schema,
            result.transcript.text,
            result.draft.proposed_fields,
            mappers.candidate_ids(result.transcript_gateway_verification, result.draft_gateway_verification),
            audio.trace_id,
            mappers.draft_candidates(result.transcript_gateway_verification, result.draft_gateway_verification),
            patient_id,
            encounter_id,
            result.numeric_sources,
            result.reference_time,
            result.llm_applied_term_hints,
        )
        return mappers.workflow_result_response(result, review, schema, audio.trace_id)
    return work


@router.post("/v1/nursing-note-workflows", status_code=status.HTTP_202_ACCEPTED, response_model=WorkflowJob)
async def create_nursing_note_workflow(
    response: Response,
    audio: UploadFile = File(..., description="간호기록 초안에 사용할 음성 파일"),
    patient_id: str = Form(..., min_length=1),
    encounter_id: str = Form(..., min_length=1),
    purpose: str | None = Form(default=None, min_length=1),
    schema_id: str | None = Form(default=None, min_length=1, description="양식 ID. 생략하면 N_NOTE_DEFAULT_FORM"),
    schema_version: str | None = Form(default=None, min_length=1),
    language_hint: str | None = Form(default=None, max_length=16),
    x_trace_id: str | None = Header(default=None),
    idempotency_key: str | None = Header(default=None, max_length=128, description="같은 키로 다시 보내면 새 작업을 만들지 않고 기존 작업을 반환한다"),
    context_flow: ClinicalContextFlow = Depends(get_workflow_context_flow),
    workflow: NursingNoteWorkflow = Depends(get_nursing_note_workflow),
) -> dict[str, Any]:
    """음성을 검증하고 초안 생성 작업을 접수한다. 처리는 백그라운드에서 진행되며 즉시 202를 반환한다."""
    services = get_services()
    content = await read_validated_audio(audio)
    schema_id, purpose = resolve_workflow_form(schema_id, purpose)
    trace_id = x_trace_id or str(uuid4())
    query = ClinicalContextQuery(patient_id, encounter_id, purpose, datetime.now(timezone.utc), schema_id, trace_id, schema_version)
    try:
        snapshot = await run_in_threadpool(context_flow.build, query)
        selected_schema = context_flow.registry.resolve_active(schema_id, purpose, schema_version)
    except (PermissionError, LookupError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="workflow schema or context is not available") from exc
    try:
        services.job_runner.ensure_capacity()
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(QUEUE_FULL_RETRY_SECONDS)}) from exc
    job, created = await run_in_threadpool(
        services.jobs.create, JOB_KIND, trace_id,
        {"schema_id": selected_schema.schema_id, "schema_version": selected_schema.version, "purpose": purpose},
        idempotency_key,
    )
    if created:
        audio_input = AudioInput(content, audio.filename or "audio", audio.content_type, trace_id, language_hint)
        try:
            services.job_runner.submit(job["job_id"], _workflow_job(workflow, snapshot, audio_input, selected_schema, patient_id, encounter_id))
        except JobQueueFull as exc:
            services.jobs.update(job["job_id"], status="failed", error={"code": "queue_full", "message": str(exc)})
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(QUEUE_FULL_RETRY_SECONDS)}) from exc
    response.headers["Location"] = f"/v1/nursing-note-jobs/{job['job_id']}"
    return job_response(job)


@router.get("/v1/nursing-note-jobs/{job_id}", response_model=WorkflowJob)
def get_nursing_note_job(job_id: str) -> dict[str, Any]:
    """작업 상태를 반환한다. ``succeeded``면 ``result``에 초안 생성 결과가 들어 있다."""
    job = get_services().jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="nursing note job not found")
    return job_response(job)


@router.get("/v1/nursing-note-jobs/{job_id}/events", response_class=StreamingResponse)
async def stream_nursing_note_job(job_id: str, request: Request) -> StreamingResponse:
    """작업 진행을 Server-Sent Events로 보낸다: ``progress``(단계 변경)… → ``succeeded`` 또는 ``failed``에서 종료."""
    jobs = get_services().jobs
    if await run_in_threadpool(jobs.get, job_id) is None:
        raise HTTPException(status_code=404, detail="nursing note job not found")

    async def events() -> AsyncIterator[str]:
        last: tuple[Any, Any] | None = None
        idle = 0.0
        while not await request.is_disconnected():
            job = await run_in_threadpool(jobs.get, job_id)
            if job is None:
                return
            current = (job["status"], job["stage"])
            terminal = job["status"] in TERMINAL_STATUSES
            if current != last:
                last, idle = current, 0.0
                event = job["status"] if terminal else "progress"
                yield f"event: {event}\ndata: {json.dumps(job_response(job), ensure_ascii=False, default=str)}\n\n"
                if terminal:
                    return
            elif idle >= _SSE_KEEPALIVE_SECONDS:
                idle = 0.0
                yield ": keep-alive\n\n"
            await asyncio.sleep(_SSE_POLL_SECONDS)
            idle += _SSE_POLL_SECONDS

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
