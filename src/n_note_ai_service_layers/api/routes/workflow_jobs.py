"""음성 → 간호기록 초안 생성을 비동기 작업으로 접수하고, 진행 상태·결과를 조회·구독하는 API이다.

흐름: ``POST /v1/nursing-note-workflows``(202, 작업 ID) → ``GET /v1/nursing-note-jobs/{job_id}`` 폴링
또는 ``GET /v1/nursing-note-jobs/{job_id}/events``(SSE) 구독 → ``succeeded``면 ``result.draft_id``로 검토 시작.

``schema_ids``로 양식을 여러 개 요청하면 STT·비식별화·전사문 검토를 한 번만 하고 양식별 초안을 병렬로 만든다.
양식별 진행은 ``forms``에 실시간으로 남고, 먼저 끝난 양식은 ``forms[].draft_id``로 바로 검토를 시작할 수 있다.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.audio_preprocessing import audio_creation_time
from n_note_ai_service_layers.api import mappers
from n_note_ai_service_layers.api.deps import get_nursing_note_workflow, get_workflow_context_flow, parse_request_emr, read_validated_audio
from n_note_ai_service_layers.api.schemas import WorkflowJob
from n_note_ai_service_layers.api.state import get_services
from n_note_ai_service_layers.contracts.audio import AudioInput, PreTranscribedAudio
from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery
from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus
from n_note_ai_service_layers.observability import get_logger, log_event
from n_note_ai_service_layers.services.clock import now_local, now_utc
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.jobs import TERMINAL_STATUSES, JobFailure, JobQueueFull, Progress
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow, NursingNoteWorkflowResult, WorkflowStageError
from n_note_ai_service_layers.services.recording_time import parse_recorded_at, resolve_recording_time

router = APIRouter(tags=["nursing-note"])
JOB_KIND = "nursing_note_workflow"
QUEUE_FULL_RETRY_SECONDS = 5
# 이 worker 밖(다른 worker)에서 바뀐 작업 상태를 다시 읽어 보는 간격이다. 이 worker의 변경은 바로 알린다.
_SSE_RECHECK_SECONDS = 2.0
_SSE_KEEPALIVE_SECONDS = 15.0
# 실패 단계별 오류 코드다. 본문·예외 메시지는 노출하지 않는다.
_logger = get_logger("workflow_jobs")
_STAGE_FAILURES = {
    "audio_decode": ("audio_undecodable", "audio could not be decoded; send a supported, undamaged audio file"),
    "stt": ("stt_unavailable", "speech transcription is unavailable"),
    "redaction": ("redaction_failed", "transcript de-identification failed; nothing was sent to external services"),
    "draft_generation": ("draft_generation_unavailable", "draft generation is unavailable"),
}


def stage_failure(error: Exception) -> tuple[str, str]:
    """실패를 작업 상태의 (code, message)로 바꾼다. ``WorkflowStageError``는 단계별 코드, 그 밖의 예외는 ``internal_error``다."""
    if isinstance(error, WorkflowStageError):
        return _STAGE_FAILURES.get(error.stage, ("workflow_failed", "nursing note workflow failed"))
    return "internal_error", "form draft failed"


def resolve_workflow_form(schema_id: str | None, purpose: str | None) -> tuple[str, str]:
    """요청 양식이 없으면 기본 양식(N_NOTE_DEFAULT_FORM)을, 목적이 없으면 활성 양식의 목적을 사용한다."""
    schema_id = schema_id or settings.get().default_form_id
    if not schema_id:
        raise HTTPException(status_code=422, detail="schema_id is required when N_NOTE_DEFAULT_FORM is not set")
    if purpose:
        return schema_id, purpose
    active = [item for item in get_services().template_registry.list_versions(schema_id) if item.status is SchemaStatus.ACTIVE]
    if len(active) != 1:
        raise HTTPException(status_code=422, detail="workflow schema or context is not available")
    return schema_id, active[0].purpose


def job_response(job: dict[str, Any]) -> dict[str, Any]:
    links: dict[str, Any] = {"self": f"/v1/nursing-note-jobs/{job['job_id']}", "events": f"/v1/nursing-note-jobs/{job['job_id']}/events"}
    if job.get("status") == "succeeded" and job.get("result") and job["result"].get("draft_id"):
        links["draft"] = f"/v1/nursing-note-drafts/{job['result']['draft_id']}"
    if job.get("forms") is not None:
        links["drafts"] = [f"/v1/nursing-note-drafts/{form['draft_id']}" for form in job["forms"] if form.get("draft_id")]
    return {**job, "links": links}


def check_form_selection(schema_id: str | None, schema_ids: list[str] | None, schema_version: str | None) -> None:
    """양식은 ``schema_id``(+``schema_version``) 하나 또는 ``schema_ids`` 목록 중 하나로만 고른다."""
    if schema_ids and (schema_id or schema_version):
        raise HTTPException(status_code=422, detail="use either schema_id/schema_version or schema_ids")


def requested_form_ids(values: list[str]) -> list[str]:
    """반복 필드(``schema_ids=a&schema_ids=b``)와 쉼표 구분(``a,b``)을 모두 받아 순서를 지키며 중복을 없앤다."""
    ids = [item.strip() for value in values for item in value.split(",") if item.strip()]
    unique = list(dict.fromkeys(ids))
    if not unique:
        raise HTTPException(status_code=422, detail="schema_ids must name at least one form")
    limit = settings.get().max_forms_per_job
    if len(unique) > limit:
        raise HTTPException(status_code=422, detail=f"schema_ids may name at most {limit} forms")
    return unique


def _save_draft(result: NursingNoteWorkflowResult, schema: FormSchema, trace_id: str, patient_id: str, encounter_id: str) -> dict[str, Any]:
    """생성 결과를 초안 검토 레코드로 저장하고 작업 결과 사전을 만든다."""
    review = get_services().draft_review_store.create(
        schema,
        result.transcript.text,
        result.draft.proposed_fields,
        mappers.candidate_ids(result.transcript_gateway_verification, result.draft_gateway_verification),
        trace_id,
        mappers.draft_candidates(result.transcript_gateway_verification, result.draft_gateway_verification),
        patient_id,
        encounter_id,
        result.numeric_sources,
        result.reference_time,
        result.llm_applied_term_hints,
        result.emr_context,
    )
    return mappers.workflow_result_response(result, review, schema, trace_id)


def _workflow_job(
    workflow: NursingNoteWorkflow, context, audio: AudioInput | PreTranscribedAudio, schema: FormSchema, patient_id: str, encounter_id: str,
):
    """작업 루프에서 실행할 함수다: 초안 생성 → 초안 검토 레코드 저장 → 결과 사전."""
    async def work(progress: Progress) -> dict[str, Any]:
        try:
            result = await _call_workflow(
                workflow, "generate", context, audio, schema, known_identifiers=(patient_id, encounter_id), progress=progress,
            )
        except WorkflowStageError as exc:
            raise JobFailure(*stage_failure(exc)) from exc
        await asyncio.to_thread(progress, "saving_draft")
        return await asyncio.to_thread(_save_draft, result, schema, audio.trace_id, patient_id, encounter_id)
    return work


async def _call_workflow(workflow: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """async 워크플로(``agenerate``·``agenerate_many``)는 작업 루프에서 기다리고, 동기 대역(테스트)은 스레드에서 부른다."""
    asynchronous = getattr(workflow, f"a{name}", None)
    if asynchronous is not None:
        return await asynchronous(*args, **kwargs)
    return await asyncio.to_thread(getattr(workflow, name), *args, **kwargs)


def _multi_form_workflow_job(
    workflow: NursingNoteWorkflow, forms: list[tuple[FormSchema, Any]], audio: AudioInput | PreTranscribedAudio, patient_id: str, encounter_id: str, job_id: str,
):
    """여러 양식 작업이다: 공통 단계 한 번 → 양식별 초안을 병렬 생성하고 끝나는 대로 저장해 ``forms``에 알린다."""
    jobs = get_services().jobs
    # 양식 스레드들이 같은 작업 문서를 읽고 고쳐 쓰므로, 갱신이 서로 덮어쓰지 않게 직렬화한다.
    lock = threading.Lock()
    responses: list[dict[str, Any] | None] = [None] * len(forms)

    def update_form(index: int, **changes: Any) -> None:
        with lock:
            states = list((jobs.get(job_id) or {}).get("forms") or [])
            states[index] = {**states[index], **changes}
            jobs.update(job_id, forms=states)

    async def work(progress: Progress) -> dict[str, Any]:
        def notify(stage: str) -> None:
            with lock:
                progress(stage)

        def on_stage(index: int, stage: str) -> None:
            update_form(index, status="running", stage=stage)

        def on_done(index: int, outcome: NursingNoteWorkflowResult | Exception) -> None:
            schema = forms[index][0]
            if isinstance(outcome, NursingNoteWorkflowResult):
                try:
                    on_stage(index, "saving_draft")
                    response = _save_draft(outcome, schema, audio.trace_id, patient_id, encounter_id)
                    responses[index] = response
                    update_form(index, status="succeeded", stage=None, draft_id=response["draft_id"])
                    return
                except Exception as exc:
                    outcome = exc
            code, message = stage_failure(outcome)
            log_event(_logger, "form_draft_failed", trace_id=audio.trace_id, schema_id=schema.schema_id, code=code, error=type(outcome).__name__)
            update_form(index, status="failed", stage=None, error={"code": code, "message": message})

        try:
            await _call_workflow(
                workflow, "generate_many", audio, forms, known_identifiers=(patient_id, encounter_id), progress=notify,
                on_form_stage=on_stage, on_form_done=on_done, max_parallel=settings.get().form_concurrency,
            )
        except WorkflowStageError as exc:
            raise JobFailure(*stage_failure(exc)) from exc
        if not any(responses):
            raise JobFailure("all_forms_failed", "every selected form failed; see forms[].error")
        states = await asyncio.to_thread(lambda: (jobs.get(job_id) or {}).get("forms") or [])
        return {
            "trace_id": audio.trace_id,
            "forms": [
                {
                    "schema_id": schema.schema_id, "schema_version": schema.version,
                    "status": "succeeded" if response else "failed",
                    "result": response, "error": None if response else states[index].get("error"),
                }
                for index, ((schema, _), response) in enumerate(zip(forms, responses, strict=True))
            ],
        }
    return work


@router.post("/v1/nursing-note-workflows", status_code=status.HTTP_202_ACCEPTED, response_model=WorkflowJob)
async def create_nursing_note_workflow(
    response: Response,
    audio: UploadFile = File(..., description="간호기록 초안에 사용할 음성 파일"),
    patient_id: str = Form(..., min_length=1),
    encounter_id: str = Form(..., min_length=1),
    purpose: str | None = Form(default=None, min_length=1),
    schema_id: str | None = Form(default=None, min_length=1, description="양식 ID. 생략하면 N_NOTE_DEFAULT_FORM"),
    schema_ids: list[str] | None = Form(
        default=None,
        description="여러 양식을 한 음성으로 만들 때의 양식 ID 목록(반복 필드 또는 쉼표 구분). schema_id·schema_version과 함께 쓸 수 없고, 각 양식의 활성 버전을 쓴다",
    ),
    schema_version: str | None = Form(default=None, min_length=1),
    language_hint: str | None = Form(default=None, max_length=16),
    recorded_at: str | None = Form(
        default=None, max_length=64,
        description="녹음 시작 시각(ISO 8601, 예: 2026-11-18T09:00:00+09:00). 생략하면 음성 파일 메타데이터의 생성 시각, 그것도 없으면 접수 시각을 쓴다",
    ),
    emr: str | None = Form(
        default=None,
        description="bridge가 보낸 EMR JSON 객체 문자열(선택). 중첩·배열을 받으며, 식별 정보 키는 빼고 비식별화한 뒤 모든 양식 초안의 참고 문맥으로 쓴다",
    ),
    x_trace_id: str | None = Header(default=None),
    idempotency_key: str | None = Header(default=None, max_length=128, description="같은 키로 다시 보내면 새 작업을 만들지 않고 기존 작업을 반환한다"),
    context_flow: ClinicalContextFlow = Depends(get_workflow_context_flow),
    workflow: NursingNoteWorkflow = Depends(get_nursing_note_workflow),
) -> dict[str, Any]:
    """음성을 검증하고 초안 생성 작업을 접수한다. 처리는 백그라운드에서 진행되며 즉시 202를 반환한다."""
    # 음성을 읽기 전에 거부한다(prepare_workflow_forms도 같은 검사를 한다).
    check_form_selection(schema_id, schema_ids, schema_version)
    received_at = now_local()
    requested_time = _parse_recorded_at(recorded_at)
    emr_data = parse_request_emr(emr)
    content = await read_validated_audio(audio)
    plan = await prepare_workflow_forms(patient_id, encounter_id, purpose, schema_id, schema_ids, schema_version, x_trace_id, context_flow)
    metadata_time = None if requested_time else await run_in_threadpool(audio_creation_time, content)
    recording = resolve_recording_time(requested_time, metadata_time, received_at)
    log_event(_logger, "recording_time_resolved", trace_id=plan.trace_id, source=recording.source)
    source = AudioInput(
        content, audio.filename or "audio", audio.content_type, plan.trace_id, language_hint, recorded_at=recording.value, emr=emr_data,
    )
    job = await run_in_threadpool(submit_workflow_job, plan, source, workflow, idempotency_key)
    response.headers["Location"] = f"/v1/nursing-note-jobs/{job['job_id']}"
    return job_response(job)


def _parse_recorded_at(value: str | None):
    if value is None or not value.strip():
        return None
    try:
        return parse_recorded_at(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="recorded_at must be an ISO 8601 date-time") from exc


@dataclass(frozen=True)
class WorkflowPlan:
    """접수 전에 확인한 양식·요청 문맥이다. ``multi``면 여러 양식 작업(``forms``가 양식별 진행을 가진다)이다."""

    trace_id: str
    patient_id: str
    encounter_id: str
    forms: tuple[tuple[FormSchema, Any], ...]
    multi: bool
    purpose: str | None = None


async def prepare_workflow_forms(
    patient_id: str, encounter_id: str, purpose: str | None, schema_id: str | None, schema_ids: list[str] | None,
    schema_version: str | None, trace_id: str | None, context_flow: ClinicalContextFlow,
) -> WorkflowPlan:
    """요청한 양식의 활성 버전과 요청 문맥(접근 검사)을 준비한다. 쓸 수 없는 양식은 작업을 만들기 전에 422로 거부한다."""
    trace_id = trace_id or str(uuid4())
    check_form_selection(schema_id, schema_ids, schema_version)
    requested: list[tuple[str | None, str | None]]
    if schema_ids:
        requested = [(form_id, None) for form_id in requested_form_ids(schema_ids)]
    else:
        requested = [(schema_id, schema_version)]
    forms: list[tuple[FormSchema, Any]] = []
    resolved_purpose = purpose
    def resolve(form_id: str | None, version: str | None) -> tuple[FormSchema, Any, str]:
        # 양식 레지스트리는 저장소(SQLite) 변경을 확인하므로 API 이벤트 루프가 아니라 스레드에서 부른다.
        form_id, form_purpose = resolve_workflow_form(form_id, purpose)
        query = ClinicalContextQuery(patient_id, encounter_id, form_purpose, now_utc(), form_id, trace_id, version)
        try:
            context = context_flow.build(query)
            return context_flow.registry.resolve_active(form_id, form_purpose, version), context, form_purpose
        except (PermissionError, LookupError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="workflow schema or context is not available") from exc

    for form_id, version in requested:
        selected_schema, context, resolved_purpose = await run_in_threadpool(resolve, form_id, version)
        forms.append((selected_schema, context))
    return WorkflowPlan(trace_id, patient_id, encounter_id, tuple(forms), bool(schema_ids), resolved_purpose)


def submit_workflow_job(
    plan: WorkflowPlan, source: AudioInput | PreTranscribedAudio, workflow: NursingNoteWorkflow, idempotency_key: str | None = None,
) -> dict[str, Any]:
    """준비한 양식으로 초안 생성 작업을 만들고 실행 대기열에 넣는다(작업 스레드에서 호출한다). 대기열이 차면 429다."""
    services = get_services()
    try:
        services.job_runner.ensure_capacity()
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(QUEUE_FULL_RETRY_SECONDS)}) from exc
    if plan.multi:
        forms = list(plan.forms)
        request = {
            "schema_ids": [schema.schema_id for schema, _ in forms],
            "forms": [{"schema_id": schema.schema_id, "schema_version": schema.version, "purpose": schema.purpose} for schema, _ in forms],
        }
        initial_forms = [
            {"schema_id": schema.schema_id, "schema_version": schema.version, "status": "queued", "stage": None, "draft_id": None, "error": None}
            for schema, _ in forms
        ]
        job, created = services.jobs.create(JOB_KIND, plan.trace_id, request, idempotency_key, {"forms": initial_forms})
        work = _multi_form_workflow_job(workflow, forms, source, plan.patient_id, plan.encounter_id, job["job_id"]) if created else None
    else:
        schema, context = plan.forms[0]
        request = {"schema_id": schema.schema_id, "schema_version": schema.version, "purpose": plan.purpose}
        job, created = services.jobs.create(JOB_KIND, plan.trace_id, request, idempotency_key)
        work = _workflow_job(workflow, context, source, schema, plan.patient_id, plan.encounter_id) if created else None
    if work is not None:
        try:
            services.job_runner.submit(job["job_id"], work)
        except JobQueueFull as exc:
            services.jobs.update(job["job_id"], status="failed", error={"code": "queue_full", "message": str(exc)})
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(QUEUE_FULL_RETRY_SECONDS)}) from exc
    return job


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
        # 이 worker에서 상태가 바뀌면 바로 깨어난다. 다른 worker(다중 worker)가 바꾼 상태는 _SSE_RECHECK_SECONDS마다 읽어 본다.
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()
        def notify() -> None:
            loop.call_soon_threadsafe(changed.set)

        unsubscribe = jobs.subscribe(job_id, notify)
        try:
            last: tuple[Any, ...] | None = None
            last_sent = time.monotonic()
            while not await request.is_disconnected():
                changed.clear()
                job = await run_in_threadpool(jobs.get, job_id)
                if job is None:
                    return
                # 여러 양식 작업은 양식별 진행(먼저 끝난 초안의 draft_id 포함)이 바뀌어도 알린다.
                forms = tuple((form.get("status"), form.get("stage"), form.get("draft_id")) for form in job.get("forms") or ())
                current = (job["status"], job["stage"], forms)
                terminal = job["status"] in TERMINAL_STATUSES
                if current != last:
                    last, last_sent = current, time.monotonic()
                    event = job["status"] if terminal else "progress"
                    yield f"event: {event}\ndata: {json.dumps(job_response(job), ensure_ascii=False, default=str)}\n\n"
                    if terminal:
                        return
                elif time.monotonic() - last_sent >= _SSE_KEEPALIVE_SECONDS:
                    last_sent = time.monotonic()
                    yield ": keep-alive\n\n"
                try:
                    await asyncio.wait_for(changed.wait(), timeout=_SSE_RECHECK_SECONDS)
                except TimeoutError:
                    pass
        finally:
            unsubscribe()

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
