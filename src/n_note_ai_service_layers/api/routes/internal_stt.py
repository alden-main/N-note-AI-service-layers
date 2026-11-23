"""간호노트 백엔드용 동기 STT 및 동적 서식 추출 API.

HTTP 입력 검사·응답과 서식별 동시 실행만 여기에 있다. 서식 추출 로직은 ``services/form_extraction.py``다.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.audio_preprocessing import audio_creation_time
from n_note_ai_service_layers.api.deps import get_transcript_preparation, parse_request_emr, read_audio
from n_note_ai_service_layers.contracts.audio import AudioInput
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.observability import span
from n_note_ai_service_layers.services import form_extraction, work_loop
from n_note_ai_service_layers.services.clock import now_local
from n_note_ai_service_layers.services.emr_context import EmrContext
from n_note_ai_service_layers.services.nursing_note_workflow import TranscriptPreparation, WorkflowStageError
from n_note_ai_service_layers.services.recording_time import parse_recorded_at, resolve_recording_time


class NumericVerification(BaseModel):
    status: str
    findings: list[dict[str, Any]]


class FormResult(BaseModel):
    templateType: str
    templateName: str
    fields: dict
    relatedSentences: dict[str, list[str]]
    numericVerification: NumericVerification


class InternalSttResponse(BaseModel):
    transcript: str
    recordedAt: str
    forms: list[FormResult]


router = APIRouter(tags=["internal-stt"])
_MAX_AUDIO = 50 * 1024 * 1024
_EXTENSIONS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".webm"}
# 공통 단계 실패를 이 API의 HTTP 응답으로 바꾼다. 본문·예외 메시지는 노출하지 않는다.
_PREPARATION_FAILURES = {
    "audio_decode": (415, "audio could not be decoded"),
    "stt": (503, "speech transcription is unavailable"),
    "redaction": (503, "transcript de-identification failed"),
}


async def _emr_part(form) -> str | None:
    """선택 part ``emr``(EMR JSON 객체)다. 텍스트 part 또는 application/json 파일로 받는다."""
    parts = form.getlist("emr")
    if len(parts) > 1:
        raise HTTPException(status_code=422, detail="emr must be a single JSON part")
    if not parts:
        return None
    part = parts[0]
    if isinstance(part, StarletteUploadFile):
        try:
            return (await part.read()).decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="emr must be UTF-8 JSON") from None
    return part


def _recorded_at_part(form) -> datetime | None:
    """선택 part ``recorded_at``(녹음 시작 시각, ISO 8601)이다. 시간대가 없으면 병동 시간대로 본다."""
    parts = form.getlist("recorded_at")
    if len(parts) > 1 or (parts and not isinstance(parts[0], str)):
        raise HTTPException(status_code=422, detail="recorded_at must be a single ISO 8601 text part")
    if not parts or not parts[0].strip():
        return None
    try:
        return parse_recorded_at(parts[0])
    except ValueError:
        raise HTTPException(status_code=422, detail="recorded_at must be an ISO 8601 date-time") from None


@router.post(
    "/internal/v1",
    summary="음성 전사 및 서식 필드 추출",
    response_model=InternalSttResponse,
    description="음성 전사 후 비식별화하고 Gateway 용어 검토 후보를 참고해 서식을 생성합니다. templates는 JSON 문자열 또는 application/json 파일로 전달할 수 있습니다. Gateway 검토가 불가능하면 503을 반환합니다.",
    openapi_extra={
        "requestBody": {"content": {"multipart/form-data": {"schema": {
            "type": "object",
            "required": ["file", "templates"],
            "properties": {
                "file": {"type": "string", "format": "binary", "description": "음성 파일 (최대 50 MiB)"},
                "templates": {"type": "string", "description": "서식 목록을 담은 JSON 배열", "example": '[{"templateType":"HAND_OVER","templateName":"인계기록지","schemaDefinition":{"sections":[]}}]'},
            },
        }}}},
    },
)
async def internal_transcription(request: Request, preparation: TranscriptPreparation = Depends(get_transcript_preparation)) -> dict:
    # 요청 하나가 추적 하나다. 공통 단계·양식별 추출 span이 이 아래에 놓인다(run_in_threadpool은 문맥을 넘긴다).
    with span("internal_stt"):
        return await _internal_transcription(request, preparation)


async def _internal_transcription(request: Request, preparation: TranscriptPreparation) -> dict:
    form = await request.form()
    parts = form.getlist("templates")
    if len(parts) != 1 or len(form.getlist("file")) != 1:
        raise HTTPException(status_code=422, detail="exactly one file and templates part are required")
    file = form.get("file")
    if not isinstance(file, StarletteUploadFile):
        raise HTTPException(status_code=422, detail="file must be an upload")
    part = parts[0]
    if isinstance(part, StarletteUploadFile):
        body = await part.read()
        try:
            raw = body.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="templates must be UTF-8 JSON") from None
    else:
        raw = part
    try:
        templates = form_extraction.parse_templates(raw)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid templates JSON or schema") from None
    received_at = now_local()
    requested_time = _recorded_at_part(form)
    if Path(file.filename or "").suffix.lower() not in _EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported audio file extension")
    # 백엔드 계약(docs/STT_API.md)은 확장자와 50 MiB 상한이다.
    content = await read_audio(file, _MAX_AUDIO)
    trace_id = str(uuid4())
    # 녹음 시각: 요청 recorded_at → 음성 메타데이터 생성 시각 → 접수 시각(초안 작업 API와 같은 규칙).
    metadata_time = None if requested_time else await run_in_threadpool(audio_creation_time, content)
    recording = resolve_recording_time(requested_time, metadata_time, received_at)
    emr = parse_request_emr(await _emr_part(form))
    audio = AudioInput(content, file.filename or "audio", file.content_type or "application/octet-stream", trace_id, "ko", recorded_at=recording.value, emr=emr)
    # 공통 단계·서식 추출은 서비스 작업 루프에서 돈다(API 루프는 HTTP만 처리한다).
    return await work_loop.run(_transcribe_and_extract(preparation, audio, templates, trace_id))


async def _transcribe_and_extract(preparation: TranscriptPreparation, audio: AudioInput, templates: list[dict], trace_id: str) -> dict:
    """작업 루프에서 실행한다: STT → 비식별화 → 전사문 검토(힌트 대기 상한까지) → 서식별 추출."""
    try:
        prepared = await preparation.aprepare(audio, (), lambda _stage: None)
    except WorkflowStageError as exc:
        status_code, detail = _PREPARATION_FAILURES.get(exc.stage, (503, "speech transcription is unavailable"))
        raise HTTPException(status_code=status_code, detail=detail) from exc
    # 이 API는 검토가 성공해야 결과를 돌려준다(docs/STT_API.md). 상한 안에 끝난 검토는 추출 전에 확인하고,
    # 늦은 검토는 추출을 먼저 시작한 뒤 확인한다(실패면 그 추출 결과는 버린다).
    reviewed_in_time = prepared.review.done()
    if reviewed_in_time:
        _require_review(prepared.review.result())
    forms = await _extract_forms(
        templates, prepared.transcript.text, prepared.hints, trace_id, settings.get().form_concurrency,
        hints_deferred=not reviewed_in_time, recorded_at=prepared.reference_time, emr=prepared.emr,
    )
    if not reviewed_in_time:
        _require_review(await prepared.review)
    return {"transcript": prepared.transcript.text, "recordedAt": prepared.reference_time.isoformat(), "forms": forms}


def _require_review(review: GatewayTermVerificationResult) -> None:
    """전사문 검토가 성공하지 않았으면(장애·미지원) 503이다."""
    if review.status != "ok":
        raise HTTPException(status_code=503, detail="term review is unavailable")


async def _extract_forms(
    templates: list[dict], transcript: str, hints: tuple[dict, ...], trace_id: str, concurrency: int, hints_deferred: bool,
    recorded_at: datetime | None = None, emr: EmrContext | None = None,
) -> list[dict]:
    """양식별 추출은 서로 독립이라 동시에 보낸다. 로컬 vLLM은 동시 요청을 묶어 처리하므로 총 시간이 가장 느린 양식 수준이 된다."""
    sentences = form_extraction.split_sentences(transcript)
    limit = asyncio.Semaphore(concurrency)
    recorded_at = recorded_at or now_local()

    async def extract_form(template: dict) -> dict:
        async with limit:
            # 추출은 동기 LLM 입구(complete)를 쓰므로 스레드에서 부른다.
            return await asyncio.to_thread(
                form_extraction.extract_form, template, sentences, transcript, hints, trace_id, recorded_at, hints_deferred, emr,
            )

    return list(await asyncio.gather(*(extract_form(template) for template in templates)))
