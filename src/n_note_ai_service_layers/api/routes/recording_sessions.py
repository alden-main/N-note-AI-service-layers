"""녹음하는 동안 청크 단위로 음성을 받아 전사문을 바로 돌려주고, 녹음이 끝나면 초안 생성 작업을 시작하는 API이다.

흐름: ``POST /v1/recording-sessions``(세션 열기) → 녹음 중 ``POST .../chunks``를 청크마다 호출(응답의 누적
``transcript``를 화면에 표시) → 녹음 종료 시 ``POST .../workflows``(202, 작업 ID) → 이후는 ``/v1/nursing-note-jobs``와 같다.

청크는 각각 따로 재생할 수 있는 음성 파일이어야 한다(예: 청크마다 MediaRecorder를 새로 시작하거나 WAV로 인코딩).
작업은 청크 전사문을 이어 붙여 시작하므로 STT를 다시 하지 않는다. 녹음 중에는 양식 초안(LLM)을 만들지 않는다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from n_note_ai_service_layers.api.deps import get_nursing_note_workflow, get_speech_to_text_provider, get_transcript_redactor, get_workflow_context_flow, read_validated_audio
from n_note_ai_service_layers.api.routes.workflow_jobs import job_response, prepare_workflow_forms, submit_workflow_job
from n_note_ai_service_layers.api.schemas import (
    RecordingChunkResponse, RecordingSessionCreateRequest, RecordingSessionResponse, RecordingWorkflowRequest, WorkflowJob,
)
from n_note_ai_service_layers.api.state import get_services
from n_note_ai_service_layers.contracts.ai_provider import SpeechToTextProvider
from n_note_ai_service_layers.contracts.audio import AudioInput, PreTranscribedAudio
from n_note_ai_service_layers.observability import get_logger, stage
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from n_note_ai_service_layers.services.recording_sessions import RecordingSession, RecordingSessionError
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor

router = APIRouter(tags=["recording-sessions"])
_logger = get_logger("api.recording_sessions")
_ERROR_STATUS = {
    "too_many_sessions": 429,
    "session_finished": 409,
    "sequence_gap": 409,
    "too_many_chunks": 422,
    "empty_transcript": 422,
}


def _http_error(exc: RecordingSessionError) -> HTTPException:
    return HTTPException(status_code=_ERROR_STATUS.get(exc.code, status.HTTP_409_CONFLICT), detail=exc.message)


def _session(session_id: str) -> RecordingSession:
    session = get_services().recording_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="recording session not found")
    return session


def _session_response(session: RecordingSession) -> dict[str, Any]:
    base = f"/v1/recording-sessions/{session.session_id}"
    return {
        "session_id": session.session_id,
        "trace_id": session.trace_id,
        "status": "workflow_started" if session.job_id else "recording",
        "next_sequence": session.next_sequence,
        "chunk_count": len(session.chunks),
        "transcript": session.redacted_transcript,
        "job_id": session.job_id,
        "links": {
            "self": base, "chunks": f"{base}/chunks", "workflows": f"{base}/workflows",
            "job": f"/v1/nursing-note-jobs/{session.job_id}" if session.job_id else None,
        },
    }


@router.post("/v1/recording-sessions", status_code=status.HTTP_201_CREATED, response_model=RecordingSessionResponse)
def create_recording_session(body: RecordingSessionCreateRequest, response: Response) -> dict[str, Any]:
    """녹음을 시작할 때 세션을 연다. 마지막 청크 이후 ``N_NOTE_RECORDING_SESSION_IDLE_SECONDS``가 지나면 사라진다."""
    try:
        session = get_services().recording_sessions.create(body.patient_id, body.encounter_id, body.language_hint)
    except RecordingSessionError as exc:
        raise _http_error(exc) from exc
    response.headers["Location"] = f"/v1/recording-sessions/{session.session_id}"
    return _session_response(session)


@router.get("/v1/recording-sessions/{session_id}", response_model=RecordingSessionResponse)
def get_recording_session(session_id: str) -> dict[str, Any]:
    """지금까지의 화면용 전사문과 다음 청크 번호다. 화면을 새로 열었을 때 이어서 녹음하는 데 쓴다."""
    return _session_response(_session(session_id))


@router.delete("/v1/recording-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recording_session(session_id: str) -> Response:
    """녹음을 취소하고 모은 전사문을 지운다. 이미 시작한 초안 생성 작업은 취소하지 않는다."""
    if not get_services().recording_sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="recording session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/v1/recording-sessions/{session_id}/chunks", response_model=RecordingChunkResponse)
async def add_recording_chunk(
    session_id: str,
    audio: UploadFile = File(..., description="따로 재생할 수 있는 음성 청크(권장 20~30초)"),
    sequence: int = Form(..., ge=0, description="0부터 1씩 늘어나는 청크 번호. 응답을 못 받았으면 같은 번호로 다시 보낸다"),
    provider: SpeechToTextProvider = Depends(get_speech_to_text_provider),
    redactor: TranscriptRedactor = Depends(get_transcript_redactor),
) -> dict[str, Any]:
    """청크를 전사·비식별화해 세션에 붙이고, 이 청크의 전사문과 누적 전사문을 돌려준다.

    청크는 순서대로 하나씩 처리한다. 다음 번호가 아니면 409이고, 이미 받은 번호면 다시 전사하지 않는다.
    """
    sessions = get_services().recording_sessions
    session = _session(session_id)
    content = await read_validated_audio(audio)
    identifiers = (session.patient_id, session.encounter_id)

    def transcribe(preceding_text: str | None):
        source = AudioInput(content, audio.filename or "audio", audio.content_type, session.trace_id, session.language_hint, preceding_text)
        with stage(_logger, session.trace_id, "stt_chunk", sequence=sequence, audio_bytes=len(content)):
            try:
                return provider.transcribe(source)
            except Exception as exc:
                raise HTTPException(status_code=503, detail="speech transcription is unavailable") from exc

    def redact(text: str):
        try:
            return redactor.redact(text, identifiers)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="transcript de-identification failed") from exc

    try:
        chunk, created = await run_in_threadpool(sessions.add_chunk, session, sequence, transcribe, redact)
    except RecordingSessionError as exc:
        raise _http_error(exc) from exc
    return {
        "session_id": session.session_id, "sequence": chunk.sequence, "text": chunk.redacted_text,
        "redaction_count": chunk.redaction_count, "duplicate": not created,
        "next_sequence": session.next_sequence, "transcript": session.redacted_transcript,
    }


@router.post("/v1/recording-sessions/{session_id}/workflows", status_code=status.HTTP_202_ACCEPTED, response_model=WorkflowJob)
async def finish_recording_session(
    session_id: str,
    body: RecordingWorkflowRequest,
    response: Response,
    context_flow: ClinicalContextFlow = Depends(get_workflow_context_flow),
    workflow: NursingNoteWorkflow = Depends(get_nursing_note_workflow),
) -> dict[str, Any]:
    """녹음을 마치고 모은 전사문으로 초안 생성 작업을 시작한다(STT 단계 없이 비식별화부터).

    다시 호출하면 새 작업을 만들지 않고 처음 시작한 작업을 돌려준다. 시작한 뒤에는 청크를 더 받지 않는다.
    """
    services = get_services()
    session = _session(session_id)
    job = services.jobs.get(session.job_id) if session.job_id else None
    if job is None:
        plan = await prepare_workflow_forms(
            session.patient_id, session.encounter_id, body.purpose, body.schema_id, body.schema_ids, body.schema_version,
            session.trace_id, context_flow,
        )

        def submit(transcript) -> str:
            return submit_workflow_job(plan, PreTranscribedAudio(transcript, session.trace_id), workflow)["job_id"]

        try:
            job_id, _ = await run_in_threadpool(services.recording_sessions.start_workflow, session, submit)
        except RecordingSessionError as exc:
            raise _http_error(exc) from exc
        job = services.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="nursing note job not found")
    response.headers["Location"] = f"/v1/nursing-note-jobs/{job['job_id']}"
    return job_response(job)
