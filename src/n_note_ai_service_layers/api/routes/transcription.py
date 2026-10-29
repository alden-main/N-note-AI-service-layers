"""음성 파일만 전사·비식별화해 돌려주는 API이다."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from n_note_ai_service_layers.api.deps import get_speech_to_text_provider, get_transcript_redactor, read_validated_audio
from n_note_ai_service_layers.api.schemas import TranscriptionResponse
from n_note_ai_service_layers.contracts.ai_provider import SpeechToTextProvider
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.observability import get_logger, stage
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor

router = APIRouter(tags=["speech-to-text"])
_logger = get_logger("api.transcription")


@router.post("/v1/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="전사할 음성 파일"),
    language_hint: str | None = Form(default=None, max_length=16),
    x_trace_id: str | None = Header(default=None),
    provider: SpeechToTextProvider = Depends(get_speech_to_text_provider),
    redactor: TranscriptRedactor = Depends(get_transcript_redactor),
) -> TranscriptionResponse:
    """음성 파일을 전사하고 응답에는 명시적 식별자를 치환한 텍스트만 포함한다."""
    content = await read_validated_audio(audio)
    trace_id = x_trace_id or str(uuid4())
    audio_input = AudioInput(content, audio.filename or "audio", audio.content_type, trace_id, language_hint)
    try:
        # STT SDK는 동기 호출이다. 이벤트 루프를 막지 않도록 스레드에서 실행한다.
        with stage(_logger, trace_id, "stt", audio_bytes=len(content)):
            transcript: Transcript = await run_in_threadpool(provider.transcribe, audio_input)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="speech transcription is unavailable") from exc
    redacted = redactor.redact(transcript.text)
    return TranscriptionResponse(
        text=redacted.text, provider=transcript.provider, model=transcript.model, language=transcript.language,
        trace_id=trace_id, redacted=True, redaction_count=redacted.replacement_count,
    )
