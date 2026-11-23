"""엔드포인트가 주입받는 외부 provider·저장소와 공통 입력 검증이다."""

from __future__ import annotations

import threading
from typing import Any

from fastapi import Depends, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.note_artifact_store import FileNursingNoteStore
from n_note_ai_service_layers.adapters.pdf_renderer import NursingNotePdfRenderer
from n_note_ai_service_layers.api.state import get_services
from n_note_ai_service_layers.contracts.ai_provider import SpeechToTextProvider
from n_note_ai_service_layers.contracts.form_schema import SchemaStatus
from n_note_ai_service_layers.observability import get_logger, log_event
from n_note_ai_service_layers.services.access_policy import PurposeAccessPolicy
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.emr_context import EmrInputError, parse_emr
from n_note_ai_service_layers.services.gateway_term_verification import GatewayTermVerificationFlow
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow, TranscriptPreparation
from n_note_ai_service_layers.services.report_validation import FormReportValidationFlow
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg", "audio/flac", "video/mp4"})
_logger = get_logger("api")


def audio_media_type(content_type: str | None) -> str:
    """``audio/webm;codecs=opus``처럼 매개변수가 붙은 MIME(브라우저 MediaRecorder 기본값)에서 형식만 꺼낸다."""
    return (content_type or "").split(";", 1)[0].strip().lower()


async def read_audio(audio: StarletteUploadFile, max_bytes: int) -> bytes:
    """음성 바이트를 ``max_bytes``까지만 읽는다. 비었으면 422, 넘으면 413이다(형식 검사는 호출자가 한다)."""
    content = await audio.read(max_bytes + 1)
    if not content:
        raise HTTPException(status_code=422, detail="audio file is empty")
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"audio file exceeds {max_bytes // (1024 * 1024)} MiB limit")
    return content


async def read_validated_audio(audio: UploadFile) -> bytes:
    """MIME 형식과 크기(25 MiB, OpenAI 전사 API 상한)를 확인하고 음성 바이트를 반환한다."""
    if audio_media_type(audio.content_type) not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="unsupported audio content type")
    return await read_audio(audio, MAX_AUDIO_BYTES)


def get_speech_to_text_provider() -> SpeechToTextProvider:
    """설정한 STT 서버(OpenAI 또는 로컬 Whisper)로 STT provider를 만든다."""
    try:
        return model_providers.speech_to_text_provider()
    except Exception as exc:
        log_event(_logger, "stt_not_configured", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="speech transcription is not configured") from exc


_gateway: tuple[tuple[Any, ...], McpGatewayClient] | None = None
_gateway_lock = threading.Lock()


def _gateway_client(url: str, timeout_seconds: float, catalog_ttl_seconds: float, expected_tools: tuple[str, ...]) -> McpGatewayClient:
    """설정이 같으면 클라이언트 하나를 공유한다. 설정이 바뀌면(테스트 등) 새로 만들고 이전 클라이언트의 연결을 닫는다."""
    global _gateway
    key = (url, timeout_seconds, catalog_ttl_seconds, expected_tools)
    with _gateway_lock:
        if _gateway is not None and _gateway[0] == key:
            return _gateway[1]
        previous = _gateway[1] if _gateway is not None else None
        client = McpGatewayClient(url, timeout_seconds, catalog_ttl_seconds=catalog_ttl_seconds, expected_tools=expected_tools)
        _gateway = (key, client)
    if previous is not None:
        previous.close()
    return client


def _expected_gateway_tools() -> tuple[str, ...]:
    """workflow가 쓰는 Gateway 도구다. 하나라도 catalog에 없으면(MCP 기동 중·장애) catalog를 자주 다시 조회한다."""
    config = settings.get()
    names = [*config.term_review_tool_names]
    if config.report_validation_tool_name:
        names.append(config.report_validation_tool_name)
    if config.llm_tool_calling_enabled:
        names.extend(config.llm_tool_names)
    return tuple(sorted(set(names)))


def get_gateway_client() -> McpGatewayClient:
    """요청 간에 공유하는 Gateway 클라이언트다(커넥션 풀·catalog TTL 캐시)."""
    config = settings.get()
    return _gateway_client(
        config.mcp_gateway_url, config.mcp_gateway_timeout_seconds, config.mcp_gateway_catalog_ttl_seconds, _expected_gateway_tools(),
    )


def get_term_verification(gateway: McpGatewayClient) -> GatewayTermVerificationFlow:
    """설정한 검토 도구·청크 크기로 Gateway 용어 검토 흐름을 만든다."""
    config = settings.get()
    return GatewayTermVerificationFlow(gateway, config.term_review_tool_names, config.term_review_chunk_chars)


def get_note_store() -> FileNursingNoteStore:
    """확정 간호기록 JSON·PDF 저장소를 만든다."""
    return FileNursingNoteStore(settings.get().artifact_dir)


def get_pdf_renderer() -> NursingNotePdfRenderer:
    """설정한 한글 글꼴로 확정본 PDF renderer를 만든다."""
    return NursingNotePdfRenderer(settings.get().pdf_font_path)


def get_transcript_redactor() -> TranscriptRedactor:
    """외부 처리 경계에 전달할 전사문을 비식별화하는 서비스를 만든다(로컬 NER이 설정되면 함께 사용)."""
    return TranscriptRedactor(get_services().ner)


def get_transcript_preparation(
    stt: SpeechToTextProvider = Depends(get_speech_to_text_provider),
    redactor: TranscriptRedactor = Depends(get_transcript_redactor),
    gateway: McpGatewayClient = Depends(get_gateway_client),
) -> TranscriptPreparation:
    """STT → 비식별화 → 전사문 용어 검토 공통 단계다(동기 서식 추출 API가 쓴다)."""
    config = settings.get()
    return TranscriptPreparation(stt, get_term_verification(gateway), redactor, config.term_hint_budget_seconds, config.emr_identifier_keys)


def get_workflow_context_flow() -> ClinicalContextFlow:
    """요청 문맥 흐름을 만든다(목적·식별자 접근 검사와 활성 양식 확인). EMR은 요청에 함께 오는 JSON으로 받는다."""
    registry = get_services().template_registry
    purposes = {schema.purpose for schema in registry.list_versions() if schema.status is SchemaStatus.ACTIVE}
    return ClinicalContextFlow(PurposeAccessPolicy(purposes), registry)


def get_nursing_note_workflow(
    stt: SpeechToTextProvider = Depends(get_speech_to_text_provider),
) -> NursingNoteWorkflow:
    """Gateway를 유일한 외부 MCP 경계로 사용해 초안과 검토 artifact를 조립한다."""
    config = settings.get()
    gateway = get_gateway_client()
    try:
        generator = model_providers.draft_generator(gateway)
    except Exception as exc:
        log_event(_logger, "draft_generation_not_configured", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="draft generation is not configured") from exc
    return NursingNoteWorkflow(
        stt, generator, get_term_verification(gateway),
        redactor=get_transcript_redactor(), report_validation=FormReportValidationFlow(gateway, config.report_validation_tool_name),
        hint_budget_seconds=config.term_hint_budget_seconds, emr_identifier_keys=config.emr_identifier_keys,
    )


def get_authenticated_user(request: Request) -> str | None:
    """앞단(브리지·인증 프록시)이 인증해 넘긴 사용자 ID다(``N_NOTE_USER_HEADER``)."""
    value = request.headers.get(settings.get().user_header)
    return value.strip() if value and value.strip() else None


def resolve_actor(claimed: str | None, authenticated: str | None, field_name: str) -> str:
    """감사 기록에 남길 수행자를 정한다.

    인증 사용자가 있으면 그 값을 쓰고 본문 값이 다르면 403이다. 인증을 요구하는 설정에서 헤더가 없으면 401이다.
    """
    if authenticated:
        if claimed and claimed != authenticated:
            raise HTTPException(status_code=403, detail=f"{field_name} does not match the authenticated user")
        return authenticated
    if settings.get().require_authenticated_user:
        raise HTTPException(status_code=401, detail="authenticated user header is required")
    if not claimed:
        raise HTTPException(status_code=422, detail=f"{field_name} is required")
    return claimed


def parse_request_emr(raw: str | bytes | dict[str, Any] | None) -> dict[str, Any] | None:
    """요청에 함께 온 EMR JSON을 검사한다(상한은 ``N_NOTE_EMR_MAX_*``). 형식이 틀리거나 상한을 넘으면 422다."""
    config = settings.get()
    try:
        return parse_emr(raw, config.emr_max_bytes, config.emr_max_entries, config.emr_max_depth)
    except EmrInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
