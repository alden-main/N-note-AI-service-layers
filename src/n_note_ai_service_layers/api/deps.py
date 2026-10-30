"""엔드포인트가 주입받는 외부 provider·저장소와 공통 입력 검증이다."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, Request, UploadFile

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.note_artifact_store import FileNursingNoteStore
from n_note_ai_service_layers.adapters.pdf_renderer import NursingNotePdfRenderer
from n_note_ai_service_layers.adapters.transcript_only_repository import TranscriptOnlyRepository
from n_note_ai_service_layers.api.state import get_services
from n_note_ai_service_layers.contracts.ai_provider import SpeechToTextProvider
from n_note_ai_service_layers.contracts.form_schema import SchemaStatus
from n_note_ai_service_layers.observability import get_logger, log_event
from n_note_ai_service_layers.services.access_policy import PurposeAccessPolicy
from n_note_ai_service_layers.services.context_assembler import ClinicalContextAssembler
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.gateway_term_verification import GatewayTermVerificationFlow
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from n_note_ai_service_layers.services.report_validation import FormReportValidationFlow
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg", "audio/flac", "video/mp4"})
_logger = get_logger("api")


async def read_validated_audio(audio: UploadFile) -> bytes:
    """MIME 형식과 크기(25 MiB)를 확인하고 음성 바이트를 반환한다."""
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="unsupported audio content type")
    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail="audio file is empty")
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio file exceeds 25 MiB limit")
    return content


def get_speech_to_text_provider() -> SpeechToTextProvider:
    """설정한 STT 서버(OpenAI 또는 로컬 Whisper)로 STT provider를 만든다."""
    try:
        return model_providers.speech_to_text_provider()
    except Exception as exc:
        log_event(_logger, "stt_not_configured", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="speech transcription is not configured") from exc


@lru_cache(maxsize=4)
def _gateway_client(url: str, timeout_seconds: float, catalog_ttl_seconds: float, expected_tools: tuple[str, ...]) -> McpGatewayClient:
    return McpGatewayClient(url, timeout_seconds, catalog_ttl_seconds=catalog_ttl_seconds, expected_tools=expected_tools)


def _expected_gateway_tools() -> tuple[str, ...]:
    """workflow가 쓰는 Gateway 도구다. 하나라도 catalog에 없으면(MCP 기동 중·장애) catalog를 자주 다시 조회한다."""
    names = [*settings.term_review_tool_names()]
    if settings.report_validation_tool_name():
        names.append(settings.report_validation_tool_name())
    if settings.llm_tool_calling_enabled():
        names.extend(settings.llm_tool_names())
    return tuple(sorted(set(names)))


def get_gateway_client() -> McpGatewayClient:
    """요청 간에 공유하는 Gateway 클라이언트다(커넥션 풀·catalog TTL 캐시)."""
    return _gateway_client(
        settings.mcp_gateway_url(), settings.mcp_gateway_timeout_seconds(), settings.mcp_gateway_catalog_ttl_seconds(), _expected_gateway_tools(),
    )


def get_note_store() -> FileNursingNoteStore:
    """확정 간호기록 JSON·PDF 저장소를 만든다."""
    return FileNursingNoteStore(settings.artifact_dir())


def get_pdf_renderer() -> NursingNotePdfRenderer:
    """설정한 한글 글꼴로 확정본 PDF renderer를 만든다."""
    return NursingNotePdfRenderer(settings.pdf_font_path())


def get_transcript_redactor() -> TranscriptRedactor:
    """외부 처리 경계에 전달할 전사문을 비식별화하는 서비스를 만든다(로컬 NER이 설정되면 함께 사용)."""
    return TranscriptRedactor(get_services().ner)


def get_workflow_context_flow() -> ClinicalContextFlow:
    """임상 문맥 흐름을 만든다. 전사문만으로 초안을 만들며, 운영 EMR repository는 별도 adapter로 주입한다."""
    registry = get_services().template_registry
    repository = TranscriptOnlyRepository()
    purposes = {schema.purpose for schema in registry.list_versions() if schema.status is SchemaStatus.ACTIVE}
    return ClinicalContextFlow(PurposeAccessPolicy(purposes), registry, repository, ClinicalContextAssembler())


def get_nursing_note_workflow(
    stt: SpeechToTextProvider = Depends(get_speech_to_text_provider),
) -> NursingNoteWorkflow:
    """Gateway를 유일한 외부 MCP 경계로 사용해 초안과 검토 artifact를 조립한다."""
    gateway = get_gateway_client()
    try:
        generator = model_providers.draft_generator(gateway)
    except Exception as exc:
        log_event(_logger, "draft_generation_not_configured", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="draft generation is not configured") from exc
    return NursingNoteWorkflow(
        stt, generator, GatewayTermVerificationFlow(gateway),
        redactor=get_transcript_redactor(), report_validation=FormReportValidationFlow(gateway),
        hint_budget_seconds=settings.term_hint_budget_seconds(),
    )


def get_authenticated_user(request: Request) -> str | None:
    """앞단(브리지·인증 프록시)이 인증해 넘긴 사용자 ID다(``N_NOTE_USER_HEADER``)."""
    value = request.headers.get(settings.user_header())
    return value.strip() if value and value.strip() else None


def resolve_actor(claimed: str | None, authenticated: str | None, field_name: str) -> str:
    """감사 기록에 남길 수행자를 정한다.

    인증 사용자가 있으면 그 값을 쓰고 본문 값이 다르면 403이다. 인증을 요구하는 설정에서 헤더가 없으면 401이다.
    """
    if authenticated:
        if claimed and claimed != authenticated:
            raise HTTPException(status_code=403, detail=f"{field_name} does not match the authenticated user")
        return authenticated
    if settings.require_authenticated_user():
        raise HTTPException(status_code=401, detail="authenticated user header is required")
    if not claimed:
        raise HTTPException(status_code=422, detail=f"{field_name} is required")
    return claimed
