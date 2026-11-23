"""설정(OpenAI 또는 OpenAI 호환 로컬 서버)에 맞는 STT·초안 생성 어댑터를 만든다."""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.audio_preprocessing import DEFAULT_HALLUCINATION_PATTERNS, AudioPreprocessingSpeechToText, HallucinationFilter, VadConfig
from n_note_ai_service_layers.adapters.draft_generation import DraftGenerationAdapter
from n_note_ai_service_layers.adapters.llm_completion import GenerationOptions, ToolExecutor
from n_note_ai_service_layers.adapters.openai_stt import OpenAiSpeechToTextAdapter
from n_note_ai_service_layers.adapters.pydantic_ai_completion import PydanticAiCompletion
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient


class LocalSttRequired(RuntimeError):
    """``N_NOTE_REQUIRE_LOCAL_STT``인데 STT 서버가 외부 OpenAI API로 설정되어 있다."""


def openai_client(endpoint: settings.ModelEndpoint, timeout_seconds: float | None = None, max_retries: int | None = None) -> Any:
    """접속 정보로 OpenAI SDK 클라이언트를 만든다. 로컬 서버는 API 키가 없어도 된다.

    같은 접속 정보의 클라이언트는 재사용해 HTTP 커넥션 풀을 요청 간에 공유한다.
    """
    return _cached_client(endpoint, timeout_seconds, max_retries)


def _client_options(endpoint: settings.ModelEndpoint, timeout_seconds: float | None, max_retries: int | None) -> dict[str, Any]:
    """OpenAI SDK 클라이언트 생성 인자다. 로컬 서버는 API 키가 없어도 된다(``EMPTY``).

    ``base_url``이 없으면 OpenAI API이며 키가 없으면 SDK가 ``OPENAI_API_KEY``에서 읽는다(없으면 예외).
    """
    options: dict[str, Any] = {}
    if timeout_seconds is not None:
        options["timeout"] = timeout_seconds
    if max_retries is not None:
        options["max_retries"] = max_retries
    if endpoint.base_url:
        options.update(base_url=endpoint.base_url, api_key=endpoint.api_key or "EMPTY")
    elif endpoint.api_key:
        options["api_key"] = endpoint.api_key
    return options


@lru_cache(maxsize=8)
def _cached_client(endpoint: settings.ModelEndpoint, timeout_seconds: float | None, max_retries: int | None) -> Any:
    from openai import OpenAI

    return OpenAI(**_client_options(endpoint, timeout_seconds, max_retries))


_async_clients: dict[tuple[Any, ...], Any] = {}
_async_clients_lock = threading.Lock()


def async_openai_client(endpoint: settings.ModelEndpoint, timeout_seconds: float | None = None, max_retries: int | None = None) -> Any:
    """LLM 호출용 AsyncOpenAI 클라이언트다. 같은 접속 정보면 프로세스에서 하나를 공유한다(동시에 처음 불려도 하나).

    비동기 HTTP 연결은 처음 쓴 이벤트 루프에 묶인다. 모든 LLM 요청은 서비스 작업 루프(``services/work_loop``) 하나에서
    돌므로 스레드 사이에 공유해도 된다.
    """
    key = (endpoint, timeout_seconds, max_retries)
    with _async_clients_lock:
        client = _async_clients.get(key)
        if client is None:
            from openai import AsyncOpenAI

            client = _async_clients[key] = AsyncOpenAI(**_client_options(endpoint, timeout_seconds, max_retries))
        return client


def speech_to_text_provider() -> AudioPreprocessingSpeechToText:
    """STT 어댑터를 전처리(WAV 변환·VAD·환각 필터)로 감싸 만든다."""
    config = settings.get()
    endpoint = config.stt_endpoint
    if config.require_local_stt and not endpoint.base_url:
        # 비식별화는 전사 이후이므로, 원본 음성이 외부 SaaS로 나가는 구성을 막는다.
        raise LocalSttRequired("N_NOTE_REQUIRE_LOCAL_STT requires N_NOTE_STT_BASE_URL")
    client = openai_client(endpoint, config.stt_timeout_seconds, config.model_max_retries)
    adapter = OpenAiSpeechToTextAdapter(client, model=endpoint.model, prompt=config.stt_prompt, provider=endpoint.provider)
    vad = VadConfig(
        config.stt_vad_min_speech_seconds, config.stt_vad_min_dbfs, config.stt_vad_margin_db, config.stt_vad_padding_seconds,
    ) if config.stt_vad_enabled else None
    hallucination_filter = HallucinationFilter((*DEFAULT_HALLUCINATION_PATTERNS, *config.stt_hallucination_patterns)) if config.stt_hallucination_filter else None
    return AudioPreprocessingSpeechToText(adapter, config.stt_normalizes_audio, vad, hallucination_filter, config.stt_max_audio_seconds)


def draft_generator(gateway: McpGatewayClient | None) -> DraftGenerationAdapter:
    """설정한 LLM 서버·API 형식(``N_NOTE_LLM_API``)으로 초안 생성 어댑터를 만든다.

    tool call을 지원하지 않는 모델(``N_NOTE_LLM_TOOL_CALLING=false``)은 Gateway 사전 도구 없이
    전사문 용어 검토 힌트만으로 초안을 만든다.
    """
    config = settings.get()
    tool_gateway = gateway if config.llm_tool_calling_enabled else None
    tools = ToolExecutor(tool_gateway, config.llm_tool_names, config.llm_max_tool_calls, config.llm_max_tool_rounds)
    return DraftGenerationAdapter(structured_completion(), config.llm_endpoint.model, tools)


def structured_completion() -> PydanticAiCompletion:
    """설정한 LLM 서버·API 형식·옵션의 구조화 출력 요청기다(초안 생성과 동기 서식 추출이 함께 쓴다)."""
    config = settings.get()
    endpoint = config.llm_endpoint
    return PydanticAiCompletion(
        config.llm_api,
        lambda: async_openai_client(endpoint, config.llm_timeout_seconds, config.model_max_retries),
        endpoint.model,
        GenerationOptions(config.llm_max_tokens, config.llm_temperature, dict(config.llm_extra_body)),
    )
