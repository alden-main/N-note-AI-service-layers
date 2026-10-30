"""설정(OpenAI 또는 OpenAI 호환 로컬 서버)에 맞는 STT·초안 생성 어댑터를 만든다."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.openai_chat_draft_generation import OpenAiChatDraftGenerationAdapter
from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter
from n_note_ai_service_layers.adapters.openai_stt import OpenAiSpeechToTextAdapter
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient


class LocalSttRequired(RuntimeError):
    """``N_NOTE_REQUIRE_LOCAL_STT``인데 STT 서버가 외부 OpenAI API로 설정되어 있다."""


def openai_client(endpoint: settings.ModelEndpoint, timeout_seconds: float | None = None, max_retries: int | None = None) -> Any:
    """접속 정보로 OpenAI SDK 클라이언트를 만든다. 로컬 서버는 API 키가 없어도 된다.

    같은 접속 정보의 클라이언트는 재사용해 HTTP 커넥션 풀을 요청 간에 공유한다.
    """
    return _cached_client(endpoint, timeout_seconds, max_retries)


@lru_cache(maxsize=8)
def _cached_client(endpoint: settings.ModelEndpoint, timeout_seconds: float | None, max_retries: int | None) -> Any:
    from openai import OpenAI

    options: dict[str, Any] = {}
    if timeout_seconds is not None:
        options["timeout"] = timeout_seconds
    if max_retries is not None:
        options["max_retries"] = max_retries
    if endpoint.base_url:
        return OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key or "EMPTY", **options)
    # base_url이 없으면 OpenAI API이며 키는 OPENAI_API_KEY에서 읽는다(없으면 예외).
    return OpenAI(api_key=endpoint.api_key, **options) if endpoint.api_key else OpenAI(**options)


def speech_to_text_provider() -> OpenAiSpeechToTextAdapter:
    config = settings.get()
    endpoint = config.stt_endpoint
    if config.require_local_stt and not endpoint.base_url:
        # 비식별화는 전사 이후이므로, 원본 음성이 외부 SaaS로 나가는 구성을 막는다.
        raise LocalSttRequired("N_NOTE_REQUIRE_LOCAL_STT requires N_NOTE_STT_BASE_URL")
    client = openai_client(endpoint, config.stt_timeout_seconds, config.model_max_retries)
    return OpenAiSpeechToTextAdapter(client, model=endpoint.model, prompt=config.stt_prompt, provider=endpoint.provider)


def draft_generator(gateway: McpGatewayClient | None) -> OpenAiDraftGenerationAdapter:
    """``N_NOTE_LLM_API``에 따라 Responses 또는 Chat Completions 어댑터를 만든다.

    tool call을 지원하지 않는 모델(``N_NOTE_LLM_TOOL_CALLING=false``)은 Gateway 사전 도구 없이
    전사문 용어 검토 힌트만으로 초안을 만든다.
    """
    config = settings.get()
    endpoint = config.llm_endpoint
    client = openai_client(endpoint, config.llm_timeout_seconds, config.model_max_retries)
    tool_gateway = gateway if config.llm_tool_calling_enabled else None
    if config.llm_api == "chat":
        return OpenAiChatDraftGenerationAdapter(
            client,
            endpoint.model,
            gateway_client=tool_gateway,
            allowed_tool_names=config.llm_tool_names,
            extra_body=config.llm_extra_body,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
    return OpenAiDraftGenerationAdapter(client, endpoint.model, gateway_client=tool_gateway, allowed_tool_names=config.llm_tool_names)
