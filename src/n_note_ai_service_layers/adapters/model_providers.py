"""설정(OpenAI 또는 OpenAI 호환 로컬 서버)에 맞는 STT·초안 생성 어댑터를 만든다."""

from __future__ import annotations

from typing import Any

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.openai_chat_draft_generation import OpenAiChatDraftGenerationAdapter
from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter
from n_note_ai_service_layers.adapters.openai_stt import OpenAiSpeechToTextAdapter
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient


def openai_client(endpoint: settings.ModelEndpoint) -> Any:
    """접속 정보로 OpenAI SDK 클라이언트를 만든다. 로컬 서버는 API 키가 없어도 된다."""
    from openai import OpenAI

    if endpoint.base_url:
        return OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key or "EMPTY")
    # base_url이 없으면 OpenAI API이며 키는 OPENAI_API_KEY에서 읽는다(없으면 예외).
    return OpenAI(api_key=endpoint.api_key) if endpoint.api_key else OpenAI()


def speech_to_text_provider() -> OpenAiSpeechToTextAdapter:
    endpoint = settings.stt_endpoint()
    return OpenAiSpeechToTextAdapter(openai_client(endpoint), model=endpoint.model, prompt=settings.stt_prompt(), provider=endpoint.provider)


def draft_generator(gateway: McpGatewayClient | None) -> OpenAiDraftGenerationAdapter:
    """``N_NOTE_LLM_API``에 따라 Responses 또는 Chat Completions 어댑터를 만든다.

    tool call을 지원하지 않는 모델(``N_NOTE_LLM_TOOL_CALLING=false``)은 Gateway 사전 도구 없이
    전사문 용어 검토 힌트만으로 초안을 만든다.
    """
    endpoint = settings.llm_endpoint()
    client = openai_client(endpoint)
    tool_gateway = gateway if settings.llm_tool_calling_enabled() else None
    if settings.llm_api() == "chat":
        return OpenAiChatDraftGenerationAdapter(
            client,
            endpoint.model,
            gateway_client=tool_gateway,
            allowed_tool_names=settings.llm_tool_names(),
            extra_body=settings.llm_extra_body(),
            max_tokens=settings.llm_max_tokens(),
            temperature=settings.llm_temperature(),
        )
    return OpenAiDraftGenerationAdapter(client, endpoint.model, gateway_client=tool_gateway, allowed_tool_names=settings.llm_tool_names())
