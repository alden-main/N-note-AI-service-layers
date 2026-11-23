"""OpenAI 호환 서버 흉내(가짜 HTTP)와 응답 본문 생성기다. 실제 AsyncOpenAI SDK가 보내는 JSON 본문을 확인한다."""

from __future__ import annotations

import json
from typing import Any

import httpx2
from openai import AsyncOpenAI

from n_note_ai_service_layers.adapters.draft_generation import DraftGenerationAdapter
from n_note_ai_service_layers.adapters.llm_completion import GenerationOptions, ToolExecutor
from n_note_ai_service_layers.adapters.pydantic_ai_completion import PydanticAiCompletion
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient


class FakeServer:
    """OpenAI 호환 서버 흉내다. 받은 요청 본문을 ``requests``에 남기고 ``replies``를 차례로 돌려준다."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.clients = 0

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if not self.replies:
            raise AssertionError("unexpected extra model request")
        # 함수는 요청 본문을 받아 응답을 만든다(계속 남겨 두고 매 요청에 쓴다).
        reply = self.replies[0](body) if callable(self.replies[0]) else self.replies.pop(0)
        if isinstance(reply, int):
            return httpx2.Response(reply, json={"error": {"message": "tool choice is not enabled", "type": "BadRequestError"}})
        return httpx2.Response(200, json=reply)

    def client(self) -> AsyncOpenAI:
        self.clients += 1
        transport = httpx2.MockTransport(self.handle)
        return AsyncOpenAI(base_url="http://local-llm/v1", api_key="EMPTY", max_retries=0, http_client=httpx2.AsyncClient(transport=transport))


def chat(content: str | None = None, *tool_calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
            for call_id, name, arguments in tool_calls
        ]
    return {
        "id": "chatcmpl", "object": "chat.completion", "created": 0, "model": "qwen",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def responses(text: str | None = None, *calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = [
        {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name, "arguments": json.dumps(arguments, ensure_ascii=False), "status": "completed"}
        for call_id, name, arguments in calls
    ]
    if text is not None:
        output.append({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]})
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-4o-mini", "status": "completed",
        "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }


def completion(server: FakeServer, api: str = "chat", model: str = "qwen", options: GenerationOptions | None = None) -> PydanticAiCompletion:
    return PydanticAiCompletion(api, server.client, model, options)


def adapter(
    server: FakeServer, api: str = "chat", gateway: Any = None, max_tool_rounds: int = 2, options: GenerationOptions | None = None,
    model: str = "qwen", allowed_tool_names: tuple[str, ...] | None = None, max_tool_calls: int = 6, tool_timeout_seconds: float = 30.0,
) -> DraftGenerationAdapter:
    """가짜 서버로 보내는 초안 생성 어댑터다. ``gateway``는 FakeGateway(transport) 또는 McpGatewayClient다."""
    tools = None
    if gateway is not None:
        client = gateway if isinstance(gateway, McpGatewayClient) else McpGatewayClient(transport=gateway)
        tools = ToolExecutor(client, allowed_tool_names, max_tool_calls, max_tool_rounds, tool_timeout_seconds)
    return DraftGenerationAdapter(completion(server, api, model, options), model, tools)


def function_outputs(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Responses 요청 ``input``의 도구 결과(``function_call_output``) 항목이다."""
    return [item for item in request.get("input", []) if isinstance(item, dict) and item.get("type") == "function_call_output"]
