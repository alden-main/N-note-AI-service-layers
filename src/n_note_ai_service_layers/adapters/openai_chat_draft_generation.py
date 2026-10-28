"""OpenAI 호환 Chat Completions 서버(vLLM 등 로컬 모델)로 간호 기록 초안을 생성하는 어댑터이다."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter, _value
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormSchema

FINAL_OUTPUT_INSTRUCTION = (
    "Now return only the final nursing draft as one JSON object that matches the required schema. "
    "Write every text value in Korean."
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAiChatDraftGenerationAdapter(OpenAiDraftGenerationAdapter):
    """Chat Completions의 JSON 스키마 출력으로 초안을 만든다.

    도구 호출 단계와 구조화 출력 단계를 나눈다. vLLM 등은 구조화 출력이 생성 전체를 JSON 문법으로
    제한하므로 같은 요청에서 tool call을 낼 수 없기 때문이다. 서버가 도구 호출을 지원하지 않으면
    (400 응답) 도구 없이 초안을 만든다.

    입력: OpenAI 호환 클라이언트, 서버의 모델명, 선택 추가 요청 필드(``extra_body``)이다.
    출력: 생성 호출 시 허용 필드와 근거 참조만 담은 ``DraftResult``이다.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "local-llm",
        *args: Any,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(client, model, *args, **kwargs)
        self._extra_body = dict(extra_body or {})
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, payload: dict, schema: FormSchema) -> DraftResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._instructions(schema)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        tools = [_chat_tool(tool) for tool in self._model_tools()]
        if tools:
            messages = self._run_tool_phase(messages, tools)
        output_format = self._output_format(schema)
        response = self._client.chat.completions.create(
            **self._request(_with_final_instruction(messages, output_format["schema"])),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": output_format["name"], "schema": output_format["schema"], "strict": True},
            },
        )
        return self._draft_result(_json_text(_message(response).get("content")), schema)

    def _request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        request: dict[str, Any] = {"model": self._model, "messages": messages}
        if self._max_tokens is not None:
            request["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._extra_body:
            request["extra_body"] = self._extra_body
        return request

    def _run_tool_phase(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """사전 도구 호출 결과를 대화에 쌓는다. 호출 상한·시간 상한에 닿으면 최종 단계로 넘어간다."""
        started = time.monotonic()
        calls = 0
        history = list(messages)
        while calls < self._max_tool_calls and time.monotonic() - started < self._tool_timeout_seconds:
            try:
                response = self._client.chat.completions.create(**self._request(history), tools=tools, tool_choice="auto")
            except Exception as exc:
                # 서버가 --enable-auto-tool-choice 없이 떠 있으면 400을 반환한다. 도구 없이 진행한다.
                if getattr(exc, "status_code", None) == 400:
                    return history
                raise
            message = _message(response)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return history
            history.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {"id": str(_value(call, "id")), "type": "function", "function": {"name": _value(_value(call, "function"), "name"), "arguments": _value(_value(call, "function"), "arguments")}}
                    for call in tool_calls
                ],
            })
            for call in tool_calls:
                function = _value(call, "function")
                if calls >= self._max_tool_calls or time.monotonic() - started >= self._tool_timeout_seconds:
                    result: dict[str, Any] = {"error": "tool_limit_reached"}
                else:
                    calls += 1
                    result = self._call_gateway(_value(function, "name"), _value(function, "arguments"))
                history.append({"role": "tool", "tool_call_id": str(_value(call, "id")), "content": json.dumps(result, ensure_ascii=False)})
        return history


def _with_final_instruction(messages: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """최종 출력 지시와 JSON 스키마를 붙인다.

    스키마를 프롬프트에도 넣어 구조화 출력을 강제하지 못하는 서버(MLX, 일부 llama.cpp 등)에서도
    같은 구조를 따르게 한다. Gemma 계열 chat template은 user 메시지 연속을 거부하므로 합친다.
    """
    instruction = FINAL_OUTPUT_INSTRUCTION + "\nJSON schema:\n" + json.dumps(schema, ensure_ascii=False)
    last = messages[-1]
    if last["role"] == "user":
        return messages[:-1] + [{"role": "user", "content": last["content"] + "\n\n" + instruction}]
    return messages + [{"role": "user", "content": instruction}]


def _chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Responses API 함수 도구 형식을 Chat Completions 형식으로 바꾼다."""
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool["parameters"]}}


def _message(response: Any) -> dict[str, Any]:
    """SDK 객체나 fixture 사전에서 첫 번째 응답 메시지를 읽는다."""
    choices = _value(response, "choices") or []
    if not choices:
        raise ValueError("LLM response did not contain any choice")
    message = _value(choices[0], "message")
    return {"content": _value(message, "content"), "tool_calls": _value(message, "tool_calls")}


def _json_text(content: Any) -> Any:
    """구조화 출력을 강제하지 못하는 서버를 위해 thinking 블록과 코드 펜스를 걷어낸다."""
    if not isinstance(content, str):
        return content
    text = _THINK_BLOCK.sub("", content).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text
