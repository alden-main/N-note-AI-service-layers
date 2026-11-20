"""OpenAI 호환 LLM에 구조화 출력(JSON 스키마) 요청 한 번을 보내는 공통 계층이다.

초안 생성과 동기 서식 추출(internal_stt)이 함께 쓴다. API 형식별 차이(Responses·Chat Completions)는
``ResponsesCompletion``·``ChatCompletion``에만 있고, 사전 도구 호출(Gateway)은 ``ToolExecutor``가 맡는다.

- ``GenerationOptions``: temperature·max_tokens·extra_body
- ``ToolExecutor``: 허용 도구 catalog, 호출 수·왕복 수·시간 상한, 동시 호출, Gateway 오류 응답, 결과 축약
- ``StructuredOutput``: 출력 JSON 스키마(이름·스키마)
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import BadRequestError

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.observability import current_span, set_attributes, set_content, span
from n_note_ai_service_layers.services import executors
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient, McpGatewayError, McpGatewayToolError

# 응답 하나에서 동시에 Gateway로 보내는 도구 호출 수다. 넘는 호출은 이어서 순차로 보낸다.
MAX_PARALLEL_TOOL_CALLS = 4
# 사전 뜻풀이는 문맥 판단에 필요한 첫 부분만 준다. 동음이의어가 여럿이면 앞부분만으로 구분된다.
MAX_DEFINITION_CHARS = 120
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class GenerationOptions:
    """LLM 요청 공통 옵션이다(``N_NOTE_LLM_MAX_TOKENS``·``N_NOTE_LLM_TEMPERATURE``·``N_NOTE_LLM_EXTRA_BODY``)."""

    max_tokens: int | None = None
    temperature: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def chat_kwargs(self) -> dict[str, Any]:
        return self._kwargs("max_tokens")

    def responses_kwargs(self) -> dict[str, Any]:
        return self._kwargs("max_output_tokens")

    def _kwargs(self, max_tokens_key: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.max_tokens is not None:
            kwargs[max_tokens_key] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        return kwargs


@dataclass(frozen=True)
class StructuredOutput:
    """LLM이 따라야 할 출력 JSON 스키마다(strict)."""

    name: str
    schema: dict[str, Any]


@dataclass
class ToolBudget:
    """초안 하나가 쓴 도구 왕복·호출 수와 시작 시각이다."""

    started: float = field(default_factory=time.monotonic)
    calls: int = 0
    rounds: int = 0


class ToolExecutor:
    """LLM이 요청한 사전 도구를 Gateway로 호출한다.

    왕복 수(``max_tool_rounds``)·호출 수(``max_tool_calls``)·경과 시간(``timeout_seconds``)에 상한을 둔다.
    허용 목록 밖의 이름·깨진 인자·Gateway 오류는 예외 대신 LLM에 돌려줄 오류 객체가 된다.
    """

    def __init__(
        self,
        gateway: McpGatewayClient | None,
        allowed_tool_names: tuple[str, ...] | None = None,
        max_tool_calls: int = 6,
        max_tool_rounds: int = 2,
        timeout_seconds: float = 30.0,
        max_gateway_retries: int = 1,
    ) -> None:
        self.gateway = gateway
        self.allowed_tool_names = allowed_tool_names
        self.max_tool_calls = max_tool_calls
        # LLM 호출은 최대 ``1 + max_tool_rounds``번이다. 0이면 도구를 주지 않는다.
        self.max_tool_rounds = max_tool_rounds
        self.timeout_seconds = timeout_seconds
        self.max_gateway_retries = max_gateway_retries

    def model_tools(self) -> list[dict[str, Any]]:
        """Gateway catalog의 허용 도구(Responses 함수 도구 형식)다. Gateway 장애·도구 비허용이면 빈 목록이다."""
        if self.gateway is None or self.max_tool_rounds <= 0:
            return []
        try:
            self.gateway.discover()
        except McpGatewayError:
            return []
        return self.gateway.model_tools(self.allowed_tool_names)

    def exhausted_by_time(self, budget: ToolBudget) -> bool:
        return time.monotonic() - budget.started >= self.timeout_seconds

    def exhausted(self, budget: ToolBudget) -> bool:
        """왕복·호출 수 또는 시간 상한에 닿았는지다. 닿으면 다음 LLM 요청이 마지막이다."""
        return (
            budget.rounds >= self.max_tool_rounds or budget.calls >= self.max_tool_calls
            or time.monotonic() - budget.started >= self.timeout_seconds
        )

    def run(self, calls: list[tuple[Any, Any]], budget: ToolBudget) -> list[dict[str, Any]]:
        """한 응답의 도구 호출을 상한 안에서 동시에 보내고 입력 순서대로 결과를 돌려준다.

        상한을 넘은 호출에는 ``tool_limit_reached``로 답한다.
        """
        results: list[dict[str, Any]] = [{"error": "tool_limit_reached"} for _ in calls]
        permitted: list[int] = []
        for index in range(len(calls)):
            if budget.calls >= self.max_tool_calls or time.monotonic() - budget.started >= self.timeout_seconds:
                break
            budget.calls += 1
            permitted.append(index)
        if len(permitted) > 1:
            pool = executors.shared("llm-tool", executors.worker_parallelism() * MAX_PARALLEL_TOOL_CALLS)
            outcomes = executors.map_ordered(pool, lambda index: self._call(*calls[index]), permitted[:MAX_PARALLEL_TOOL_CALLS])
            outcomes += [self._call(*calls[index]) for index in permitted[MAX_PARALLEL_TOOL_CALLS:]]
        else:
            outcomes = [self._call(*calls[index]) for index in permitted]
        for index, outcome in zip(permitted, outcomes):
            results[index] = outcome
        return results

    def _call(self, name: Any, raw_arguments: Any) -> dict[str, Any]:
        """도구 호출 하나다. span(``n_note.llm.tool``)에 도구 이름·결과 코드를 남긴다(인자·결과 본문은 본문 기록 시에만)."""
        with span("llm.tool", **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": str(name)}) as current:
            set_content(current, "gen_ai.tool.call.arguments", raw_arguments)
            result = self._call_gateway(name, raw_arguments)
            set_attributes(current, {"llm.tool_error": result.get("error")})
            set_content(current, "gen_ai.tool.call.result", result)
            return result

    def _call_gateway(self, name: Any, raw_arguments: Any) -> dict[str, Any]:
        assert self.gateway is not None
        if not isinstance(name, str) or (self.allowed_tool_names is not None and name not in self.allowed_tool_names):
            return {"error": "invalid_tool_call"}
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            return {"error": "invalid_tool_arguments"}
        try:
            return model_tool_output(self.gateway.call_tool(name, arguments))
        except McpGatewayToolError as exc:
            if exc.code == "not_found":
                # Gateway catalog가 바뀌었을 수 있다. TTL 캐시를 무효화해야 실제로 다시 조회한다.
                # 재조회 실패(Gateway 장애)는 이 도구 호출의 오류일 뿐이므로 초안 생성을 중단시키지 않는다.
                self.gateway.invalidate_catalog()
                try:
                    self.gateway.discover()
                except McpGatewayError:
                    pass
                return {"error": exc.code, "retryable": False}
            if exc.retryable:
                for _ in range(self.max_gateway_retries):
                    try:
                        return model_tool_output(self.gateway.call_tool(name, arguments))
                    except McpGatewayToolError as retry_error:
                        exc = retry_error
                        if not exc.retryable:
                            break
            return {"error": exc.code, "retryable": exc.retryable}
        except McpGatewayError:
            return {"error": "gateway_unavailable"}


class StructuredCompletion(Protocol):
    """구조화 출력 요청 한 번이다. 반환값은 모델이 낸 JSON 텍스트(정리 전)이다."""

    def complete(
        self, instructions: str, user_content: str, output: StructuredOutput,
        tools: ToolExecutor | None = None, tool_instruction: str = "",
        accept_unconstrained: Callable[[Any], str | None] | None = None,
    ) -> Any: ...


class ResponsesCompletion:
    """OpenAI Responses API다. 전사문·초안을 OpenAI 쪽에 보관하지 않도록 ``store=False``로 보낸다."""

    def __init__(self, client: Any, model: str, options: GenerationOptions | None = None) -> None:
        self.client = client
        self.model = model
        self.options = options or GenerationOptions()

    def complete(
        self, instructions: str, user_content: str, output: StructuredOutput,
        tools: ToolExecutor | None = None, tool_instruction: str = "",
        accept_unconstrained: Callable[[Any], str | None] | None = None,
    ) -> Any:
        """도구를 주면 상한 안에서 도구 결과를 돌려주며 최종 출력을 받는다. ``accept_unconstrained``는 쓰지 않는다."""
        with completion_span("responses", self.model, output, instructions, user_content) as current:
            model_tools = tools.model_tools() if tools else []
            request: dict[str, Any] = {
                "model": self.model,
                "instructions": instructions + (tool_instruction if model_tools else ""),
                "input": [{"role": "user", "content": user_content}],
                "text": {"format": {"type": "json_schema", "name": output.name, "schema": output.schema, "strict": True}},
                "store": False,
                **self.options.responses_kwargs(),
            }
            if model_tools:
                request["tools"] = model_tools
            set_attributes(current, {"llm.tools_offered": len(model_tools)})
            response = self._create(request, 0)
            if model_tools and tools is not None:
                response = self._tool_loop(response, request, tools)
            text = getattr(response, "output_text", None)
            set_content(current, "llm.output", text)
            return text

    def _create(self, request: dict[str, Any], round_index: int) -> Any:
        """Responses 요청 한 번이다. span(``n_note.llm.request``)에 왕복 번호·토큰 수·요청한 도구 수를 남긴다."""
        with request_span(self.model, round_index, request.get("tool_choice")) as current:
            set_content(current, "llm.request_input", request["input"])
            response = self.client.responses.create(**request)
            usage = getattr(response, "usage", None)
            calls = [item for item in (getattr(response, "output", None) or []) if _value(item, "type") == "function_call"]
            set_attributes(current, {
                "gen_ai.usage.input_tokens": getattr(usage, "input_tokens", None),
                "gen_ai.usage.output_tokens": getattr(usage, "output_tokens", None),
                "llm.tool_calls_requested": len(calls),
            })
            set_content(current, "llm.response_output", [_plain(item) for item in calls] or getattr(response, "output_text", None))
            return response

    def _tool_loop(self, response: Any, request: dict[str, Any], tools: ToolExecutor) -> Any:
        """상한에 닿으면 마지막 요청을 ``tool_choice="none"``으로 보내 모델이 도구 없이 출력을 내게 한다.

        이렇게 하지 않으면 상한 뒤에도 모델이 도구를 계속 요청해 LLM 왕복이 끝없이 늘어날 수 있다.
        """
        budget = ToolBudget()
        while True:
            function_calls = [item for item in getattr(response, "output", []) if _value(item, "type") == "function_call"]
            if not function_calls or request.get("tool_choice") == "none":
                return response
            budget.rounds += 1
            results = tools.run([(_value(call, "name"), _value(call, "arguments")) for call in function_calls], budget)
            # store=False라 서버가 앞 응답을 기억하지 않는다. 모델이 낸 function_call과 그 결과를 이력에 이어 붙인다.
            echoed = [
                {"type": "function_call", "call_id": str(_value(call, "call_id")), "name": _value(call, "name"), "arguments": _value(call, "arguments")}
                for call in function_calls
            ]
            outputs = [
                {"type": "function_call_output", "call_id": str(_value(call, "call_id")), "output": json.dumps(result, ensure_ascii=False)}
                for call, result in zip(function_calls, results)
            ]
            request = {**request, "input": [*request["input"], *echoed, *outputs]}
            if tools.exhausted(budget):
                request["tool_choice"] = "none"
            response = self._create(request, budget.rounds)
            record_budget(budget)


class ChatCompletion:
    """OpenAI 호환 Chat Completions(vLLM 등 로컬 서버)다.

    고정 내용(지침·출력 스키마)은 system 메시지에, 요청별 내용은 그 뒤 user 메시지에 둔다. 같은 지침의 요청이
    같은 접두부를 공유하므로 vLLM prefix caching·OpenAI prompt caching이 적중한다.

    도구 호출 단계와 구조화 출력 단계를 나눈다. vLLM 등은 구조화 출력이 생성 전체를 JSON 문법으로 제한하므로
    같은 요청에서 tool call을 낼 수 없기 때문이다. 도구 단계에서 모델이 도구 없이 바로 낸 답을
    ``accept_unconstrained``가 받아들이면 그대로 쓰고 구조화 출력 호출을 생략한다.
    서버가 도구 호출을 지원하지 않으면(400 응답) 도구 없이 진행한다.
    """

    def __init__(self, client: Any, model: str, options: GenerationOptions | None = None) -> None:
        self.client = client
        self.model = model
        self.options = options or GenerationOptions()

    def complete(
        self, instructions: str, user_content: str, output: StructuredOutput,
        tools: ToolExecutor | None = None, tool_instruction: str = "",
        accept_unconstrained: Callable[[Any], str | None] | None = None,
    ) -> Any:
        with completion_span("chat", self.model, output, instructions, user_content) as current:
            chat_tools = [_chat_tool(tool) for tool in (tools.model_tools() if tools else [])]
            system = (
                instructions + (tool_instruction if chat_tools else "")
                + "\n\n" + chat_output_instruction(output)
            )
            messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
            set_attributes(current, {"llm.tools_offered": len(chat_tools)})
            if chat_tools and tools is not None:
                messages, answer = self._tool_phase(messages, chat_tools, tools, accept_unconstrained)
                if answer is not None:
                    set_attributes(current, {"llm.unconstrained_answer_accepted": True})
                    set_content(current, "llm.output", answer)
                    return answer
            final = _with_final_turn(messages)
            response = self._create(
                {**self._request(final),
                 "response_format": {"type": "json_schema", "json_schema": {"name": output.name, "schema": output.schema, "strict": True}}},
                final, "structured_output",
            )
            text = json_text(first_message(response).get("content"))
            set_content(current, "llm.output", text)
            return text

    def _create(self, request: dict[str, Any], messages: list[dict[str, Any]], phase: str, round_index: int = 0) -> Any:
        """Chat 요청 한 번이다. span(``n_note.llm.request``)에 단계(도구·구조화 출력)·토큰 수를 남긴다."""
        with request_span(self.model, round_index, request.get("tool_choice"), phase=phase) as current:
            # system 메시지(지침)는 prompts.toml·completion span에 있으므로 요청 span에는 그 뒤 대화만 남긴다.
            set_content(current, "llm.request_input", messages[1:])
            response = self.client.chat.completions.create(**request)
            usage = _value(response, "usage")
            message = first_message(response)
            set_attributes(current, {
                "gen_ai.usage.input_tokens": _value(usage, "prompt_tokens") if usage is not None else None,
                "gen_ai.usage.output_tokens": _value(usage, "completion_tokens") if usage is not None else None,
                "llm.tool_calls_requested": len(message.get("tool_calls") or []),
            })
            set_content(current, "llm.response_output", {"content": message.get("content"), "tool_calls": [_plain(call) for call in message.get("tool_calls") or []]})
            return response

    def _request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"model": self.model, "messages": messages, **self.options.chat_kwargs()}

    def _tool_phase(
        self, messages: list[dict[str, Any]], chat_tools: list[dict[str, Any]], tools: ToolExecutor,
        accept_unconstrained: Callable[[Any], str | None] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """도구 호출 결과를 대화에 쌓는다. 상한에 닿으면 구조화 출력 단계로 넘어간다."""
        budget = ToolBudget()
        history = list(messages)
        while not tools.exhausted(budget):
            budget.rounds += 1
            record_budget(budget)
            try:
                response = self._create({**self._request(history), "tools": chat_tools, "tool_choice": "auto"}, history, "tools", budget.rounds)
            except BadRequestError:
                # 서버가 --enable-auto-tool-choice 없이 떠 있으면 400을 반환한다. 도구 없이 진행한다.
                set_attributes(current_span(), {"llm.tool_calling_rejected": True})
                return history, None
            message = first_message(response)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return history, accept_unconstrained(message.get("content")) if accept_unconstrained else None
            history.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {"id": str(_value(call, "id")), "type": "function", "function": {"name": _value(_value(call, "function"), "name"), "arguments": _value(_value(call, "function"), "arguments")}}
                    for call in tool_calls
                ],
            })
            functions = [_value(call, "function") for call in tool_calls]
            results = tools.run([(_value(function, "name"), _value(function, "arguments")) for function in functions], budget)
            for call, result in zip(tool_calls, results):
                history.append({"role": "tool", "tool_call_id": str(_value(call, "id")), "content": json.dumps(result, ensure_ascii=False)})
            record_budget(budget)
        return history, None


@contextmanager
def completion_span(api: str, model: str, output: StructuredOutput, instructions: str, user_content: str, engine: str = "native") -> Iterator[Any]:
    """구조화 출력 요청 하나(도구 왕복 포함) 전체의 span(``n_note.llm.completion``)이다.

    지침 판(prompt version)은 지침 문장 대신 늘 남긴다. 지침·입력 본문은 본문 기록 시에만 남긴다.
    """
    catalog = prompts.get()
    with span("llm.completion", **{
        "gen_ai.operation.name": "chat", "gen_ai.request.model": model, "gen_ai.output.type": "json",
        "llm.engine": engine, "llm.api": api, "llm.output_name": output.name,
        "prompt.draft_version": catalog.draft.version, "prompt.form_extraction_version": catalog.form_extraction.version,
    }) as current:
        set_content(current, "llm.instructions", instructions)
        set_content(current, "llm.input", user_content)
        yield current


@contextmanager
def request_span(model: str, round_index: int, tool_choice: Any = None, **fields: Any) -> Iterator[Any]:
    """모델 요청 한 번의 span(``n_note.llm.request``)이다. ``round_index``는 도구 왕복 번호(0이면 첫 요청)다."""
    with span("llm.request", **{"gen_ai.request.model": model, "llm.round": round_index, "llm.tool_choice": tool_choice, **fields}) as current:
        yield current


def record_budget(budget: ToolBudget, target: Any = None) -> None:
    """completion span(기본: 현재 span)에 지금까지 쓴 도구 왕복·호출 수를 남긴다(마지막 값이 남는다)."""
    set_attributes(target or current_span(), {"llm.tool_rounds": budget.rounds, "llm.tool_calls": budget.calls})


def _plain(item: Any) -> Any:
    """SDK 객체를 span 본문용 사전으로 바꾼다(fixture 사전은 그대로)."""
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    return dump(mode="json") if callable(dump) else {"type": _value(item, "type"), "name": _value(item, "name"), "arguments": _value(item, "arguments")}


def create_completion(api: str, client: Any, model: str, options: GenerationOptions | None = None) -> ResponsesCompletion | ChatCompletion:
    """``N_NOTE_LLM_API``(``responses`` 또는 ``chat``)에 맞는 구조화 출력 요청기다."""
    return ChatCompletion(client, model, options) if api == "chat" else ResponsesCompletion(client, model, options)


def first_message(response: Any) -> dict[str, Any]:
    """Chat 응답(SDK 객체나 fixture 사전)에서 첫 번째 메시지를 읽는다."""
    choices = _value(response, "choices") or []
    if not choices:
        raise ValueError("LLM response did not contain any choice")
    message = _value(choices[0], "message")
    return {"content": _value(message, "content"), "tool_calls": _value(message, "tool_calls")}


def json_text(content: Any) -> Any:
    """구조화 출력을 강제하지 못하는 서버를 위해 thinking 블록과 코드 펜스를 걷어낸다."""
    if not isinstance(content, str):
        return content
    text = _THINK_BLOCK.sub("", content).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def chat_output_instruction(output: StructuredOutput) -> str:
    """Chat system 메시지 끝에 붙이는 출력 지침과 JSON 스키마다.

    구조화 출력을 강제하지 못하는 서버(MLX, 일부 llama.cpp 등)도 같은 구조를 따르도록 스키마를 프롬프트에도 넣는다.
    """
    return prompts.get().structured_output.final_output + "\nJSON schema:\n" + json.dumps(output.schema, ensure_ascii=False)


def _with_final_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """도구 대화 뒤에는 최종 출력을 요청하는 user 메시지를 붙인다.

    Gemma 계열 chat template은 user 메시지 연속을 거부하므로 마지막이 user면 그대로 둔다.
    """
    if messages[-1]["role"] == "user":
        return messages
    return messages + [{"role": "user", "content": prompts.get().structured_output.final_turn}]


def _chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Responses API 함수 도구 형식을 Chat Completions 형식으로 바꾼다."""
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool["parameters"]}}


def _value(item: Any, key: str) -> Any:
    """SDK 객체나 fixture 사전에서 값을 읽는다(SDK 내부 구조에 의존하지 않는다)."""
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def model_tool_output(result: dict[str, Any]) -> dict[str, Any]:
    """Gateway 도구 결과를 LLM에 돌려줄 최소 형태로 줄인다.

    MCP 결과는 같은 내용을 ``content``(JSON 문자열)와 ``structuredContent``에 두 번 담고, 사전 결과에는
    원본 파일·행·해시 같은 추적 정보와 같은 용어의 코드별 중복 항목이 들어 있다. LLM은 표기 판단에 쓰는
    표제어·품사·뜻풀이 앞부분, 한글·영문 용어명과 일치 유형만 받는다(입력 토큰과 응답 지연을 줄인다).
    모르는 구조의 결과는 ``structuredContent``만, 구조화 결과가 없으면 원래 결과를 그대로 준다.
    """
    payload = result.get("structuredContent")
    # FastMCP는 Union 반환 타입의 구조화 결과를 {"result": ...}로 감싼다.
    if isinstance(payload, dict) and set(payload) == {"result"} and isinstance(payload["result"], dict):
        payload = payload["result"]
    if not isinstance(payload, dict):
        return result
    if isinstance(payload.get("entries"), list):
        return {"entries": [_dictionary_entry(item) for item in payload["entries"]]}
    if isinstance(payload.get("results"), list):
        return {"results": _medical_terms(payload["results"])}
    return payload


def _dictionary_entry(item: Any) -> Any:
    """표준국어 사전 항목에서 표제어·품사·뜻풀이 앞부분만 남긴다."""
    if not isinstance(item, dict) or not isinstance(item.get("evidence"), dict):
        return item
    evidence = item["evidence"]
    definition = " ".join(str(evidence.get("definition") or "").split())
    return {
        "surface": item.get("surface"),
        "part_of_speech": " ".join(str(evidence.get("part_of_speech") or "").split()).strip("「」"),
        "definition": definition[:MAX_DEFINITION_CHARS],
    }


def _medical_terms(results: list[Any]) -> list[Any]:
    """의학용어 검색 결과에서 한글·영문 용어명·일치 유형·점수만 남기고, 코드만 다른 같은 용어는 하나로 합친다."""
    compact: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for item in results:
        term = item.get("term") if isinstance(item, dict) else None
        if not isinstance(term, dict):
            compact.append(item)
            continue
        entry = {"kor_name": term.get("kor_name"), "eng_name": term.get("eng_name"), "match_type": item.get("match_type"), "score": item.get("score")}
        key = (entry["kor_name"], entry["eng_name"], entry["match_type"])
        if key not in seen:
            seen.add(key)
            compact.append(entry)
    return compact
