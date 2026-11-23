"""Pydantic AI Agent로 구조화 출력 요청을 보내는 LLM 호출 계층이다(``StructuredCompletion``).

초안 생성과 동기 서식 추출이 함께 쓴다. 모델 호출·도구 대화·출력 파싱을 Pydantic AI가 맡고, 그 과정이
OpenTelemetry span으로 남는다. 지침·출력 스키마·도구 상한·Gateway 도구 실행(``ToolExecutor``)은 ``llm_completion``에 있다.

API 형식별 흐름:

- Responses: 출력 스키마(``text.format``)와 도구를 함께 주고, 도구 상한에 닿으면 ``tool_choice="none"``으로 마지막 요청을 보낸다.
  ``store=False``로 보내 전사문·초안을 OpenAI 쪽에 보관하지 않는다.
- Chat(vLLM 등 로컬): 구조화 출력이 생성 전체를 JSON 문법으로 제한하면 tool call을 낼 수 없으므로 도구 단계(텍스트 출력)와
  구조화 출력 단계를 나눈다. 도구 단계에서 모델이 바로 낸 답이 스키마를 만족하면(``accept_unconstrained``) 그대로 쓴다.
  서버가 도구 호출을 지원하지 않으면(400) 도구 없이 진행한다.

동기 호출자(스레드 풀)는 프로세스 공유 이벤트 루프 하나(``run_sync``)에 요청을 넘기고, 모든 요청이 AsyncOpenAI
클라이언트 하나(연결 풀 하나)를 함께 쓴다.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

import pydantic_ai
from pydantic import TypeAdapter
from pydantic_ai import Agent, NativeOutput, StructuredDict, capture_run_messages
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import UsageLimits
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from n_note_ai_service_layers import observability, prompts
from n_note_ai_service_layers.adapters.llm_completion import (
    MAX_PARALLEL_TOOL_CALLS,
    GenerationOptions,
    StructuredOutput,
    ToolBudget,
    ToolExecutor,
    chat_output_instruction,
    completion_span,
    json_text,
    record_budget,
)
from n_note_ai_service_layers.observability import set_attributes
from n_note_ai_service_layers.services import executors

# Pydantic AI는 관측이 꺼져 있으면 stdout에 안내 배너를 찍는다. 서비스 로그(JSON 한 줄)를 깨뜨리므로 끈다.
pydantic_ai.BANNER_ENABLED = False

# 도구 인자는 Gateway가 도구 스키마로 검증한다. 여기서는 객체인지만 본다(깨진 인자는 ToolExecutor가 오류 객체로 답한다).
_ANY_ARGUMENTS = TypeAdapter(dict[str, Any]).validator


class GatewayToolset(AbstractToolset[None]):
    """Gateway catalog의 허용 도구를 Agent에 주는 toolset이다. 실행·상한은 ``ToolExecutor``와 ``ToolBudget``이 맡는다.

    상한(호출 수·경과 시간)을 넘은 호출에는 ``tool_limit_reached``로 답한다. 한 응답의 도구 호출은
    ``MAX_PARALLEL_TOOL_CALLS``개까지 동시에 Gateway로 보낸다.
    """

    def __init__(self, executor: ToolExecutor, model_tools: list[dict[str, Any]], budget: ToolBudget) -> None:
        self.executor = executor
        self.budget = budget
        self._definitions = {
            # MCP 입력 스키마의 선택 속성·additionalProperties는 OpenAI strict 모드와 맞지 않는다. strict를 켜면 Pydantic AI가
            # 선택 인자를 필수로 바꾸거나 서버가 거부하므로 끈다(인자는 Gateway가 도구 스키마로 검사한다).
            tool["name"]: ToolDefinition(
                name=tool["name"], description=tool.get("description") or "", parameters_json_schema=tool["parameters"], strict=False,
            )
            for tool in model_tools
        }
        self._parallel = asyncio.Semaphore(MAX_PARALLEL_TOOL_CALLS)

    @property
    def id(self) -> str:
        return "n-note-gateway"

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        # 도구 호출 하나가 실패해도(오류 객체 반환) 재시도가 아니므로 max_retries는 모델이 이름·인자를 잘못 낸 경우에만 쓰인다.
        return {
            name: ToolsetTool(toolset=self, tool_def=definition, max_retries=self.executor.max_tool_calls, args_validator=_ANY_ARGUMENTS)
            for name, definition in self._definitions.items()
        }

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]) -> Any:
        if self.budget.calls >= self.executor.max_tool_calls or self.executor.exhausted_by_time(self.budget):
            return {"error": "tool_limit_reached"}
        self.budget.calls += 1
        async with self._parallel:
            # 공유 풀은 제출한 쪽의 문맥을 이어받으므로 Gateway 호출 span이 Pydantic AI의 도구 span 아래에 놓인다.
            pool = executors.shared("llm-tool", executors.worker_parallelism() * MAX_PARALLEL_TOOL_CALLS)
            return await asyncio.get_running_loop().run_in_executor(pool, self.executor.call, name, tool_args)


class PydanticAiCompletion:
    """Pydantic AI Agent로 구조화 출력 요청 한 번(도구 왕복 포함)을 보낸다. ``StructuredCompletion``이다.

    ``client_factory``는 공유 이벤트 루프에서 쓸 AsyncOpenAI 클라이언트를 돌려준다(프로세스에 하나).
    """

    def __init__(
        self, api: str, client_factory: Callable[[], Any], model: str, options: GenerationOptions | None = None,
    ) -> None:
        self.api = api
        self.client_factory = client_factory
        self.model = model
        self.options = options or GenerationOptions()

    def complete(
        self, instructions: str, user_content: str, output: StructuredOutput,
        tools: ToolExecutor | None = None, tool_instruction: str = "",
        accept_unconstrained: Callable[[Any], str | None] | None = None,
    ) -> Any:
        """모델이 낸 JSON 텍스트를 돌려준다. 스키마 검증은 호출자가 한다(파싱 실패도 원문을 돌려준다)."""
        with completion_span(self.api, self.model, output, instructions, user_content) as current:
            model_tools = tools.model_tools() if tools else []
            set_attributes(current, {"llm.tools_offered": len(model_tools)})
            if self.api == "chat":
                coroutine = self._chat(current, instructions, user_content, output, tools, model_tools, tool_instruction, accept_unconstrained)
            else:
                coroutine = self._responses(current, instructions, user_content, output, tools, model_tools, tool_instruction)
            text = run_sync(coroutine)
            observability.set_content(current, "llm.output", text)
            return text

    async def _responses(
        self, current: Any, instructions: str, user_content: str, output: StructuredOutput, tools: ToolExecutor | None,
        model_tools: list[dict[str, Any]], tool_instruction: str,
    ) -> Any:
        budget = ToolBudget()
        toolsets = [GatewayToolset(tools, model_tools, budget)] if model_tools and tools is not None else []
        base = {**self.options.model_settings(), "openai_store": False}

        def step_settings(ctx: RunContext[None]) -> ModelSettings:
            # 요청마다 불린다. 도구 왕복 수는 이 실행의 도구 호출 응답 수다. 상한이면 도구 없이 출력만 내게 한다.
            budget.rounds = _tool_rounds(ctx.messages)
            record_budget(budget, current)
            if toolsets and tools is not None and tools.exhausted(budget):
                return {**base, "tool_choice": "none"}  # type: ignore[typeddict-item]
            return base  # type: ignore[return-value]

        agent = self._agent(f"{output.name}.structured_output", instructions + (tool_instruction if toolsets else ""), _native_output(output), toolsets)
        # 상한 뒤 마지막 요청은 tool_choice="none"이지만, 이를 무시하고 도구를 계속 요청하는 모델도 있다.
        # 요청 수를 1 + 왕복 수로 막고, 넘으면 마지막 응답의 텍스트를 쓴다(도구 왕복이 끝없이 늘지 않게).
        limit = UsageLimits(request_limit=1 + tools.max_tool_rounds) if toolsets and tools is not None else None
        return await _structured_run(agent, user_content, None, step_settings, limit)

    async def _chat(
        self, current: Any, instructions: str, user_content: str, output: StructuredOutput, tools: ToolExecutor | None,
        model_tools: list[dict[str, Any]], tool_instruction: str, accept_unconstrained: Callable[[Any], str | None] | None,
    ) -> Any:
        system = instructions + (tool_instruction if model_tools else "") + "\n\n" + chat_output_instruction(output)
        history: list[ModelMessage] = []
        if model_tools and tools is not None:
            history, answer = await self._chat_tool_phase(current, output, system, user_content, tools, model_tools, accept_unconstrained)
            if answer is not None:
                set_attributes(current, {"llm.unconstrained_answer_accepted": True})
                return answer
        agent = self._agent(f"{output.name}.structured_output", system, _native_output(output), [])
        if not history:
            return await _structured_run(agent, user_content, None, self.options.model_settings())
        # Gemma 계열 chat template은 user 메시지 연속을 거부하므로, 마지막이 user 요청이면 최종 출력 요청을 붙이지 않는다.
        final_turn = None if _ends_with_user_prompt(history) else prompts.get().structured_output.final_turn
        return await _structured_run(agent, final_turn, history, self.options.model_settings())

    async def _chat_tool_phase(
        self, current: Any, output: StructuredOutput, system: str, user_content: str, tools: ToolExecutor, model_tools: list[dict[str, Any]],
        accept_unconstrained: Callable[[Any], str | None] | None,
    ) -> tuple[list[ModelMessage], str | None]:
        """도구 대화(텍스트 출력)다. 상한에 닿으면 다음 모델 요청 전에 멈추고 그때까지의 대화를 돌려준다.

        모델이 도구 없이 답하면 ``accept_unconstrained``가 받아들인 답을 돌려준다. 받아들이지 않으면 그 답은 버리고
        구조화 출력 단계가 같은 대화에서 다시 요청한다.
        """
        budget = ToolBudget()
        agent = self._agent(f"{output.name}.tools", system, str, [GatewayToolset(tools, model_tools, budget)])
        settings = {**self.options.model_settings(), "tool_choice": "auto"}
        try:
            async with agent.iter(user_content, model_settings=settings) as run:  # type: ignore[arg-type]
                async for node in run:
                    if not Agent.is_model_request_node(node):
                        continue
                    budget.rounds = _tool_rounds(run.all_messages())
                    if tools.exhausted(budget):
                        record_budget(budget, current)
                        return _with_pending(run.all_messages(), node.request), None
                    # 이 모델 요청이 도구 단계의 다음 왕복이다(요청 전에 센다).
                    budget.rounds += 1
                    record_budget(budget, current)
                result = run.result
        except ModelHTTPError as exc:
            if exc.status_code != 400:
                raise
            # 서버가 --enable-auto-tool-choice 없이 떠 있으면 400을 반환한다. 도구 없이 진행한다.
            set_attributes(current, {"llm.tool_calling_rejected": True})
            return [], None
        assert result is not None
        messages = result.all_messages()
        answer = accept_unconstrained(result.output) if accept_unconstrained else None
        return (messages if answer is not None else messages[:-1]), answer

    def _agent(self, name: str, instructions: str, output_type: Any, toolsets: list[AbstractToolset[None]]) -> Agent[None, Any]:
        """``name``은 추적 화면의 Agent 실행 span 이름이다(예: ``nursing_draft.tools``·``nursing_draft.structured_output``)."""
        content = observability.capture_content()
        return Agent(
            self._model(),
            name=name,
            instructions=instructions,
            output_type=output_type,
            toolsets=toolsets,
            # 출력 파싱 실패는 재요청하지 않는다(호출자가 실패로 처리한다).
            retries={"output": 0},
            # Agent 실행·모델 요청·도구 실행 span을 만든다(전역 tracer provider). 본문은 본문 기록 시에만 넣는다.
            capabilities=[Instrumentation(settings=InstrumentationSettings(
                include_content=content, include_binary_content=False, include_model_request_parameters=content,
            ))],
        )

    def _model(self) -> Any:
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(openai_client=self.client_factory())
        if self.api == "chat":
            # max_tokens로 보낸다. llama.cpp·MLX 등 일부 OpenAI 호환 서버는 max_completion_tokens를 모른다.
            return OpenAIChatModel(self.model, provider=provider, profile={"openai_chat_supports_max_completion_tokens": False})  # type: ignore[typeddict-unknown-key]
        return OpenAIResponsesModel(self.model, provider=provider)


async def _structured_run(
    agent: Agent[None, Any], prompt: str | None, history: list[ModelMessage] | None, settings: Any, usage_limits: UsageLimits | None = None,
) -> Any:
    """구조화 출력 실행이다. 결과 사전을 JSON 텍스트로 돌려준다.

    모델이 스키마 밖의 답을 내 Pydantic AI가 파싱하지 못하면 마지막 응답 원문을 돌려준다. 호출자가
    ``ValueError``로 처리하게 하고, 같은 내용을 다시 요청해 지연을 늘리지 않는다.
    """
    with capture_run_messages() as messages:
        try:
            result = await agent.run(prompt, message_history=history, model_settings=settings, usage_limits=usage_limits)
        except (UnexpectedModelBehavior, UsageLimitExceeded):
            return _last_text(messages)
    return json.dumps(result.output, ensure_ascii=False)


def _native_output(output: StructuredOutput) -> NativeOutput[Any]:
    # 스키마는 프롬프트(Chat system 메시지)에 이미 있으므로 Pydantic AI가 다시 넣지 않게 template=False로 둔다.
    return NativeOutput(StructuredDict(output.schema, name=output.name), strict=True, template=False)


def _tool_rounds(messages: list[ModelMessage]) -> int:
    """모델이 도구를 요청한 응답 수(도구 왕복 수)다."""
    return sum(1 for message in messages if isinstance(message, ModelResponse) and any(isinstance(part, ToolCallPart) for part in message.parts))


def _with_pending(messages: list[ModelMessage], request: ModelRequest) -> list[ModelMessage]:
    """아직 보내지 않은 요청(도구 결과)을 대화 끝에 붙인다(이미 들어 있으면 그대로)."""
    return messages if messages and messages[-1] is request else [*messages, request]


def _ends_with_user_prompt(messages: list[ModelMessage]) -> bool:
    last = messages[-1]
    return isinstance(last, ModelRequest) and all(isinstance(part, UserPromptPart) for part in last.parts)


def _last_text(messages: list[ModelMessage]) -> Any:
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            text = "".join(part.content for part in message.parts if isinstance(part, TextPart))
            return json_text(text) if text else None
    return None


class _SharedLoop:
    """프로세스가 함께 쓰는 이벤트 루프 하나다(데몬 스레드에서 돈다).

    동기 호출자(작업 스레드 풀)는 ``run_sync``로 코루틴을 이 루프에 넘기고 결과를 기다린다. 모든 LLM 요청이 한 루프와
    한 AsyncOpenAI 연결 풀을 쓰므로 스레드마다 루프·클라이언트를 만들지 않는다. 호출한 스레드의 contextvars(현재
    추적 span)는 ``run_coroutine_threadsafe``가 넘기므로 span이 호출한 양식의 추적 아래에 놓인다.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def get(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="llm-event-loop", daemon=True).start()
                self._loop = loop
            return self._loop


_shared_loop = _SharedLoop()


def event_loop() -> asyncio.AbstractEventLoop:
    """LLM 요청이 도는 공유 이벤트 루프다(AsyncOpenAI 클라이언트는 이 루프에서만 쓴다)."""
    return _shared_loop.get()


def run_sync(coroutine: Any) -> Any:
    """코루틴을 공유 이벤트 루프에서 끝까지 실행하고 결과를 돌려준다. 공유 루프 스레드 안에서는 부르지 않는다."""
    loop = event_loop()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        coroutine.close()
        raise RuntimeError("run_sync must not be called from the shared LLM event loop")
    return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
