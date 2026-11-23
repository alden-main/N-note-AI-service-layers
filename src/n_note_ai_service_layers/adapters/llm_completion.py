"""구조화 출력(JSON 스키마) LLM 요청의 공통 계약과 사전 도구 실행기다.

모델 호출·도구 대화는 ``pydantic_ai_completion.PydanticAiCompletion``이 맡는다. 여기에는 요청 형식과 무관한 부분만 있다.

- ``GenerationOptions``: temperature·max_tokens·extra_body
- ``StructuredOutput``: 출력 JSON 스키마(이름·스키마)
- ``ToolExecutor``: 허용 도구 catalog, 호출 수·왕복 수·시간 상한, Gateway 오류 응답, 결과 축약
- ``completion_span``·``record_budget``: 추적 span
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.observability import current_span, set_attributes, set_content, span
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient, McpGatewayError, McpGatewayToolError

if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings

# 응답 하나에서 동시에 Gateway로 보내는 도구 호출 수다. 넘는 호출은 앞 호출이 끝나기를 기다린다.
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

    def model_settings(self) -> ModelSettings:
        """Pydantic AI ``ModelSettings``다. 값이 없는 옵션은 서버 기본값을 쓴다."""
        settings: ModelSettings = {}
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.extra_body:
            settings["extra_body"] = dict(self.extra_body)
        return settings


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
        """경과 시간 상한에 닿았는지다(상한 뒤의 도구 호출은 ``tool_limit_reached``로 답한다)."""
        return time.monotonic() - budget.started >= self.timeout_seconds

    def exhausted(self, budget: ToolBudget) -> bool:
        """왕복·호출 수 또는 시간 상한에 닿았는지다. 닿으면 다음 LLM 요청이 마지막이다."""
        return (
            budget.rounds >= self.max_tool_rounds or budget.calls >= self.max_tool_calls
            or time.monotonic() - budget.started >= self.timeout_seconds
        )

    def call(self, name: Any, raw_arguments: Any) -> dict[str, Any]:
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


@contextmanager
def completion_span(api: str, model: str, output: StructuredOutput, instructions: str, user_content: str) -> Iterator[Any]:
    """구조화 출력 요청 하나(도구 왕복 포함) 전체의 span(``n_note.llm.completion``)이다.

    지침 판(prompt version)은 지침 문장 대신 늘 남긴다. 지침·입력 본문은 본문 기록 시에만 남긴다.
    """
    catalog = prompts.get()
    with span("llm.completion", **{
        "gen_ai.operation.name": "chat", "gen_ai.request.model": model, "gen_ai.output.type": "json",
        "llm.api": api, "llm.output_name": output.name,
        "prompt.draft_version": catalog.draft.version, "prompt.form_extraction_version": catalog.form_extraction.version,
    }) as current:
        set_content(current, "llm.instructions", instructions)
        set_content(current, "llm.input", user_content)
        yield current


def record_budget(budget: ToolBudget, target: Any = None) -> None:
    """completion span(기본: 현재 span)에 지금까지 쓴 도구 왕복·호출 수를 남긴다(마지막 값이 남는다)."""
    set_attributes(target or current_span(), {"llm.tool_rounds": budget.rounds, "llm.tool_calls": budget.calls})


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


def model_tool_output(result: dict[str, Any]) -> dict[str, Any]:
    """Gateway 도구 결과를 LLM에 돌려줄 최소 형태로 줄인다.

    MCP 결과는 같은 내용을 ``content``(JSON 문자열)와 ``structuredContent``에 두 번 담고, 사전 결과에는
    원본 파일·행·해시 같은 추적 정보와 같은 용어의 코드별 중복 항목이 들어 있다. LLM은 표기 판단에 쓰는
    표제어·품사·뜻풀이 앞부분, 한글·영문 용어명과 일치 유형만 받는다(입력 토큰과 응답 지연을 줄인다).
    모르는 구조의 결과는 ``structuredContent``만, 구조화 결과가 없으면 원래 결과를 그대로 준다.
    """
    payload = result.get("structuredContent")
    # 예전 medic(FastMCP Union 반환)은 구조화 결과를 {"result": ...}로 감쌌다. 지금은 감싸지 않지만 둘 다 받는다.
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
