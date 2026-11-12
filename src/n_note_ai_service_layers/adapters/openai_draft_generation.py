"""OpenAI 구조화 출력으로 간호 기록 초안을 생성하는 어댑터이다."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema
from n_note_ai_service_layers.services.mcp_gateway_client import (
    McpGatewayClient,
    McpGatewayError,
    McpGatewayToolError,
)

BASE_INSTRUCTIONS = (
    "You write Korean nursing records. Fill the requested nursing form only from the supplied payload "
    "(transcript and available facts). Do not infer unavailable, stale, or conflicting facts, and return null "
    "for a field when the payload has no information for it. "
    "Rewrite spoken language, dialect, fillers, and misrecognized words into standard Korean (표준어) and write "
    "text fields in concise Korean nursing-record style (for example sentences ending in '~함', '~임', '~호소함'). "
    "Use standard medical terminology; keep widely used clinical abbreviations (e.g. V/S, NPO, I/O) as spoken. "
    "The payload's term_review_hints are dictionary candidates that the standard Korean and medical terminology "
    "services found in the transcript; apply a hint only when it fits the clinical context. "
    "You may call the provided dictionary tools to confirm a term before using it. "
    "Treat privacy tokens in the form [[TYPE_n]] as immutable and never expand, replace, or use them as clinical evidence. "
    "Do not create risk scores, risk levels, diagnoses, or interventions that the transcript does not state. "
    "Never invent dates or times: resolve relative expressions such as '오후 2시' or '어제' against the payload's "
    "reference_time, and leave date/time fields null when the transcript states no time. "
    "For each filled field only, add one evidence_refs item with its field_id and refs such as 'transcript' or "
    "'fact:<fact_id>'; do not add items for null fields. "
    "Set needs_confirmation to true when any statement is uncertain."
)


class OpenAiDraftGenerationAdapter:
    """OpenAI 구조화 출력을 허용된 초안 필드로 제한해 생성한다.

    입력: OpenAI 클라이언트, 모델명 및 프롬프트 버전이다.
    출력: 생성 호출 시 허용 필드와 근거 참조만 담은 ``DraftResult``이다.
    """
    def __init__(
        self,
        client: Any | None = None,
        model: str = "gpt-4o-mini",
        prompt_version: str = "draft-v1",
        gateway_client: McpGatewayClient | None = None,
        max_tool_calls: int = 6,
        tool_timeout_seconds: float = 30.0,
        max_tool_rounds: int = 2,
        max_gateway_retries: int = 1,
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self._client = client
        self._model = model
        self._prompt_version = prompt_version
        self._gateway_client = gateway_client
        self._max_tool_calls = max_tool_calls
        # 도구 결과를 LLM에 돌려주는 왕복 횟수 상한이다. LLM 호출은 최대 ``1 + max_tool_rounds``번이다. 0이면 도구를 주지 않는다.
        self._max_tool_rounds = max_tool_rounds
        self._tool_timeout_seconds = tool_timeout_seconds
        self._max_gateway_retries = max_gateway_retries
        self._allowed_tool_names = allowed_tool_names

    def generate(self, payload: dict, schema: FormSchema) -> DraftResult:
        """최소화된 LLM 입력으로 초안을 생성하고 스키마 허용 필드만 반환한다.

        입력: ``payload``는 전사문과 사용 가능한 사실이고, ``schema``는 허용 필드를 정한다.
        출력: 검증된 구조화 응답을 변환한 ``DraftResult``이며, 잘못된 응답은 ``ValueError``를 발생시킨다.
        """
        request = {
            "model": self._model,
            "instructions": self._instructions(schema),
            "input": json.dumps(payload, ensure_ascii=False),
            "text": {"format": self._output_format(schema)},
        }
        tools = self._model_tools()
        if tools:
            request["tools"] = tools
        response = self._client.responses.create(**request)
        if tools:
            response = self._run_tool_loop(response, request)
        return self._draft_result(getattr(response, "output_text", None), schema)

    @staticmethod
    def _instructions(schema: FormSchema) -> str:
        """공통 작성 지침에 양식 제목과 양식별 지침을 덧붙인다."""
        instructions = BASE_INSTRUCTIONS + f" The form is '{schema.display_title}'."
        if schema.instructions:
            instructions += " Form-specific instructions: " + schema.instructions
        return instructions

    def _draft_result(self, output_text: Any, schema: FormSchema) -> DraftResult:
        """구조화 응답 JSON을 검증하고 스키마의 LLM 허용 필드만 남긴다."""
        try:
            parsed = json.loads(output_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM response did not contain valid structured draft output") from exc
        if not isinstance(parsed, dict):
            raise ValueError("structured draft output must be an object")
        if "proposed_fields" not in parsed:
            # 구조화 출력을 강제하지 못하는 서버에서 모델이 임의 구조를 내면 빈 초안으로 오인하지 않는다.
            raise ValueError("structured draft output did not follow the draft schema")
        allowed = {field.field_id for field in schema.fields if field.llm_allowed}
        proposed = parsed["proposed_fields"]
        if not isinstance(proposed, dict):
            raise ValueError("proposed_fields must be an object")
        evidence = _evidence_by_field(parsed.get("evidence_refs", []))
        return DraftResult(
            proposed_fields={key: value for key, value in proposed.items() if key in allowed and value is not None},
            evidence_refs={key: value for key, value in evidence.items() if key in allowed},
            needs_confirmation=bool(parsed.get("needs_confirmation", False)),
            model_version=self._model,
            prompt_version=self._prompt_version,
        )

    def _model_tools(self) -> list[dict[str, Any]]:
        """Gateway catalog의 허용 도구만 반환하고, Gateway 장애 시 도구 없이 초안을 만든다."""
        if self._gateway_client is None or self._max_tool_rounds <= 0:
            return []
        try:
            self._gateway_client.discover()
        except McpGatewayError:
            return []
        return self._gateway_client.model_tools(self._allowed_tool_names)

    def _run_tool_loop(self, response: Any, request: dict[str, Any]) -> Any:
        """도구 결과를 돌려주며 초안을 받는다. 왕복 수·Gateway 호출 수·경과 시간에 상한을 둔다.

        상한에 닿으면 마지막 요청을 ``tool_choice="none"``으로 보내 모델이 도구를 더 부르지 못하고 초안을 내게 한다.
        이렇게 하지 않으면 상한 뒤에도 모델이 도구를 계속 요청해 LLM 왕복이 끝없이 늘어날 수 있다.
        """
        assert self._gateway_client is not None
        started = time.monotonic()
        calls = 0
        rounds = 0
        while True:
            function_calls = [item for item in getattr(response, "output", []) if _value(item, "type") == "function_call"]
            if not function_calls or request.get("tool_choice") == "none":
                return response
            rounds += 1
            results, calls = self._run_tool_calls([(_value(call, "name"), _value(call, "arguments")) for call in function_calls], started, calls)
            outputs = [
                {"type": "function_call_output", "call_id": str(_value(call, "call_id")), "output": json.dumps(result, ensure_ascii=False)}
                for call, result in zip(function_calls, results)
            ]
            next_request = dict(request)
            next_request["input"] = outputs
            response_id = getattr(response, "id", None)
            if response_id:
                next_request["previous_response_id"] = response_id
            if self._tool_budget_spent(rounds, calls, started):
                next_request["tool_choice"] = "none"
            request = next_request
            response = self._client.responses.create(**next_request)

    def _tool_budget_spent(self, rounds: int, calls: int, started: float) -> bool:
        """도구 왕복·호출 수 또는 시간 상한에 닿았는지다. 닿으면 다음 LLM 요청이 마지막이다."""
        return rounds >= self._max_tool_rounds or calls >= self._max_tool_calls or time.monotonic() - started >= self._tool_timeout_seconds

    def _run_tool_calls(self, calls: list[tuple[Any, Any]], started: float, used: int) -> tuple[list[dict[str, Any]], int]:
        """한 응답의 도구 호출을 상한(횟수·시간) 안에서 동시에 Gateway로 보낸다.

        입력 순서대로 결과를 반환하고, 상한을 넘은 호출에는 ``tool_limit_reached``로 답한다.
        두 번째 값은 지금까지 사용한 호출 수다.
        """
        results: list[dict[str, Any]] = [{"error": "tool_limit_reached"} for _ in calls]
        permitted: list[int] = []
        for index in range(len(calls)):
            if used >= self._max_tool_calls or time.monotonic() - started >= self._tool_timeout_seconds:
                break
            used += 1
            permitted.append(index)
        if len(permitted) > 1:
            with ThreadPoolExecutor(max_workers=min(len(permitted), 4), thread_name_prefix="llm-tool") as pool:
                outcomes = list(pool.map(lambda index: self._call_gateway(*calls[index]), permitted))
        else:
            outcomes = [self._call_gateway(*calls[index]) for index in permitted]
        for index, outcome in zip(permitted, outcomes):
            results[index] = outcome
        return results, used

    def _call_gateway(self, name: Any, raw_arguments: Any) -> dict[str, Any]:
        if not isinstance(name, str) or (self._allowed_tool_names is not None and name not in self._allowed_tool_names):
            return {"error": "invalid_tool_call"}
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            return {"error": "invalid_tool_arguments"}
        try:
            return model_tool_output(self._gateway_client.call_tool(name, arguments))
        except McpGatewayToolError as exc:
            if exc.code == "not_found":
                self._gateway_client.discover()
                return {"error": exc.code, "retryable": False}
            if exc.retryable:
                for _ in range(self._max_gateway_retries):
                    try:
                        return model_tool_output(self._gateway_client.call_tool(name, arguments))
                    except McpGatewayToolError as retry_error:
                        exc = retry_error
                        if not exc.retryable:
                            break
            return {"error": exc.code, "retryable": exc.retryable}
        except McpGatewayError:
            return {"error": "gateway_unavailable"}

    @staticmethod
    def _output_format(schema: FormSchema) -> dict:
        """스키마의 LLM 허용 필드에 맞는 OpenAI JSON 스키마를 만든다.

        입력: ``schema``는 필드 ID와 자료형, LLM 허용 여부를 제공한다.
        출력: Responses API ``text.format``에 전달할 JSON 스키마 사전이다.
        """
        allowed_fields = [field for field in schema.fields if field.llm_allowed]
        allowed = [field.field_id for field in allowed_fields]
        value_schemas = {
            field.field_id: OpenAiDraftGenerationAdapter._value_schema(field)
            for field in allowed_fields
        }
        # 채운 필드의 근거만 배열로 받는다. 필드별 객체(모든 필드 필수)는 null 필드마다 빈 배열을 출력하게 해
        # 출력 토큰이 필드 수에 비례해 늘어난다.
        evidence_item = {
            "type": "object",
            "properties": {
                "field_id": {"type": "string", "enum": allowed} if allowed else {"type": "string"},
                "refs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["field_id", "refs"],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "name": "nursing_draft",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "proposed_fields": {"type": "object", "properties": value_schemas, "required": allowed, "additionalProperties": False},
                    "evidence_refs": {"type": "array", "items": evidence_item},
                    "needs_confirmation": {"type": "boolean"},
                },
                "required": ["proposed_fields", "evidence_refs", "needs_confirmation"],
                "description": "Allowed draft field IDs: " + ", ".join(allowed),
            },
        }

    @staticmethod
    def _value_schema(field: FormField) -> dict:
        """필드 라벨·작성 지침·선택지를 담은 nullable JSON 스키마를 만든다."""
        if field.options:
            value = {"type": "string", "enum": list(field.options)}
        else:
            value = OpenAiDraftGenerationAdapter._json_type(field.type)
        schema: dict[str, Any] = {"anyOf": [value, {"type": "null"}]}
        description = " - ".join(item for item in (field.label, field.description) if item)
        if field.type == "date":
            description += " (YYYY-MM-DD)"
        elif field.type == "datetime":
            description += " (YYYY-MM-DD HH:MM)"
        if description.strip():
            schema["description"] = description.strip()
        return schema

    @staticmethod
    def _json_type(field_type: str) -> dict:
        """도메인 필드 자료형을 JSON 스키마 자료형으로 변환한다.

        입력: ``field_type``은 양식 필드의 자료형 문자열이다.
        출력: 대응하는 JSON 스키마 조각이며, 알 수 없는 형식은 문자열로 처리한다.
        """
        types = {"integer": "integer", "number": "number", "boolean": "boolean", "array": "array", "object": "object"}
        kind = types.get(field_type, "string")
        if kind == "array":
            return {"type": "array", "items": {"type": "string"}}
        if kind == "object":
            return {"type": "object", "properties": {}, "additionalProperties": False}
        return {"type": kind}


def _evidence_by_field(evidence: Any) -> dict[str, tuple[str, ...]]:
    """``[{"field_id", "refs"}]``(현재 schema) 또는 ``{field_id: [refs]}``(구조화 출력을 강제하지 못한 서버)를 필드별 근거로 바꾼다."""
    if isinstance(evidence, dict):
        pairs = [(key, value) for key, value in evidence.items()]
    elif isinstance(evidence, list):
        pairs = [(item.get("field_id"), item.get("refs")) for item in evidence if isinstance(item, dict)]
    else:
        raise ValueError("evidence_refs must be an array or object")
    merged: dict[str, tuple[str, ...]] = {}
    for field_id, refs in pairs:
        if isinstance(field_id, str) and isinstance(refs, list):
            merged[field_id] = merged.get(field_id, ()) + tuple(str(item) for item in refs)
    return merged


def _value(item: Any, key: str) -> Any:
    """Read an SDK object or a fixture dictionary without depending on SDK internals."""
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


# 사전 뜻풀이는 문맥 판단에 필요한 첫 부분만 준다. 동음이의어가 여럿이면 앞부분만으로 구분된다.
MAX_DEFINITION_CHARS = 120


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
