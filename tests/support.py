"""여러 테스트가 공유하는 중립 양식 스키마와 가짜 Gateway다(특정 업무 양식에 묶이지 않는다)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus

SCORE_FIELD = "assessment.pain_score"
DERIVED_FIELD = "summary.derived_level"


def sample_form_schema(status: SchemaStatus = SchemaStatus.DRAFT) -> FormSchema:
    """LLM이 채우는 정수 필드 2개와 LLM에 노출하지 않는 파생 필드 1개를 가진 양식이다."""
    fields = (
        FormField(SCORE_FIELD, "integer", required=True, fact_id=SCORE_FIELD),
        FormField("assessment.mobility_score", "integer", fact_id="assessment.mobility_score"),
        FormField(DERIVED_FIELD, "string", llm_allowed=False),
    )
    return FormSchema("sample-note", "sample-note/v1", "SAMPLE_NOTE", (FormSection("assessment", fields),), status=status)


LOOKUP_TOOL = {
    "name": "dictionary__lookup", "description": "lookup",
    "inputSchema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"], "additionalProperties": False},
}
BATCH_CALL_SUFFIX = "/v1/tools/batch-call"


class FakeGateway:
    """``McpGatewayClient(transport=...)``에 넣는 가짜 Gateway다(네트워크 없이 MCP 응답을 흉내 낸다).

    - ``/mcp``: ``initialize``·``notifications/initialized``(202)·``tools/list``·``tools/call``
    - ``/v1/tools/batch-call``: 각 항목을 ``tools/call``과 같은 방식으로 처리한다.

    ``handler(name, arguments)``가 도구 결과를 만든다. 없으면 ``tool_result``를 돌려준다.
    기록: ``requests``(``/mcp`` 메시지와 헤더, 알림 포함), ``called``(도구 호출의 ``{"name", "arguments"}``),
    ``batches``(일괄 호출 본문). ``batch_error``를 주면 일괄 호출은 그 예외를 던진다(연결 실패 흉내).
    """

    def __init__(
        self,
        tools: Sequence[dict[str, Any]] = (LOOKUP_TOOL,),
        handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        tool_result: dict[str, Any] | None = None,
        batch_error: Exception | None = None,
    ) -> None:
        self.tools = list(tools)
        self.handler = handler
        self.tool_result = tool_result or {"content": [{"type": "text", "text": "found"}]}
        self.batch_error = batch_error
        self.requests: list[tuple[dict[str, Any], dict[str, str]]] = []
        self.called: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []

    def tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.called.append({"name": name, "arguments": arguments})
        return self.handler(name, arguments) if self.handler else self.tool_result

    @property
    def methods(self) -> list[str]:
        return [message["method"] for message, _headers in self.requests]

    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], bytes]:
        message = json.loads(body)
        echo = {"X-Request-ID": headers["X-Request-ID"]}
        if url.endswith(BATCH_CALL_SUFFIX):
            if self.batch_error is not None:
                raise self.batch_error
            self.batches.append(message)
            results = [self.tool(item["name"], item["arguments"]) for item in message["calls"]]
            return 200, echo, json.dumps({"results": results}, ensure_ascii=False).encode()
        self.requests.append((message, dict(headers)))
        if "id" not in message:  # notifications/initialized
            return 202, {}, b""
        if message["method"] == "initialize":
            result: dict[str, Any] = {"protocolVersion": "2025-06-18"}
        elif message["method"] == "tools/list":
            result = {"tools": self.tools}
        else:
            result = self.tool(message["params"]["name"], message["params"]["arguments"])
        return 200, echo, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}, ensure_ascii=False).encode()
