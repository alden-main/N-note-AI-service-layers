"""Gateway 내부에서 사용할 최소 Streamable HTTP MCP 진입점을 제공한다."""

from __future__ import annotations

import json
import logging
import os
from time import perf_counter
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from . import validation
from .contracts import InputError, ToolError
from .observability import AUDIT_LOGGER, SERVICE_ID, SERVICE_VERSION, SafeAuditLog

EVALUATE_TOOL = "evaluate_report_against_standard"
GUIDANCE_TOOL = "get_validation_guidance"
EVIDENCE_SEARCH_TOOL = "find_evidence_candidates"
MAX_REQUEST_BYTES = 1_048_576
MAX_TEXT_CHARS = 100_000


class McpApplication:
    """MCP JSON-RPC와 도구 계약을 연결하는 상태 없는 애플리케이션이다(HTTP 계층은 `create_asgi_app`)."""

    def __init__(self, audit: SafeAuditLog | None = None) -> None:
        """입력 한도 설정과 민감 원문 없는 감사 저장소를 구성한다. 한도 설정이 잘못되면 준비되지 않은 상태로 둔다."""
        self._audit = audit or SafeAuditLog()
        max_request_bytes = _positive_limit("MCP_MAX_REQUEST_BYTES", MAX_REQUEST_BYTES)
        max_text_chars = _positive_limit("MCP_MAX_TEXT_CHARS", MAX_TEXT_CHARS)
        self._configuration_ready = max_request_bytes is not None and max_text_chars is not None
        self._max_request_bytes = max_request_bytes or MAX_REQUEST_BYTES
        self._max_text_chars = max_text_chars or MAX_TEXT_CHARS

    def evaluate(self, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """전용 Gateway network에서 전달된 평가 요청을 처리한다."""
        return self._run_tool(
            EVALUATE_TOOL, arguments, request_id, "evaluation could not be completed",
            lambda: validation.evaluate(arguments.get("report"), arguments.get("validation_standard"), arguments.get("semantic_assessment")).as_dict(),
        )

    def guidance(self, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """LLM용 검토 정책을 반환하고 호출 metadata만 감사한다.

        Args:
            arguments: `validation_standard`만 포함하는 도구 인자다.
        """
        return self._run_tool(
            GUIDANCE_TOOL, arguments, request_id, "guidance could not be prepared",
            lambda: validation.guidance(arguments.get("validation_standard")),
        )

    def find_evidence_candidates(self, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """외부 AI의 검색어에 맞는 원문 후보 문장을 찾아 반환한다.

        이 도구는 문자 일치 검색만 수행한다. 반환된 후보가 기준을 충족하는지의
        의미 판단은 외부 AI와 평가 도구의 역할이다.
        """
        return self._run_tool(
            EVIDENCE_SEARCH_TOOL, arguments, request_id, "evidence candidates could not be found",
            lambda: validation.find_evidence_candidates(arguments.get("report"), arguments.get("validation_standard"), arguments.get("queries")),
        )

    def _run_tool(
        self, tool_name: str, arguments: dict[str, Any], request_id: str | None, failure_message: str,
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """입력 크기 검사·오류 변환·감사 기록을 모든 도구에 같은 방식으로 적용한다."""
        started_at = perf_counter()
        result_code = "internal_error"
        try:
            _enforce_text_limit(arguments, self._max_text_chars)
            payload = operation()
            result_code = "ok"
            return payload
        except RequestTooLarge:
            result_code = "request_too_large"
            return ToolError("request_too_large", "text input exceeds configured size limit").as_dict()
        except InputError:
            result_code = "invalid_input"
            return ToolError("invalid_input", "request does not satisfy the tool contract").as_dict()
        except (RuntimeError, TypeError):
            return ToolError("internal_error", failure_message, True).as_dict()
        finally:
            standard = arguments.get("validation_standard")
            self._audit.record(
                request_id=request_id,
                tool_name=tool_name,
                result_code=result_code,
                standard_version=standard.get("standard_version") if isinstance(standard, dict) else None,
                started_at=started_at,
            )

    def call_tool(self, name: Any, arguments: Any, request_id: str | None = None) -> dict[str, Any]:
        """도구 하나를 실행하고 MCP tool result(``structuredContent``·``isError``)를 돌려준다.

        Gateway가 해석하는 오류 형식(``structuredContent.error``의 code·message·retryable)을 유지한다.
        잘못된 이름·인자는 예외 대신 ``invalid_input`` tool result다. Gateway는 catalog 접두사(``report__``)를
        떼고 호출하므로 내부 도구명만 받는다.
        """
        handlers = {GUIDANCE_TOOL: self.guidance, EVIDENCE_SEARCH_TOOL: self.find_evidence_candidates, EVALUATE_TOOL: self.evaluate}
        handler = handlers.get(name) if isinstance(name, str) else None
        if handler is None or not isinstance(arguments, dict):
            payload = ToolError("invalid_input", "unknown tool or invalid arguments").as_dict()
        elif not self.is_ready():
            payload = ToolError("upstream_unavailable", "service is not ready", True).as_dict()
        else:
            payload = handler(arguments, request_id)
        return {"structuredContent": payload, "isError": "error" in payload}

    def tools(self) -> list[dict[str, Any]]:
        """실제 handler 계약과 같은 중첩 JSON Schema를 Gateway catalog에 공개한다."""
        size_limit = (
            f"입력 제한: 전체 JSON-RPC 요청 본문은 {self._max_request_bytes} bytes 이하이고, "
            f"모든 문자열은 {self._max_text_chars}자 이하여야 합니다. "
            "긴 보고서는 문단 또는 문장 경계에서 분할해 조각별로 호출하십시오. "
            "반환되는 citation offset은 각 조각의 text를 기준으로 합니다."
        )
        standard = _validation_standard_schema(size_limit)
        return [
            {
                "name": GUIDANCE_TOOL,
                "description": "검증 기준별 검토 정책을 반환합니다. validation_standard 전체를 전달해야 합니다. " + size_limit,
                "inputSchema": _tool_input_schema({"validation_standard": standard}, ["validation_standard"]),
            },
            {
                "name": EVIDENCE_SEARCH_TOOL,
                "description": "보고서에서 criterion/element별 검색어의 문자 일치 후보 문장과 offset을 반환합니다. " + size_limit,
                "inputSchema": _tool_input_schema(
                    {"report": _report_schema(self._max_text_chars), "validation_standard": standard, "queries": _evidence_queries_schema(self._max_text_chars)},
                    ["report", "validation_standard", "queries"],
                ),
            },
            {
                "name": EVALUATE_TOOL,
                "description": "보고서와 검증 기준, 선택적인 semantic_assessment 후보를 검증합니다. " + size_limit,
                "inputSchema": _tool_input_schema(
                    {"report": _report_schema(self._max_text_chars), "validation_standard": standard, "semantic_assessment": _semantic_assessment_schema(self._max_text_chars)},
                    ["report", "validation_standard"],
                ),
            },
        ]

    @property
    def max_request_bytes(self) -> int:
        """HTTP 요청 본문 상한(바이트)이다. SDK 전송 계층이 넘는 요청을 413으로 거부한다."""
        return self._max_request_bytes

    def health(self, ready: bool) -> dict[str, Any]:
        """민감 정보 없는 `/healthz` 본문이다."""
        return self._audit.health(ready)

    def is_ready(self) -> bool:
        """입력 한도 환경 설정이 유효하면 true다. 그 밖의 준비 조건은 생성 성공으로 충족된다."""
        return self._configuration_ready


def create_mcp_server(app: McpApplication) -> Server:
    """`McpApplication`의 도구를 MCP SDK 서버로 공개한다.

    도구 스키마는 손으로 정의한 계약(`McpApplication.tools`)을 그대로 쓰므로 FastMCP의 시그니처 기반 스키마 대신
    SDK 저수준 ``Server``의 ``list_tools``/``call_tool`` 핸들러를 둔다. 입력 검증도 도구가 직접 해
    잘못된 인자를 ``invalid_input`` tool result로 돌려준다(SDK 검증은 끈다).
    검증·근거 검색은 CPU 작업이라 스레드에서 실행해 이벤트 루프와 `/healthz`를 막지 않는다.
    """
    server: Server = Server(SERVICE_ID, version=SERVICE_VERSION)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool.model_validate(tool) for tool in app.tools()]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        request_id = _request_id(server)
        result = await anyio.to_thread.run_sync(lambda: app.call_tool(name, arguments, request_id))
        # content는 MCP 필수 필드다. 구조화 결과와 같은 JSON을 텍스트로도 싣는다(사양 권장).
        text = json.dumps(result["structuredContent"], ensure_ascii=False)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], **result)

    return server


def _request_id(server: Server) -> str | None:
    """Gateway가 전달한 ``X-Request-ID``다. 감사 이벤트에 원문 없이 이 값만 남긴다."""
    try:
        request = server.request_context.request
    except LookupError:
        return None
    return request.headers.get("x-request-id") if request is not None else None


def create_asgi_app(app: McpApplication | None = None) -> Starlette:
    """`/mcp`(SDK Streamable HTTP, 상태 없는 JSON 응답)와 `/healthz`를 제공하는 ASGI 앱이다.

    Gateway가 요청·응답 1회로 처리하도록 세션 없이 JSON으로 답한다(kodict·medicaldict와 같은 모드).
    `McpApplication`은 요청 간에 바뀌는 상태가 없다(감사 이벤트 추가만 스레드 안전한 deque에 한다).
    """
    mcp = app or McpApplication()
    session_manager = StreamableHTTPSessionManager(
        app=create_mcp_server(mcp), stateless=True, json_response=True, max_request_body_size=mcp.max_request_bytes,
    )

    async def healthz(_request: Request) -> Response:
        ready = mcp.is_ready()
        return JSONResponse(mcp.health(ready), status_code=200 if ready else 503)

    async def not_found(_request: Request) -> Response:
        return JSONResponse({"error": {"code": "not_found", "message": "not found", "retryable": False}}, status_code=404)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/mcp", _McpEndpoint(session_manager), methods=["GET", "POST", "DELETE"]),
            Route("/{path:path}", not_found, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
        ],
        lifespan=lifespan,
    )


class _McpEndpoint:
    """Starlette 경로에서 SDK 세션 관리자로 ASGI 요청을 넘긴다."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


def main() -> None:
    """Gateway 전용 내부 HTTP 서버(uvicorn)를 시작한다.

    운영 환경에서는 host port를 publish하지 않고 Gateway 전용 Docker network에서만
    이 앱을 구동해야 한다. MCP_HOST와 MCP_PORT는 컨테이너 내부 listener만
    설정하며 외부 공개 여부는 Docker Compose의 ports 설정이 결정한다.
    """
    import uvicorn

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    # 감사 이벤트(본문 없는 metadata)만 stderr로 내보낸다. SDK의 요청별 INFO 로그는 켜지 않는다.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    AUDIT_LOGGER.addHandler(handler)
    AUDIT_LOGGER.setLevel(logging.INFO)
    AUDIT_LOGGER.propagate = False
    # 접근 로그는 남기지 않는다(감사는 SafeAuditLog가 본문 없이 기록한다).
    uvicorn.run(create_asgi_app(), host=host, port=port, access_log=False)


class RequestTooLarge(ValueError):
    """도구 인자의 개별 문자열이 허용 한도를 넘었음을 나타낸다."""


def _positive_limit(environment_name: str, default: int) -> int | None:
    """환경 설정의 입력 한도다. 미설정이면 `default`를, 양의 정수가 아니면 None을 반환한다."""
    raw = os.getenv(environment_name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _enforce_text_limit(value: Any, maximum: int) -> None:
    """도구 입력의 모든 문자열이 설정한 문자 수를 넘지 않도록 검사한다."""
    if isinstance(value, str):
        if len(value) > maximum:
            raise RequestTooLarge()
    elif isinstance(value, dict):
        for item in value.values():
            _enforce_text_limit(item, maximum)
    elif isinstance(value, list):
        for item in value:
            _enforce_text_limit(item, maximum)


def _report_schema(maximum_text_chars: int) -> dict[str, Any]:
    """보고서 text 길이와 분할 시 citation 좌표 기준을 tool schema에 노출한다."""
    return {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "maxLength": maximum_text_chars,
                "description": (
                    "보고서 조각. 한 조각은 문단 또는 문장 경계에서 분할하고 "
                    "최대 길이를 넘기지 마십시오. citation offset은 이 조각의 text 기준입니다."
                ),
            },
        },
        "additionalProperties": True,
    }


def _tool_input_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """MCP arguments 객체의 공통 schema를 만든다.

    호출자가 보관하는 report_id 같은 부가 metadata는 handler가 무시하므로 거절하지
    않는다. 반면 tool 동작에 필요한 필수 필드는 catalog에서 명시한다.
    """
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": True}


def _validation_standard_schema(size_limit: str) -> dict[str, Any]:
    """지원하는 표준의 공통 진입 필드를 Gateway에 공개한다.

    표준은 schema_version별 interpreter가 최종 검증한다. 여기서는 Gateway가 원본
    표준 JSON을 축소하거나 필드를 추측해 제거하지 않도록 각 형식을 모두 허용한다.
    """
    return {
        "type": "object",
        "description": (
            "지원 schema_version: report-validation-standard/v1 또는 "
            "hmd-analysis-requirement-set/v1. 원본 표준 객체 전체를 그대로 전달합니다. " + size_limit
        ),
        "required": ["schema_version"],
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["report-validation-standard/v1", "hmd-analysis-requirement-set/v1"],
            },
            "standard_id": {"type": "string"},
            "standard_version": {"type": "string"},
            "status": {"type": "string"},
            "usage": {"type": "object"},
            "criteria": {"type": "array"},
            "requirement_set_id": {"type": "string"},
            "requirement_set_version": {"type": "string"},
            "analysis_requirements": {"type": "array"},
        },
        "additionalProperties": True,
    }


def _evidence_queries_schema(maximum_text_chars: int) -> dict[str, Any]:
    """근거 후보 검색 handler가 받는 queries 계약을 반환한다."""
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["criterion_id", "terms"],
            "properties": {
                "criterion_id": {"type": "string", "minLength": 1, "maxLength": maximum_text_chars},
                "element_id": {"type": "string", "minLength": 1, "maxLength": maximum_text_chars},
                "terms": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1, "maxLength": maximum_text_chars},
                },
                "aliases": {
                    "type": "array",
                    "description": "Station C가 승인 standard에서 전달한 표현 별칭입니다. 빈 값과 중복은 허용하지 않습니다.",
                    "items": {"type": "string", "minLength": 1, "maxLength": maximum_text_chars},
                },
                "pattern_ids": {
                    "type": "array",
                    "description": "서버 allow-list의 패턴 ID입니다. raw regex는 허용하지 않습니다.",
                    "items": {"type": "string", "enum": ["korean_duration_expression/v1"]},
                },
            },
            "additionalProperties": True,
        },
    }


def _semantic_assessment_schema(maximum_text_chars: int) -> dict[str, Any]:
    """선택 semantic 후보의 전달 형태를 공개한다.

    criterion별 형식 오류는 서버가 전체 호출을 거절하지 않고 uncertain으로
    정규화한다. 따라서 Gateway가 이 선택 후보를 과도하게 선행 거절하지 않도록
    내부 항목은 객체로만 제한한다.
    """
    citation = {
        "type": "object",
        "required": ["start", "end", "quote"],
        "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "quote": {"type": "string", "minLength": 1, "maxLength": maximum_text_chars},
        },
        "additionalProperties": True,
    }
    element = {
        "type": "object",
        "properties": {
            "element_id": {"type": "string", "maxLength": maximum_text_chars},
            "status": {"type": "string", "enum": ["satisfied", "missing", "insufficient", "uncertain"]},
            "reason": {"type": ["string", "null"], "maxLength": maximum_text_chars},
            "citations": {"type": "array", "items": citation},
        },
        "additionalProperties": True,
    }
    return {
        "type": "object",
        "required": ["criteria"],
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string", "maxLength": maximum_text_chars},
                        "status": {"type": "string", "enum": ["satisfied", "missing", "insufficient", "uncertain"]},
                        "reason": {"type": ["string", "null"], "maxLength": maximum_text_chars},
                        "citations": {"type": "array", "items": citation},
                        "elements": {"type": "array", "items": element},
                    },
                    "additionalProperties": True,
                },
            },
        },
        "additionalProperties": True,
    }
