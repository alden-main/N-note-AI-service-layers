"""Gateway 내부에서 사용할 최소 Streamable HTTP MCP 진입점을 제공한다."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Callable
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from .contracts import InputError, ToolError
from .hmd_standard import HmdAnalysisRequirementSetV1Interpreter
from .observability import SafeAuditLog
from .standards import ReportValidationStandardV1Interpreter, StandardInterpreterRegistry
from .validation import ReportValidationService

EVALUATE_TOOL = "evaluate_report_against_standard"
GUIDANCE_TOOL = "get_validation_guidance"
EVIDENCE_SEARCH_TOOL = "find_evidence_candidates"
SERVICE_ID = "report-validation"
MAX_REQUEST_BYTES = 1_048_576
MAX_TEXT_CHARS = 100_000
SUPPORTED_PROTOCOL_VERSIONS = frozenset({"2025-06-18"})


class McpApplication:
    """MCP JSON-RPC와 도구 계약을 연결하는 상태 없는 WSGI 애플리케이션이다."""

    def __init__(self, service: ReportValidationService | None = None, audit: SafeAuditLog | None = None) -> None:
        """기본 검증 서비스와 민감 원문 없는 감사 저장소를 구성한다."""
        registry = StandardInterpreterRegistry((ReportValidationStandardV1Interpreter(), HmdAnalysisRequirementSetV1Interpreter()))
        self._service = service or ReportValidationService(registry)
        self._audit = audit or SafeAuditLog()
        self._configuration_ready = True
        try:
            self._max_request_bytes = _positive_limit("MCP_MAX_REQUEST_BYTES", MAX_REQUEST_BYTES)
            self._max_text_chars = _positive_limit("MCP_MAX_TEXT_CHARS", MAX_TEXT_CHARS)
        except ValueError:
            self._configuration_ready = False
            self._max_request_bytes = MAX_REQUEST_BYTES
            self._max_text_chars = MAX_TEXT_CHARS

    def evaluate(self, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """전용 Gateway network에서 전달된 평가 요청을 처리한다."""
        started_at = perf_counter()
        result_code = "internal_error"
        try:
            _enforce_text_limit(arguments, self._max_text_chars)
            report = arguments.get("report")
            result = self._service.evaluate(report, arguments.get("validation_standard"), arguments.get("semantic_assessment"))
            result_code = "ok"
            return result.as_dict()
        except RequestTooLarge:
            result_code = "request_too_large"
            return ToolError("request_too_large", "text input exceeds configured size limit").as_dict()
        except InputError:
            result_code = "invalid_input"
            return ToolError("invalid_input", "request does not satisfy the tool contract").as_dict()
        except (RuntimeError, TypeError):
            return ToolError("internal_error", "evaluation could not be completed", True).as_dict()
        finally:
            self._audit.record(
                request_id=request_id,
                tool_name=EVALUATE_TOOL,
                result_code=result_code,
                standard_version=arguments.get("validation_standard", {}).get("standard_version") if isinstance(arguments.get("validation_standard"), dict) else None,
                started_at=started_at,
            )

    def guidance(self, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """LLM용 검토 정책을 반환하고 호출 metadata만 감사한다.

        Args:
            arguments: `validation_standard`만 포함하는 도구 인자다.
        """
        started_at = perf_counter()
        result_code = "internal_error"
        try:
            _enforce_text_limit(arguments, self._max_text_chars)
            guidance = self._service.guidance(arguments.get("validation_standard"))
            result_code = "ok"
            return guidance
        except RequestTooLarge:
            result_code = "request_too_large"
            return ToolError("request_too_large", "text input exceeds configured size limit").as_dict()
        except InputError:
            result_code = "invalid_input"
            return ToolError("invalid_input", "request does not satisfy the tool contract").as_dict()
        except (RuntimeError, TypeError):
            return ToolError("internal_error", "guidance could not be prepared", True).as_dict()
        finally:
            self._audit.record(
                request_id=request_id,
                tool_name=GUIDANCE_TOOL,
                result_code=result_code,
                standard_version=arguments.get("validation_standard", {}).get("standard_version") if isinstance(arguments.get("validation_standard"), dict) else None,
                started_at=started_at,
            )

    def find_evidence_candidates(self, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """외부 AI의 검색어에 맞는 원문 후보 문장을 찾아 반환한다.

        이 도구는 문자 일치 검색만 수행한다. 반환된 후보가 기준을 충족하는지의
        의미 판단은 외부 AI와 평가 도구의 역할이다.
        """
        started_at = perf_counter()
        result_code = "internal_error"
        try:
            _enforce_text_limit(arguments, self._max_text_chars)
            report = arguments.get("report")
            result = self._service.find_evidence_candidates(report, arguments.get("validation_standard"), arguments.get("queries"))
            result_code = "ok"
            return result
        except RequestTooLarge:
            result_code = "request_too_large"
            return ToolError("request_too_large", "text input exceeds configured size limit").as_dict()
        except InputError:
            result_code = "invalid_input"
            return ToolError("invalid_input", "request does not satisfy the tool contract").as_dict()
        except (RuntimeError, TypeError):
            return ToolError("internal_error", "evidence candidates could not be found", True).as_dict()
        finally:
            self._audit.record(
                request_id=request_id,
                tool_name=EVIDENCE_SEARCH_TOOL,
                result_code=result_code,
                standard_version=arguments.get("validation_standard", {}).get("standard_version") if isinstance(arguments.get("validation_standard"), dict) else None,
                started_at=started_at,
            )

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        """`/mcp` JSON-RPC와 `/healthz`를 처리한다.

        Docker의 Gateway 전용 내부 network가 호출 신뢰 경계다. 이 서버는
        Authorization, JWT, API key 및 호출자 claim을 해석하거나 저장하지 않는다.
        """
        path = environ.get("PATH_INFO", "")
        if path == "/healthz":
            ready = self._is_ready()
            return self._json(start_response, "200 OK" if ready else "503 Service Unavailable", self._audit.health(ready))
        if path != "/mcp" or environ.get("REQUEST_METHOD") != "POST":
            return self._json(start_response, "404 Not Found", {"error": {"code": "not_found", "message": "not found", "retryable": False}})
        try:
            content_length = environ.get("CONTENT_LENGTH")
            if not content_length:
                return self._json(start_response, "411 Length Required", ToolError("invalid_input", "Content-Length header is required").as_dict())
            request_bytes = int(content_length)
            if request_bytes < 0:
                raise ValueError("CONTENT_LENGTH must not be negative")
            if request_bytes > self._max_request_bytes:
                return self._json(start_response, "413 Payload Too Large", _tool_error_response(None, "request_too_large", "request exceeds configured size limit"))
            # WSGI input is an unbounded socket stream.  Reading the configured
            # maximum here can wait for bytes that a normal keep-alive client will
            # never send, preventing even tools/list from receiving a response.
            raw = environ["wsgi.input"].read(request_bytes)
            if len(raw) != request_bytes:
                raise ValueError("request body is shorter than Content-Length")
            request = json.loads(raw)
            response = self._handle_rpc(request, environ.get("HTTP_X_REQUEST_ID"))
            if response is None:
                start_response("202 Accepted", [("Content-Length", "0")])
                return [b""]
            return self._json(start_response, "200 OK", response)
        except (ValueError, json.JSONDecodeError):
            return self._json(start_response, "400 Bad Request", ToolError("invalid_input", "invalid MCP request").as_dict())

    def _handle_rpc(self, request: Any, request_id: str | None = None) -> dict[str, Any] | None:
        """MCP tools/list와 tools/call의 최소 JSON-RPC 호환 응답을 만든다."""
        if not isinstance(request, dict):
            return ToolError("invalid_input", "MCP request must be an object").as_dict()
        rpc_id = request.get("id")
        if request.get("method") == "initialize":
            params = request.get("params")
            if not isinstance(params, dict) or not isinstance(params.get("protocolVersion"), str):
                return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "invalid initialize parameters"}}
            if params["protocolVersion"] not in SUPPORTED_PROTOCOL_VERSIONS:
                return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "unsupported protocol version"}}
            return {"jsonrpc": "2.0", "id": rpc_id, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "report-validation", "version": "0.1.0"},
            }}
        if request.get("method") == "notifications/initialized":
            return None
        if request.get("method") == "tools/list":
            return {"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": self._tools()}}
        if request.get("method") == "tools/call":
            params = request.get("params")
            if not isinstance(params, dict) or not isinstance(params.get("arguments"), dict):
                payload = ToolError("invalid_input", "unknown tool or invalid arguments").as_dict()
            elif not self._is_ready():
                payload = ToolError("upstream_unavailable", "service is not ready", True).as_dict()
            elif _canonical_tool_name(params.get("name")) == GUIDANCE_TOOL:
                payload = self.guidance(params["arguments"], request_id)
            elif _canonical_tool_name(params.get("name")) == EVIDENCE_SEARCH_TOOL:
                payload = self.find_evidence_candidates(params["arguments"], request_id)
            elif _canonical_tool_name(params.get("name")) == EVALUATE_TOOL:
                payload = self.evaluate(params["arguments"], request_id)
            else:
                payload = ToolError("invalid_input", "unknown tool or invalid arguments").as_dict()
            return {"jsonrpc": "2.0", "id": rpc_id, "result": {"structuredContent": payload, "isError": "error" in payload}}
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "method not found"}}

    def _tools(self) -> list[dict[str, Any]]:
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

    def _is_ready(self) -> bool:
        """필수 설정과 기준 interpreter가 준비된 경우에만 true를 반환한다."""
        registry = getattr(self._service, "_registry", None)
        interpreters = getattr(registry, "_interpreters", None)
        return self._configuration_ready and isinstance(interpreters, dict) and bool(interpreters)

    @staticmethod
    def _json(start_response: Callable[..., Any], status: str, payload: dict[str, Any]) -> list[bytes]:
        """JSON 응답을 WSGI 형식으로 직렬화한다."""
        body = json.dumps(payload, ensure_ascii=False).encode()
        start_response(status, [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return [body]


def main() -> None:
    """Gateway 전용 내부 HTTP 서버를 시작한다.

    운영 환경에서는 host port를 publish하지 않고 Gateway 전용 Docker network에서만
    이 callable을 구동해야 한다. MCP_HOST와 MCP_PORT는 컨테이너 내부 listener만
    설정하며 외부 공개 여부는 Docker Compose의 ports 설정이 결정한다.
    """
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    create_server(host, port, McpApplication()).serve_forever()


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """요청마다 스레드를 써서 긴 검증 중에도 다른 호출과 `/healthz`를 처리한다.

    `McpApplication`은 요청 간에 바뀌는 상태가 없다(감사 이벤트 추가만 스레드 안전한 deque에 한다).
    """

    daemon_threads = True


def create_server(host: str, port: int, app: Callable[..., Any]) -> WSGIServer:
    """동시 요청을 처리하는 WSGI 서버를 만든다."""
    return make_server(host, port, app, server_class=ThreadingWSGIServer)


class RequestTooLarge(ValueError):
    """도구 인자의 개별 문자열이 허용 한도를 넘었음을 나타낸다."""


def _positive_limit(environment_name: str, default: int) -> int:
    """양의 정수 환경 설정만 입력 한도로 허용한다."""
    value = int(os.getenv(environment_name, str(default)))
    if value <= 0:
        raise ValueError(f"{environment_name} must be positive")
    return value


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


def _tool_error_response(request_id: Any, code: str, message: str) -> dict[str, Any]:
    """파싱 전 크기 초과도 Gateway가 읽을 수 있는 tool-result 형식으로 만든다."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"structuredContent": ToolError(code, message).as_dict(), "isError": True},
    }


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


def _canonical_tool_name(name: Any) -> str | None:
    """Gateway catalog가 붙이는 이 서비스 접두사를 내부 도구명으로 되돌린다."""
    if not isinstance(name, str):
        return None
    if name in {GUIDANCE_TOOL, EVIDENCE_SEARCH_TOOL, EVALUATE_TOOL}:
        return name
    for separator in ("__", ":", "/", "."):
        prefix = SERVICE_ID + separator
        if name.startswith(prefix):
            candidate = name.removeprefix(prefix)
            if candidate in {GUIDANCE_TOOL, EVIDENCE_SEARCH_TOOL, EVALUATE_TOOL}:
                return candidate
    return None
