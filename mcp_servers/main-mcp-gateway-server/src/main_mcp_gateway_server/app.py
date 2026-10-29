"""Gateway MCP HTTP endpoint와 Compose 검증용 최소 upstream 서버를 제공한다."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import uuid
from typing import Any

from .config import configured_upstreams
from .mcp import DEFAULT_BATCH_MAX_CONCURRENCY, DEFAULT_UPSTREAM_TIMEOUT_SECONDS, GatewayCatalog, HttpTransport, PROTOCOL_VERSION


DEFAULT_MAX_REQUEST_BODY_BYTES = 1_048_576
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_HEALTH_TTL_SECONDS = 5.0
DEFAULT_BATCH_MAX_CALLS = 32
BATCH_CALL_PATH = "/v1/tools/batch-call"


class GatewayHandler(BaseHTTPRequestHandler):
    """`/mcp`에서 MCP initialize, tools/list, tools/call을, `/v1/tools/batch-call`에서 일괄 도구 호출을 처리한다."""

    # keep-alive로 호출자(AI Service)가 연결을 재사용하게 한다. 모든 응답은 Content-Length를 보낸다.
    protocol_version = "HTTP/1.1"
    # 유휴 keep-alive 연결을 붙잡은 handler 스레드를 이 시간(초) 뒤 정리한다.
    timeout = 60
    catalog: GatewayCatalog
    max_request_body_bytes = DEFAULT_MAX_REQUEST_BODY_BYTES
    max_batch_calls = DEFAULT_BATCH_MAX_CALLS

    def do_GET(self) -> None:
        """Gateway readiness를 민감 정보 없이 `/healthz`로 제공한다."""
        if self.path == "/healthz":
            self._send(HTTPStatus.OK, b"ok", "text/plain")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        """MCP JSON-RPC 또는 일괄 호출 요청을 크기 제한 안에서 처리하고 request ID를 연결한다."""
        request_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
        if self.path not in ("/mcp", BATCH_CALL_PATH):
            # 읽지 않은 본문이 다음 요청으로 해석되지 않게 연결을 닫는다.
            self.close_connection = True
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain", request_id)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 0 or content_length > self.max_request_body_bytes:
                self.close_connection = True
                if self.path == BATCH_CALL_PATH:
                    self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b'{"error":"request_too_large"}', "application/json", request_id)
                else:
                    self._send_tool_error(None, "request_too_large", "요청 본문이 Gateway 제한을 초과했습니다", request_id)
                return
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("요청 본문은 객체여야 합니다")
            if self.path == BATCH_CALL_PATH:
                self._handle_batch(body, request_id)
                return
            result = self._handle(body, request_id)
            self._send(HTTPStatus.OK, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json", request_id)
        except (ValueError, json.JSONDecodeError):
            self._send(HTTPStatus.BAD_REQUEST, b'{"error":"invalid JSON-RPC request"}', "application/json", request_id)

    def _handle_batch(self, body: dict[str, Any], request_id: str) -> None:
        """``{"calls": [{"name", "arguments"}, ...]}``를 동시에 중계하고 ``{"results": [...]}``를 입력 순서대로 반환한다.

        각 result는 tools/call과 같은 MCP tool result 계약(감사 로그 포함)을 따른다.
        """
        calls = body.get("calls")
        if not isinstance(calls, list) or not calls:
            self._send(HTTPStatus.BAD_REQUEST, b'{"error":"calls must be a non-empty array"}', "application/json", request_id)
            return
        if len(calls) > self.max_batch_calls:
            self._send(HTTPStatus.BAD_REQUEST, b'{"error":"too_many_calls"}', "application/json", request_id)
            return
        results = self.catalog.call_many(calls, request_id)
        self._send(HTTPStatus.OK, json.dumps({"results": results}, ensure_ascii=False).encode("utf-8"), "application/json", request_id)

    def _handle(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        """지원 MCP method를 JSON-RPC response로 변환한다.

        body의 method와 params를 검증하고, 미지원 method 및 알 수 없는 도구는 upstream으로 전달하지 않는다.
        request_id는 Gateway가 생성했거나 호출자가 전달한 값으로, tools/call upstream 요청에 유지한다.
        """
        message_id = body.get("id")
        method = body.get("method")
        params = body.get("params") or {}
        if method == "initialize":
            if params.get("protocolVersion") != PROTOCOL_VERSION:
                return {"jsonrpc": "2.0", "id": message_id, "error": {"code": -32602, "message": "지원하지 않는 MCP protocolVersion입니다"}}
            return _response(message_id, {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}}, "serverInfo": {"name": "mcp-gateway", "version": "0.1.0"}})
        if method == "notifications/initialized":
            return _response(message_id, {})
        if method == "tools/list":
            # Gateway가 upstream보다 먼저 올라와도 다음 client discovery에서 다시 연결한다.
            self.catalog.discover_all()
            return _response(message_id, {"tools": self.catalog.list_tools()})
        if method == "tools/call" and isinstance(params.get("name"), str) and isinstance(params.get("arguments", {}), dict):
            return _response(message_id, self.catalog.call(params["name"], params.get("arguments", {}), request_id))
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": -32601, "message": "지원하지 않는 MCP method 또는 params입니다"}}

    def _send_tool_error(self, message_id: Any, code: str, message: str, request_id: str) -> None:
        """본문 제한처럼 upstream 없이 처리한 실패를 MCP tool result로 반환한다."""
        error = {"code": code, "message": message, "retryable": False}
        result = {
            "content": [{"type": "text", "text": json.dumps({"error": error}, ensure_ascii=False)}],
            "structuredContent": {"error": error},
            "isError": True,
        }
        self._send(HTTPStatus.OK, json.dumps(_response(message_id, result), ensure_ascii=False).encode("utf-8"), "application/json", request_id)

    def _send(self, status: HTTPStatus, content: bytes, content_type: str, request_id: str | None = None) -> None:
        """HTTP 응답을 쓰되, timeout으로 먼저 끊긴 클라이언트는 조용히 종료한다."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        if self.close_connection:
            self.send_header("Connection", "close")
        if request_id:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            logging.info("gateway_http client_disconnected_before_response")

    def log_message(self, format: str, *args: object) -> None:
        """도구 인수나 원문을 기록하지 않고 HTTP 상태만 표준 logging으로 남긴다."""
        logging.info("gateway_http status=%s", args[1] if len(args) > 1 else "unknown")


class DemoUpstreamHandler(BaseHTTPRequestHandler):
    """Compose 종단 간 검증을 위한 읽기 전용 예제 MCP upstream을 제공한다."""

    service_id = ""

    def do_GET(self) -> None:
        """예제 upstream의 readiness를 `/healthz`로 반환한다."""
        self.send_response(HTTPStatus.OK if self.path == "/healthz" else HTTPStatus.NOT_FOUND)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self) -> None:
        """예제 MCP의 initialize, tools/list, tools/call 요청을 처리한다."""
        if self.path != "/mcp":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
        method, params = body.get("method"), body.get("params") or {}
        if method == "initialize":
            result: dict[str, Any] = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": self.service_id, "version": "0.1.0"}}
        elif method == "tools/list":
            result = {"tools": _demo_tools(self.service_id)}
        elif method == "notifications/initialized":
            result = {}
        elif method == "tools/call":
            result = {
                "content": [{"type": "text", "text": "demo tool call complete"}],
                "structuredContent": {"service_id": self.service_id, "tool": params.get("name")},
                "isError": False,
            }
        else:
            result = {"content": [{"type": "text", "text": "unsupported"}], "isError": True}
        raw = json.dumps(_response(body.get("id"), result), ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _response(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _demo_tools(service_id: str) -> list[dict[str, Any]]:
    tools = {"korean-standard-dictionary": ["review_standard_korean_text"], "medical-term-review": ["analyze_document_terms"]}
    return [{"name": name, "description": f"{service_id} 예제 도구", "inputSchema": {"type": "object", "additionalProperties": True}} for name in tools[service_id]]


def serve_gateway(port: int) -> None:
    """discovery를 수행한 Gateway HTTP 서버를 지정 port에서 실행한다."""
    catalog = GatewayCatalog(
        configured_upstreams(),
        HttpTransport(_upstream_timeout_seconds()),
        health_ttl_seconds=_non_negative_float("GATEWAY_HEALTH_TTL_SECONDS", DEFAULT_HEALTH_TTL_SECONDS),
        batch_max_concurrency=_positive_int("GATEWAY_BATCH_MAX_CONCURRENCY", DEFAULT_BATCH_MAX_CONCURRENCY),
    )
    catalog.discover_all()
    GatewayHandler.catalog = catalog
    GatewayHandler.max_request_body_bytes = _max_request_body_bytes()
    GatewayHandler.max_batch_calls = _positive_int("GATEWAY_BATCH_MAX_CALLS", DEFAULT_BATCH_MAX_CALLS)
    ThreadingHTTPServer(("0.0.0.0", port), GatewayHandler).serve_forever()


def _max_request_body_bytes() -> int:
    """환경 변수의 양의 정수 Gateway 요청 본문 제한을 검증해 반환한다."""
    configured = os.getenv("GATEWAY_MAX_REQUEST_BODY_BYTES", str(DEFAULT_MAX_REQUEST_BODY_BYTES))
    try:
        limit = int(configured)
    except ValueError as error:
        raise ValueError("GATEWAY_MAX_REQUEST_BODY_BYTES는 양의 정수여야 합니다") from error
    if limit <= 0:
        raise ValueError("GATEWAY_MAX_REQUEST_BODY_BYTES는 양의 정수여야 합니다")
    return limit


def _non_negative_float(name: str, default: float) -> float:
    """환경 변수의 0 이상 실수 설정을 검증해 반환한다. 0이면 해당 캐시를 끈다."""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name}는 0 이상의 숫자여야 합니다") from error
    if not value >= 0:
        raise ValueError(f"{name}는 0 이상의 숫자여야 합니다")
    return value


def _positive_int(name: str, default: int) -> int:
    """환경 변수의 양의 정수 설정을 검증해 반환한다."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name}는 양의 정수여야 합니다") from error
    if value <= 0:
        raise ValueError(f"{name}는 양의 정수여야 합니다")
    return value


def _upstream_timeout_seconds() -> float:
    """환경 변수의 양수 upstream 요청 timeout(초)을 검증해 반환한다.

    호출자(Station C 등)의 Gateway timeout보다 작게 설정해야 upstream 지연이
    호출자 timeout이 아니라 구조화된 `upstream_unavailable` 결과로 전달된다.
    """
    configured = os.getenv("GATEWAY_UPSTREAM_TIMEOUT_SECONDS", str(DEFAULT_UPSTREAM_TIMEOUT_SECONDS))
    try:
        timeout = float(configured)
    except ValueError as error:
        raise ValueError("GATEWAY_UPSTREAM_TIMEOUT_SECONDS는 양수여야 합니다") from error
    if not timeout > 0:
        raise ValueError("GATEWAY_UPSTREAM_TIMEOUT_SECONDS는 양수여야 합니다")
    return timeout


def configure_logging() -> None:
    """Gateway 컨테이너 표준 오류 출력에 운영 추적용 로그 형식을 설정한다.

    `GATEWAY_LOG_LEVEL`은 DEBUG, INFO, WARNING, ERROR, CRITICAL 중 하나이며,
    설정하지 않으면 도구 호출 시간과 연결 종료를 확인할 수 있는 INFO를 사용한다.
    """
    configured = os.getenv("GATEWAY_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, configured, None)
    if not isinstance(level, int) or configured not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("GATEWAY_LOG_LEVEL은 DEBUG, INFO, WARNING, ERROR, CRITICAL 중 하나여야 합니다")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def serve_demo_upstream(service_id: str, port: int) -> None:
    """service_id에 맞춘 예제 upstream MCP 서버를 지정 port에서 실행한다."""
    handler = type("ConfiguredDemoUpstreamHandler", (DemoUpstreamHandler,), {"service_id": service_id})
    ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()


def main() -> None:
    """명령행 인수에 따라 Gateway 또는 Compose 검증용 upstream을 실행한다."""
    configure_logging()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    gateway = subparsers.add_parser("gateway")
    gateway.add_argument("--port", type=int, default=8080)
    upstream = subparsers.add_parser("demo-upstream")
    upstream.add_argument("--service-id", choices=("korean-standard-dictionary", "medical-term-review"), required=True)
    upstream.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "gateway":
        serve_gateway(args.port)
    else:
        serve_demo_upstream(args.service_id, args.port)
