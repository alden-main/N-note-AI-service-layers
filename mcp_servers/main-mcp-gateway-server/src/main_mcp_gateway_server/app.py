"""Gateway MCP endpoint(MCP Python SDK의 Streamable HTTP)와 Compose 검증용 최소 upstream 서버를 제공한다.

`/mcp`는 SDK 저수준 ``Server``가 처리한다. 도구 목록은 upstream catalog를 실행 중에 중계하므로 FastMCP의
데코레이터 등록 대신 ``list_tools``/``call_tool`` 핸들러를 직접 둔다. 상태 없는(stateless) JSON 응답 모드라
Gateway는 세션을 보관하지 않고, 어떤 표준 MCP 클라이언트든 붙을 수 있다.
`/v1/tools/batch-call`(일괄 호출)과 `/healthz`는 같은 Starlette 앱의 일반 경로다.
"""

import argparse
import contextlib
from collections.abc import AsyncIterator
from http import HTTPStatus
import json
import logging
import os
import uuid
from typing import Any

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import configured_upstreams
from .mcp import DEFAULT_BATCH_MAX_CONCURRENCY, DEFAULT_CATALOG_TTL_SECONDS, DEFAULT_UPSTREAM_TIMEOUT_SECONDS, GatewayCatalog, HttpTransport, package_version


DEFAULT_MAX_REQUEST_BODY_BYTES = 1_048_576
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_HEALTH_TTL_SECONDS = 5.0
DEFAULT_BATCH_MAX_CALLS = 32
BATCH_CALL_PATH = "/v1/tools/batch-call"
MCP_PATH = "/mcp"
KEEP_ALIVE_SECONDS = 60
SERVER_NAME = "mcp-gateway"
SERVER_VERSION = package_version()


class GatewayService:
    """Gateway 설정과 `/v1/tools/batch-call` 일괄 호출 처리다(`/mcp`는 SDK 서버가 처리한다)."""

    def __init__(self, catalog: Any, max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES, max_batch_calls: int = DEFAULT_BATCH_MAX_CALLS) -> None:
        self.catalog = catalog
        self.max_request_body_bytes = max_request_body_bytes
        self.max_batch_calls = max_batch_calls

    def handle_batch(self, body: Any, request_id: str) -> tuple[int, dict[str, Any]]:
        """``{"calls": [{"name", "arguments"}, ...]}``를 동시에 중계하고 ``{"results": [...]}``를 입력 순서대로 반환한다.

        각 result는 tools/call과 같은 MCP tool result 계약(감사 로그 포함)을 따른다.
        """
        calls = body.get("calls") if isinstance(body, dict) else None
        if not isinstance(calls, list) or not calls:
            return HTTPStatus.BAD_REQUEST, {"error": "calls must be a non-empty array"}
        if len(calls) > self.max_batch_calls:
            return HTTPStatus.BAD_REQUEST, {"error": "too_many_calls"}
        return HTTPStatus.OK, {"results": self.catalog.call_many(calls, request_id)}


def create_mcp_server(catalog: Any) -> Server:
    """upstream catalog를 중계하는 MCP 서버다. catalog는 동기 HTTP를 쓰므로 스레드에서 호출한다."""
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        # catalog TTL이 지난 서비스만 다시 찾는다. 연결되지 않은 upstream은 5초마다 다시 시도해 늦게 떠도 곧 나타난다.
        await anyio.to_thread.run_sync(catalog.refresh)
        tools = await anyio.to_thread.run_sync(catalog.list_tools)
        return [types.Tool.model_validate(tool) for tool in tools]

    # 입력 검증은 upstream이 자기 계약으로 한다(Gateway는 이름 공간 중계와 감사만 맡는다).
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        request_id = _request_id(server)
        result = await anyio.to_thread.run_sync(lambda: catalog.call(name, arguments, request_id))
        if "content" not in result:
            # MCP는 content를 필수로 둔다. 구조화 결과만 준 upstream은 그 JSON을 텍스트로도 싣는다(사양 권장).
            text = json.dumps(result.get("structuredContent"), ensure_ascii=False)
            result = {**result, "content": [{"type": "text", "text": text}]}
        return types.CallToolResult.model_validate(result)

    return server


def _request_id(server: Server) -> str | None:
    """요청 ID 미들웨어가 헤더에 넣어 둔 ``X-Request-ID``다(upstream 호출과 감사 로그에 이어 붙인다)."""
    try:
        request = server.request_context.request
    except LookupError:
        return None
    return request.headers.get("x-request-id") if request is not None else None


def create_app(service: GatewayService) -> Starlette:
    """Gateway HTTP 앱이다: `/mcp`(SDK Streamable HTTP), `/v1/tools/batch-call`, `/healthz`."""
    session_manager = StreamableHTTPSessionManager(
        app=create_mcp_server(service.catalog), stateless=True, json_response=True,
        max_request_body_size=service.max_request_body_bytes,
    )

    async def healthz(_request: Request) -> Response:
        """Gateway readiness를 민감 정보 없이 제공한다."""
        return PlainTextResponse("ok")

    async def batch(request: Request) -> Response:
        """일괄 호출을 크기 제한 안에서 처리한다."""
        request_id = request.headers["x-request-id"]
        raw = await _read_limited(request, service.max_request_body_bytes)
        if raw is None:
            status, payload = HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"}
        else:
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                status, payload = HTTPStatus.BAD_REQUEST, {"error": "invalid JSON request"}
            else:
                status, payload = await run_in_threadpool(service.handle_batch, body, request_id)
        return Response(json.dumps(payload, ensure_ascii=False), status_code=status, media_type="application/json")

    async def not_found(_request: Request) -> Response:
        return PlainTextResponse("not found", status_code=HTTPStatus.NOT_FOUND)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route(MCP_PATH, _McpEndpoint(session_manager), methods=["GET", "POST", "DELETE"]),
            Route(BATCH_CALL_PATH, batch, methods=["POST"]),
            Route("/{path:path}", not_found, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
        ],
        lifespan=lifespan,
    )
    return _RequestIdMiddleware(app)  # type: ignore[return-value]


class _McpEndpoint:
    """Starlette 경로에서 SDK 세션 관리자로 ASGI 요청을 넘긴다."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


class _RequestIdMiddleware:
    """요청마다 ``X-Request-ID``를 정한다(받은 값 또는 새 UUID). 요청 헤더에 넣어 핸들러가 읽게 하고 응답에도 붙인다.

    도구 인수나 원문은 기록하지 않고 HTTP 상태만 남긴다.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = [(key, value) for key, value in scope["headers"] if key != b"x-request-id"]
        received = next((value for key, value in scope["headers"] if key == b"x-request-id" and value), None)
        request_id = received or str(uuid.uuid4()).encode()
        scope = {**scope, "headers": [*headers, (b"x-request-id", request_id)]}

        async def send_with_id(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", request_id)]}
                logging.info("gateway_http status=%s", message["status"])
            await send(message)

        await self.app(scope, receive, send_with_id)


async def _read_limited(request: Request, limit: int) -> bytes | None:
    """본문을 ``limit`` 바이트까지만 읽는다. 넘으면(선언한 Content-Length 포함) ``None``이다."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) < 0 or int(declared) > limit:
                return None
        except ValueError:
            return b""
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


DEMO_TOOLS = {"korean": ["review_standard_korean_text"], "medic": ["analyze_document_terms"]}


def create_demo_upstream_app(service_id: str) -> Starlette:
    """Compose 종단 간 검증을 위한 읽기 전용 예제 MCP upstream이다(SDK Streamable HTTP, 세션 사용)."""
    server: Server = Server(service_id, version=SERVER_VERSION)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=name, description=f"{service_id} 예제 도구", inputSchema={"type": "object", "additionalProperties": True})
            for name in DEMO_TOOLS[service_id]
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, _arguments: dict[str, Any]) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="demo tool call complete")],
            structuredContent={"service_id": service_id, "tool": name},
            isError=False,
        )

    # 실제 upstream(kodict 등)과 다르게 세션·SSE를 쓰는 기본 모드로 둬, Gateway의 세션 처리도 함께 검증한다.
    session_manager = StreamableHTTPSessionManager(app=server)

    async def healthz(_request: Request) -> Response:
        return PlainTextResponse("ok")

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    return Starlette(
        routes=[Route("/healthz", healthz, methods=["GET"]), Route(MCP_PATH, _McpEndpoint(session_manager), methods=["GET", "POST", "DELETE"])],
        lifespan=lifespan,
    )


def _serve(app: Starlette, port: int) -> None:
    """uvicorn으로 실행한다. 접근 로그는 끄고(경로·주소만이라도 남기지 않음) 상태는 앱이 기록한다."""
    import uvicorn

    # 호출자(AI Service)의 keep-alive 연결을 재사용하도록 유휴 연결을 잠시 유지한다.
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False, log_config=None, timeout_keep_alive=KEEP_ALIVE_SECONDS)


def serve_gateway(port: int) -> None:
    """discovery를 수행한 Gateway HTTP 서버를 지정 port에서 실행한다."""
    catalog = GatewayCatalog(
        configured_upstreams(),
        HttpTransport(_upstream_timeout_seconds()),
        health_ttl_seconds=_non_negative_float("GATEWAY_HEALTH_TTL_SECONDS", DEFAULT_HEALTH_TTL_SECONDS),
        batch_max_concurrency=_positive_int("GATEWAY_BATCH_MAX_CONCURRENCY", DEFAULT_BATCH_MAX_CONCURRENCY),
        catalog_ttl_seconds=_non_negative_float("GATEWAY_CATALOG_TTL_SECONDS", DEFAULT_CATALOG_TTL_SECONDS),
    )
    catalog.discover_all()
    service = GatewayService(catalog, _max_request_body_bytes(), _positive_int("GATEWAY_BATCH_MAX_CALLS", DEFAULT_BATCH_MAX_CALLS))
    _serve(create_app(service), port)


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
    _serve(create_demo_upstream_app(service_id), port)


def main() -> None:
    """명령행 인수에 따라 Gateway 또는 Compose 검증용 upstream을 실행한다."""
    configure_logging()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    gateway = subparsers.add_parser("gateway")
    gateway.add_argument("--port", type=int, default=8080)
    upstream = subparsers.add_parser("demo-upstream")
    upstream.add_argument("--service-id", choices=("korean", "medic"), required=True)
    upstream.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "gateway":
        serve_gateway(args.port)
    else:
        serve_demo_upstream(args.service_id, args.port)
