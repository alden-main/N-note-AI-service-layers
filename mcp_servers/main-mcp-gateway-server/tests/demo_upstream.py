"""Gateway 종단 간 검증용 읽기 전용 예제 MCP upstream이다(SDK Streamable HTTP, 세션 사용)."""

from collections.abc import AsyncIterator
import contextlib
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from main_mcp_gateway_server.app import MCP_PATH, _McpEndpoint


DEMO_TOOLS = {"korean": ["review_standard_korean_text"], "medic": ["analyze_document_terms"]}


def create_demo_upstream_app(service_id: str) -> Starlette:
    """service_id에 맞춘 도구를 노출하고, 호출하면 service_id와 도구명을 구조화 결과로 돌려준다."""
    server: Server = Server(service_id, version="demo")

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
