"""Streamable HTTP MCP 도구와 비민감 healthcheck를 제공하는 진입점이다."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Annotated

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from .config import Settings
from .dictionary import DictionaryRepository
from .morphology import MorphologyAnalyzer
from .review import TextReviewer
from .recommendations import RecommendationCatalog


SERVICE_ID = "korean-standard-dictionary"
logger = logging.getLogger(__name__)


def create_server(settings: Settings | None = None) -> FastMCP:
    """의존성을 조립하고 `/mcp` 도구와 `/healthz`를 갖춘 MCP 서버를 만든다.

    Args:
        settings: 테스트 또는 배포 환경별 런타임 설정이다.

    Returns:
        Streamable HTTP 전송으로 실행할 준비가 된 FastMCP 인스턴스다.
    """
    settings = settings or Settings()
    repository = DictionaryRepository(settings.database_path)
    try:
        catalog = RecommendationCatalog.from_path(settings.recommendation_data_path)
        reviewer = TextReviewer(
            repository, MorphologyAnalyzer(), settings.max_candidates, catalog, max_findings=settings.max_findings
        )
    except Exception:
        reviewer = None
    mcp = FastMCP(
        SERVICE_ID,
        host=settings.host,
        port=settings.port,
        streamable_http_path="/mcp",
        stateless_http=True,
        # Gateway가 요청·응답 1회로 처리할 수 있도록 SSE 대신 JSON으로 응답한다.
        json_response=True,
        max_request_body_size=settings.max_request_body_size,
    )
    # FastMCP는 동기 도구 함수를 이벤트 루프에서 직접 실행한다. CPU·SQLite 작업은
    # 별도 thread로 옮기고, 이 limiter가 과도한 장문 검토의 동시 실행을 제한한다.
    execution_limiter = anyio.CapacityLimiter(settings.max_concurrent_requests)

    def error(code: str, message: str, retryable: bool) -> dict[str, object]:
        return {"error": {"code": code, "message": message, "retryable": retryable}}

    def request_id(context: Context) -> str:
        """Gateway의 요청 ID가 있으면 기록하고, 없으면 MCP JSON-RPC ID를 사용한다."""
        try:
            request = context.request_context.request
            headers = getattr(request, "headers", {})
            gateway_request_id = headers.get("x-request-id")
            candidate = str(gateway_request_id) if gateway_request_id else context.request_id
            return candidate.replace("\r", "").replace("\n", "")[:128] or "unavailable"
        except (AttributeError, ValueError):
            return "unavailable"

    def audit(tool_name: str, result_code: str, context: Context, started_at: float) -> None:
        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        logger.info(
            "audit service_id=%s service_version=%s dictionary_version=%s tool=%s request_id=%s result=%s duration_ms=%s",
            SERVICE_ID,
            settings.service_version,
            settings.dictionary_version,
            tool_name,
            request_id(context),
            result_code,
            duration_ms,
        )

    def tool_result(payload: dict[str, object], is_error: bool) -> CallToolResult:
        """모든 도구 결과를 Gateway가 해석 가능한 MCP 결과로 고정한다."""
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structuredContent=payload,
            isError=is_error,
        )

    def complete(
        tool_name: str,
        result_code: str,
        payload: dict[str, object],
        context: Context,
        started_at: float,
        *,
        is_error: bool = False,
    ) -> CallToolResult:
        audit(tool_name, result_code, context, started_at)
        return tool_result(payload, is_error)

    @mcp.tool(structured_output=False)
    async def lookup_standard_dictionary(query: Annotated[str, "조회할 표제어 또는 이형태"], context: Context) -> CallToolResult:
        """표제어 또는 검색용 이형태를 정확 일치 조회한다.

        query는 비어 있지 않아야 하며 구성된 텍스트 길이 상한(기본 10,000
        코드 포인트)을 넘을 수 없다. 결과는 구성된 상한(기본 20개)까지만
        반환하며, 부분·유사 검색이나 자동 교정은 제공하지 않는다.
        """
        started_at = perf_counter()
        if not isinstance(query, str) or not query.strip():
            return complete("lookup_standard_dictionary", "invalid_input", error("invalid_input", "query는 비어 있지 않은 문자열이어야 합니다.", False), context, started_at, is_error=True)
        if len(query) > settings.max_text_length:
            return complete("lookup_standard_dictionary", "request_too_large", error("request_too_large", "query가 허용 길이를 초과했습니다.", False), context, started_at, is_error=True)
        try:
            entries = await anyio.to_thread.run_sync(repository.lookup, query, settings.max_results, limiter=execution_limiter)
        except Exception:
            return complete("lookup_standard_dictionary", "upstream_unavailable", error("upstream_unavailable", "사전 DB를 현재 조회할 수 없습니다.", True), context, started_at, is_error=True)
        if not entries:
            return complete("lookup_standard_dictionary", "not_found", error("not_found", "일치하는 사전 항목이 없습니다.", False), context, started_at, is_error=True)
        return complete("lookup_standard_dictionary", "ok", {"entries": [entry.as_dict() for entry in entries]}, context, started_at)

    @mcp.tool(structured_output=False)
    async def review_standard_korean_text(text: Annotated[str, "검토할 한국어 문서 텍스트"], context: Context) -> CallToolResult:
        """한국어 문서를 사전 근거와 함께 검토하되 자동 교정을 확정하지 않는다.

        text는 비어 있지 않아야 하며 구성된 길이 상한(기본 10,000 코드
        포인트)을 넘을 수 없다. findings는 구성된 상한(기본 1,000개)까지만
        반환한다. 상한에 도달하면 부분 결과와 warnings의
        findings_truncated를 반환하므로, 호출자는 이를 자동 교정이나 완전한
        문서 판정으로 해석해서는 안 된다.
        """
        started_at = perf_counter()
        if not isinstance(text, str) or not text.strip():
            return complete("review_standard_korean_text", "invalid_input", error("invalid_input", "text는 비어 있지 않은 문자열이어야 합니다.", False), context, started_at, is_error=True)
        if len(text) > settings.max_text_length:
            return complete("review_standard_korean_text", "request_too_large", error("request_too_large", "text가 허용 길이를 초과했습니다.", False), context, started_at, is_error=True)
        is_ready = reviewer is not None and await anyio.to_thread.run_sync(repository.is_ready, limiter=execution_limiter)
        if not is_ready:
            return complete("review_standard_korean_text", "upstream_unavailable", error("upstream_unavailable", "분석기 또는 사전 DB를 사용할 수 없습니다.", True), context, started_at, is_error=True)
        try:
            result = await anyio.to_thread.run_sync(reviewer.review, text, limiter=execution_limiter)
        except Exception:
            return complete("review_standard_korean_text", "internal_error", error("internal_error", "텍스트 검토 중 예상하지 못한 오류가 발생했습니다.", True), context, started_at, is_error=True)
        return complete("review_standard_korean_text", "ok", result, context, started_at)

    async def healthz(request: object) -> PlainTextResponse:
        """민감한 내부 상태를 노출하지 않고 DB·분석기 준비 상태만 HTTP 상태로 반환한다."""
        # Healthcheck가 장문 검토의 대기열에 묶여 false negative가 되지 않도록
        # 짧은 readiness 조회는 별도 worker에서 실행한다.
        is_ready = reviewer is not None and await anyio.to_thread.run_sync(repository.is_ready)
        return PlainTextResponse("ok" if is_ready else "unavailable", status_code=200 if is_ready else 503)

    mcp._custom_starlette_routes = [Route("/healthz", healthz)]
    return mcp


def main() -> None:
    """운영 전송인 Streamable HTTP 모드로 MCP 서버를 시작한다."""
    create_server().run(transport="streamable-http")
