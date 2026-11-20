"""Gateway(읽기 전용 MCP)를 호출하는 Streamable HTTP MCP 클라이언트다.

MCP 2025-06-18 Streamable HTTP 규칙을 따른다: ``initialize`` → ``notifications/initialized`` 알림,
서버가 준 ``Mcp-Session-Id``와 ``MCP-Protocol-Version`` 헤더를 이후 요청에 붙이고, JSON 또는 SSE 응답을 받는다.
세션이 만료되면(세션 ID를 보냈는데 404) 한 번 다시 초기화하고 재시도한다. 그래서 표준 MCP 서버(FastMCP 등)와
상태 없는(stateless) 서버 모두에 붙는다. 일괄 호출(``/v1/tools/batch-call``)만 Gateway 확장 경로다.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2
from jsonschema import Draft202012Validator

from n_note_ai_service_layers import __version__
from n_note_ai_service_layers.observability import ContextThreadPoolExecutor
from n_note_ai_service_layers.settings import DEFAULT_MCP_GATEWAY_TIMEOUT_SECONDS, DEFAULT_MCP_GATEWAY_URL

MCP_PROTOCOL_VERSION = "2025-06-18"
# Streamable HTTP 서버는 두 형식을 모두 받겠다고 해야 요청을 받는다(아니면 406).
MCP_ACCEPT = "application/json, text/event-stream"
CLIENT_INFO = {"name": "n-note-ai-service", "version": __version__}
# 이 클라이언트(프로세스에서 하나를 공유)가 일괄·병렬 호출에 쓰는 스레드 수 상한이다.
DEFAULT_POOL_WORKERS = 16
# A catalog missing expected tools (an upstream MCP still starting or down) is refreshed this often.
INCOMPLETE_CATALOG_TTL_SECONDS = 15.0
# Gateway의 일괄 도구 호출 경로다. MCP endpoint와 같은 host·접두 경로를 쓴다.
BATCH_CALL_PATH = "/v1/tools/batch-call"
# Gateway가 일괄 호출 한 번에 받는 최대 호출 수(GATEWAY_BATCH_MAX_CALLS 기본값)다. 넘으면 요청 전체를 거부하므로 나눠 보낸다.
DEFAULT_BATCH_MAX_CALLS = 32
HttpTransport = Callable[[str, bytes, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]


class McpGatewayError(RuntimeError):
    """Base error for an MCP Gateway call; never includes tool arguments."""


class McpGatewayConnectionError(McpGatewayError):
    """The Gateway could not be reached before it returned JSON-RPC."""


class McpGatewayProtocolError(McpGatewayError):
    """The Gateway returned malformed, HTTP-error, or JSON-RPC-error output."""


class McpSessionExpired(McpGatewayProtocolError):
    """세션 ID를 보낸 요청에 서버가 404로 답했다. 다시 초기화해야 한다."""


class McpGatewayToolError(McpGatewayError):
    """A successful HTTP request returned an MCP ``isError`` tool result."""

    def __init__(self, code: str, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"Gateway tool failed: {code}")


class McpGatewayClient:
    """Call only the configured Gateway endpoint and cache its tool catalog in memory.

    One instance is shared across requests and threads: the HTTP connection pool is
    reused, and ``discover`` reuses the catalog for ``catalog_ttl_seconds`` (0 = always refresh).
    While any of ``expected_tools`` is missing, the catalog is reused only for
    ``incomplete_catalog_ttl_seconds`` so a late or recovered upstream shows up quickly.
    """

    def __init__(
        self,
        url: str | None = None,
        timeout_seconds: float | None = None,
        transport: HttpTransport | None = None,
        catalog_ttl_seconds: float = 0.0,
        batch_url: str | None = None,
        expected_tools: tuple[str, ...] = (),
        incomplete_catalog_ttl_seconds: float = INCOMPLETE_CATALOG_TTL_SECONDS,
        batch_max_calls: int = DEFAULT_BATCH_MAX_CALLS,
        pool_workers: int = DEFAULT_POOL_WORKERS,
    ) -> None:
        self._url = url or DEFAULT_MCP_GATEWAY_URL
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else DEFAULT_MCP_GATEWAY_TIMEOUT_SECONDS
        # httpx2 client는 스레드 간에 공유하는 keep-alive 연결 풀이다. Gateway는 내부망이라 프록시 환경 변수를 따르지 않는다.
        self._http = httpx2.Client(trust_env=False) if transport is None else None
        self._transport = transport or self._http_post
        self._catalog: tuple[dict[str, Any], ...] = ()
        self._catalog_ttl_seconds = catalog_ttl_seconds
        self._expected_tools = frozenset(expected_tools)
        self._incomplete_catalog_ttl_seconds = incomplete_catalog_ttl_seconds
        self._catalog_loaded_at: float | None = None
        self._catalog_lock = threading.Lock()
        # Streamable HTTP 세션이다. 상태 없는 서버는 세션 ID를 주지 않으므로 None으로 남는다.
        self._session_id: str | None = None
        self._negotiated_version: str | None = None
        self._session_lock = threading.Lock()
        self._batch_url = batch_url or _batch_url(self._url)
        self._batch_max_calls = max(1, batch_max_calls)
        # 호출마다 풀을 만들지 않고 클라이언트 수명 동안 재사용한다. 이 풀의 작업은 HTTP 요청만 하고
        # 다른 작업을 기다리지 않으므로, 위 계층(초안 검토 등)이 기다려도 교착이 생기지 않는다.
        self._pool = ContextThreadPoolExecutor(max_workers=max(1, pool_workers), thread_name_prefix="gateway-call")

    @property
    def catalog(self) -> tuple[dict[str, Any], ...]:
        """Return the last Gateway-provided catalog without inventing tool definitions."""
        return self._catalog

    def initialize(self, request_id: str | None = None) -> dict[str, Any]:
        """세션을 새로 연다: ``initialize`` 요청 → 세션 ID 저장 → ``notifications/initialized`` 알림."""
        # 새 세션이 준비될 때까지 다른 스레드의 호출은 기존 세션을 계속 쓴다. 핸드셰이크에는 기존 세션을 싣지 않는다.
        with self._session_lock:
            params = {"protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO}
            result, headers = self._exchange("initialize", params, request_id, session=(None, None))
            if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
                raise McpGatewayProtocolError("unsupported MCP protocol version")
            session = (_header(headers, "Mcp-Session-Id"), MCP_PROTOCOL_VERSION)
            self._notify("notifications/initialized", request_id, session)
            self._session_id, self._negotiated_version = session
            return result

    def list_tools(self, request_id: str | None = None) -> tuple[dict[str, Any], ...]:
        result = self._request("tools/list", {}, request_id)
        tools = result.get("tools")
        if not isinstance(tools, list) or not all(self._valid_tool(item) for item in tools):
            raise McpGatewayProtocolError("invalid tools/list result")
        return tuple(tools)

    def discover(self, request_id: str | None = None) -> tuple[dict[str, Any], ...]:
        """Refresh the in-memory catalog after a successful initialize handshake.

        A catalog younger than ``catalog_ttl_seconds`` is returned without a round trip.
        """
        with self._catalog_lock:
            if self._catalog_fresh():
                return self._catalog
            self.initialize(request_id)
            self._catalog = self.list_tools(request_id)
            self._catalog_loaded_at = time.monotonic()
            return self._catalog

    def invalidate_catalog(self) -> None:
        self._catalog_loaded_at = None

    def _catalog_fresh(self) -> bool:
        if self._catalog_ttl_seconds <= 0 or self._catalog_loaded_at is None:
            return False
        ttl = self._catalog_ttl_seconds
        if self._expected_tools - {tool["name"] for tool in self._catalog}:
            ttl = min(ttl, self._incomplete_catalog_ttl_seconds)
        return time.monotonic() - self._catalog_loaded_at < ttl

    def model_tools(self, allowed_names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Convert the current catalog to OpenAI function tools.

        ``allowed_names`` keeps state-changing upstream tools away from the model.
        MCP input schemas contain optional properties and ``additionalProperties``
        values that OpenAI strict mode rejects, so tools are sent non-strict and
        every call is still checked against the Gateway schema in ``call_tool``.
        """
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
                "strict": False,
            }
            for tool in self._catalog
            if allowed_names is None or tool["name"] in allowed_names
        ]

    def call_tool(self, name: str, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        self._check_call(name, arguments)
        return _tool_outcome(self._request("tools/call", {"name": name, "arguments": arguments}, request_id))

    def call_tools(
        self, calls: Sequence[tuple[str, dict[str, Any]]], request_id: str | None = None,
    ) -> list[dict[str, Any] | McpGatewayError]:
        """Gateway 일괄 호출(``/v1/tools/batch-call``)로 여러 도구를 부르고 입력 순서대로 결과(또는 오류)를 돌려준다.

        catalog·스키마 검사에 실패한 호출은 프로세스 밖으로 나가지 않는다. 나머지는 Gateway가 동시에 중계한다
        (동시 실행 수는 Gateway의 ``GATEWAY_BATCH_MAX_CONCURRENCY``가 정한다).
        """
        results: list[dict[str, Any] | McpGatewayError | None] = [None] * len(calls)
        pending: list[int] = []
        for index, (name, arguments) in enumerate(calls):
            try:
                self._check_call(name, arguments)
            except McpGatewayError as exc:
                results[index] = exc
            else:
                pending.append(index)
        if pending:
            outcomes = self._batches([calls[index] for index in pending], request_id)
            for index, outcome in zip(pending, outcomes):
                results[index] = outcome
        return results  # type: ignore[return-value]

    def _check_call(self, name: str, arguments: Any) -> None:
        tool = next((item for item in self._catalog if item["name"] == name), None)
        if tool is None:
            raise McpGatewayProtocolError("tool is not in the current Gateway catalog")
        if not isinstance(arguments, dict) or not Draft202012Validator(tool["inputSchema"]).is_valid(arguments):
            raise McpGatewayProtocolError("tool arguments do not match the Gateway input schema")

    def _batches(self, calls: Sequence[tuple[str, dict[str, Any]]], request_id: str | None) -> list[dict[str, Any] | McpGatewayError]:
        """Gateway 상한(``batch_max_calls``)씩 나눠 일괄 호출한다. 나눈 묶음은 동시에 보낸다."""
        size = self._batch_max_calls
        groups = [calls[start:start + size] for start in range(0, len(calls), size)]
        if len(groups) == 1:
            return self._batch(groups[0], request_id)
        return [outcome for result in self._pool.map(lambda group: self._batch(group, request_id), groups) for outcome in result]

    def _batch(self, calls: Sequence[tuple[str, dict[str, Any]]], request_id: str | None) -> list[dict[str, Any] | McpGatewayError]:
        """묶음 하나를 POST한다. 요청 단위 실패(연결·HTTP·형식)는 모든 호출의 오류가 된다."""
        trace_id = request_id or str(uuid.uuid4())
        body = json.dumps({"calls": [{"name": name, "arguments": arguments} for name, arguments in calls]}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "X-Request-ID": trace_id}
        try:
            status, response_headers, raw = self._transport(self._batch_url, body, headers, self._timeout_seconds)
        except OSError:
            self.invalidate_catalog()
            error = McpGatewayConnectionError("Gateway connection failed")
            return [error for _ in calls]
        try:
            if status < 200 or status >= 300:
                raise McpGatewayProtocolError(f"Gateway returned HTTP {status}")
            if _header(response_headers, "X-Request-ID") not in (None, trace_id):
                raise McpGatewayProtocolError("Gateway returned a mismatched request ID")
            try:
                items = json.loads(raw.decode("utf-8")).get("results")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
                raise McpGatewayProtocolError("Gateway returned invalid JSON") from exc
            if not isinstance(items, list) or len(items) != len(calls):
                raise McpGatewayProtocolError("Gateway batch result does not match the request")
        except McpGatewayProtocolError as error:
            return [error for _ in calls]
        outcomes: list[dict[str, Any] | McpGatewayError] = []
        for item in items:
            try:
                if not isinstance(item, dict):
                    raise McpGatewayProtocolError("Gateway result must be an object")
                outcomes.append(_tool_outcome(item))
            except McpGatewayError as error:
                outcomes.append(error)
        return outcomes

    def _request(self, method: str, params: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        """세션 안에서 요청한다. 세션이 만료됐으면 다시 초기화하고 한 번 재시도한다."""
        if self._negotiated_version is None:
            self.initialize(request_id)
        try:
            return self._exchange(method, params, request_id)[0]
        except McpSessionExpired:
            self.initialize(request_id)
            return self._exchange(method, params, request_id)[0]

    @staticmethod
    def _headers(trace_id: str, session_id: str | None, version: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": MCP_ACCEPT, "X-Request-ID": trace_id}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        if version:
            headers["MCP-Protocol-Version"] = version
        return headers

    def _post(
        self, body: dict[str, Any], trace_id: str, method: str, session: tuple[str | None, str | None] | None = None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        """``session``이 없으면 현재 세션(ID·프로토콜 버전)을 싣는다."""
        sent_session, version = session if session is not None else (self._session_id, self._negotiated_version)
        try:
            status, response_headers, raw = self._transport(
                self._url, json.dumps(body).encode("utf-8"), self._headers(trace_id, sent_session, version), self._timeout_seconds,
            )
        except OSError as exc:
            if method == "tools/call":
                # The Gateway may have restarted with a different catalog.
                self.invalidate_catalog()
            raise McpGatewayConnectionError("Gateway connection failed") from exc
        if status == 404 and sent_session:
            raise McpSessionExpired("MCP session expired")
        if status < 200 or status >= 300:
            raise McpGatewayProtocolError(f"Gateway returned HTTP {status}")
        if _header(response_headers, "X-Request-ID") not in (None, trace_id):
            raise McpGatewayProtocolError("Gateway returned a mismatched request ID")
        return status, response_headers, raw

    def _exchange(
        self, method: str, params: dict[str, Any], request_id: str | None, session: tuple[str | None, str | None] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        """JSON-RPC 요청 하나를 보내고 (result, 응답 헤더)를 돌려준다. 응답은 JSON 또는 SSE다."""
        trace_id = request_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        _status, response_headers, raw = self._post({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}, trace_id, method, session)
        response = _decode_response(raw, _header(response_headers, "Content-Type"), message_id)
        if not isinstance(response, dict) or response.get("jsonrpc") != "2.0" or response.get("id") != message_id:
            raise McpGatewayProtocolError("Gateway returned an invalid JSON-RPC response")
        if "error" in response:
            raise McpGatewayProtocolError("Gateway returned a JSON-RPC error")
        if not isinstance(response.get("result"), dict):
            raise McpGatewayProtocolError("Gateway result must be an object")
        return response["result"], response_headers

    def _notify(self, method: str, request_id: str | None, session: tuple[str | None, str | None] | None = None) -> None:
        """JSON-RPC 알림이다. 서버는 202(본문 없음)로 답하므로 본문을 해석하지 않는다."""
        self._post({"jsonrpc": "2.0", "method": method}, request_id or str(uuid.uuid4()), method, session)

    @staticmethod
    def _valid_tool(tool: Any) -> bool:
        return isinstance(tool, dict) and all(isinstance(tool.get(key), str) for key in ("name", "description")) and isinstance(tool.get("inputSchema"), dict)

    def _http_post(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, Mapping[str, str], bytes]:
        """POST over the shared connection pool; transport failures surface as ``OSError``/``TimeoutError``."""
        try:
            response = self._http.post(url, content=body, headers=dict(headers), timeout=timeout)
        except httpx2.TimeoutException as exc:
            raise TimeoutError("Gateway request timed out") from exc
        except httpx2.TransportError as exc:
            raise OSError("Gateway connection failed") from exc
        return response.status_code, response.headers, response.content


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """대소문자를 구분하지 않고 헤더 값을 읽는다(httpx 헤더와 일반 dict 모두)."""
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    return next((item for key, item in headers.items() if key.lower() == lowered), None)


def _decode_response(raw: bytes, content_type: str | None, message_id: str) -> Any:
    """JSON 본문 또는 SSE 스트림에서 ``message_id``에 대한 JSON-RPC 응답을 꺼낸다.

    SSE에는 응답 전에 서버 알림(진행 상황 등)이 섞일 수 있으므로, id가 맞는 메시지를 고른다.
    """
    try:
        text = raw.decode("utf-8")
        if not (content_type or "").lower().startswith("text/event-stream"):
            return json.loads(text)
        for event in text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip() for line in event.split("\n") if line.startswith("data:"))
            if not data:
                continue
            message = json.loads(data)
            if isinstance(message, dict) and message.get("id") == message_id:
                return message
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise McpGatewayProtocolError("Gateway returned invalid JSON") from exc
    raise McpGatewayProtocolError("Gateway SSE stream had no response for the request")


def _batch_url(mcp_url: str) -> str:
    """``http://gateway:8080/mcp`` → ``http://gateway:8080/v1/tools/batch-call`` (keeping any path prefix)."""
    parts = urlsplit(mcp_url)
    prefix = parts.path.removesuffix("/").removesuffix("/mcp")
    return urlunsplit((parts.scheme, parts.netloc, prefix + BATCH_CALL_PATH, "", ""))


def _tool_outcome(result: dict[str, Any]) -> dict[str, Any]:
    """Return a successful tool result or raise its ``isError`` envelope as ``McpGatewayToolError``."""
    if result.get("isError") is True:
        code, retryable = _tool_error_details(result)
        raise McpGatewayToolError(code, retryable)
    return result


def _tool_error_details(result: dict[str, Any]) -> tuple[str, bool]:
    """Read the Gateway's compact error envelope without retaining its message text."""
    code = result.get("code") or result.get("errorCode")
    retryable = result.get("retryable")
    if not code:
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                text = item.get("text") if isinstance(item, dict) else None
                if not isinstance(text, str):
                    continue
                try:
                    embedded = json.loads(text).get("error", {})
                except json.JSONDecodeError:
                    continue
                if isinstance(embedded, dict):
                    code = embedded.get("code")
                    retryable = embedded.get("retryable", retryable)
                    break
    return str(code or "tool_error"), bool(retryable)
