"""Minimal HTTP JSON-RPC client for the read-only MCP Gateway."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2
from jsonschema import Draft202012Validator

from n_note_ai_service_layers import settings



MCP_PROTOCOL_VERSION = "2025-06-18"
# A catalog missing expected tools (an upstream MCP still starting or down) is refreshed this often.
INCOMPLETE_CATALOG_TTL_SECONDS = 15.0
# Gateway의 일괄 도구 호출 경로다. MCP endpoint와 같은 host·접두 경로를 쓴다.
BATCH_CALL_PATH = "/v1/tools/batch-call"
HttpTransport = Callable[[str, bytes, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]


class McpGatewayError(RuntimeError):
    """Base error for an MCP Gateway call; never includes tool arguments."""


class McpGatewayConnectionError(McpGatewayError):
    """The Gateway could not be reached before it returned JSON-RPC."""


class McpGatewayProtocolError(McpGatewayError):
    """The Gateway returned malformed, HTTP-error, or JSON-RPC-error output."""


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
    ) -> None:
        config = settings.get()
        self._url = url or config.mcp_gateway_url
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else config.mcp_gateway_timeout_seconds
        # httpx2 client는 스레드 간에 공유하는 keep-alive 연결 풀이다. Gateway는 내부망이라 프록시 환경 변수를 따르지 않는다.
        self._http = httpx2.Client(trust_env=False) if transport is None else None
        self._transport = transport or self._http_post
        self._catalog: tuple[dict[str, Any], ...] = ()
        self._catalog_ttl_seconds = catalog_ttl_seconds
        self._expected_tools = frozenset(expected_tools)
        self._incomplete_catalog_ttl_seconds = incomplete_catalog_ttl_seconds
        self._catalog_loaded_at: float | None = None
        self._catalog_lock = threading.Lock()
        self._batch_url = batch_url or _batch_url(self._url)
        # An older Gateway answers 404 for the batch path; fall back to one tools/call per item.
        self._batch_supported = True

    @property
    def catalog(self) -> tuple[dict[str, Any], ...]:
        """Return the last Gateway-provided catalog without inventing tool definitions."""
        return self._catalog

    def initialize(self, request_id: str | None = None) -> dict[str, Any]:
        result = self._request("initialize", {"protocolVersion": MCP_PROTOCOL_VERSION}, request_id)
        if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise McpGatewayProtocolError("unsupported MCP protocol version")
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
        self, calls: Sequence[tuple[str, dict[str, Any]]], request_id: str | None = None, max_concurrency: int = 1,
    ) -> list[dict[str, Any] | McpGatewayError]:
        """Call several tools in one Gateway round trip and return results (or errors) in input order.

        Calls that fail local catalog/schema checks never leave the process. The Gateway relays the
        rest concurrently. Without batch support it falls back to ``call_tool`` with up to
        ``max_concurrency`` parallel requests.
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
            selected = [calls[index] for index in pending]
            outcomes = self._batch(selected, request_id) if self._batch_supported else None
            if outcomes is None:
                outcomes = self._each(selected, request_id, max_concurrency)
            for index, outcome in zip(pending, outcomes):
                results[index] = outcome
        return results  # type: ignore[return-value]

    def _check_call(self, name: str, arguments: Any) -> None:
        tool = next((item for item in self._catalog if item["name"] == name), None)
        if tool is None:
            raise McpGatewayProtocolError("tool is not in the current Gateway catalog")
        if not isinstance(arguments, dict) or not Draft202012Validator(tool["inputSchema"]).is_valid(arguments):
            raise McpGatewayProtocolError("tool arguments do not match the Gateway input schema")

    def _batch(self, calls: Sequence[tuple[str, dict[str, Any]]], request_id: str | None) -> list[dict[str, Any] | McpGatewayError] | None:
        """POST the batch; ``None`` means this Gateway has no batch endpoint."""
        trace_id = request_id or str(uuid.uuid4())
        body = json.dumps({"calls": [{"name": name, "arguments": arguments} for name, arguments in calls]}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "X-Request-ID": trace_id}
        try:
            status, response_headers, raw = self._transport(self._batch_url, body, headers, self._timeout_seconds)
        except OSError:
            self.invalidate_catalog()
            error = McpGatewayConnectionError("Gateway connection failed")
            return [error for _ in calls]
        if status == 404:
            self._batch_supported = False
            return None
        try:
            if status < 200 or status >= 300:
                raise McpGatewayProtocolError(f"Gateway returned HTTP {status}")
            if response_headers.get("X-Request-ID") not in (None, trace_id):
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

    def _each(self, calls: Sequence[tuple[str, dict[str, Any]]], request_id: str | None, max_concurrency: int) -> list[dict[str, Any] | McpGatewayError]:
        def call(item: tuple[str, dict[str, Any]]) -> dict[str, Any] | McpGatewayError:
            try:
                return self.call_tool(item[0], item[1], request_id)
            except McpGatewayError as error:
                return error
        if len(calls) <= 1 or max_concurrency <= 1:
            return [call(item) for item in calls]
        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(calls)), thread_name_prefix="gateway-call") as pool:
            return list(pool.map(call, calls))

    def _request(self, method: str, params: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        trace_id = request_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        body = json.dumps({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "X-Request-ID": trace_id}
        try:
            status, response_headers, raw = self._transport(self._url, body, headers, self._timeout_seconds)
        except OSError as exc:
            if method == "tools/call":
                # The Gateway may have restarted with a different catalog.
                self.invalidate_catalog()
            raise McpGatewayConnectionError("Gateway connection failed") from exc
        if status < 200 or status >= 300:
            raise McpGatewayProtocolError(f"Gateway returned HTTP {status}")
        if response_headers.get("X-Request-ID") not in (None, trace_id):
            raise McpGatewayProtocolError("Gateway returned a mismatched request ID")
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise McpGatewayProtocolError("Gateway returned invalid JSON") from exc
        if not isinstance(response, dict) or response.get("jsonrpc") != "2.0" or response.get("id") != message_id:
            raise McpGatewayProtocolError("Gateway returned an invalid JSON-RPC response")
        if "error" in response:
            raise McpGatewayProtocolError("Gateway returned a JSON-RPC error")
        if not isinstance(response.get("result"), dict):
            raise McpGatewayProtocolError("Gateway result must be an object")
        return response["result"]

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
