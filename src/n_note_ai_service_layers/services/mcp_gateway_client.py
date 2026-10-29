"""Minimal HTTP JSON-RPC client for the read-only MCP Gateway."""

from __future__ import annotations

import http.client
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit, urlunsplit


MCP_PROTOCOL_VERSION = "2025-06-18"
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
    """

    def __init__(
        self,
        url: str | None = None,
        timeout_seconds: float | None = None,
        transport: HttpTransport | None = None,
        catalog_ttl_seconds: float = 0.0,
        batch_url: str | None = None,
    ) -> None:
        # Imported here to keep this transport module usable without app startup.
        from n_note_ai_service_layers.settings import mcp_gateway_timeout_seconds, mcp_gateway_url
        self._url = url or mcp_gateway_url()
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else mcp_gateway_timeout_seconds()
        self._transport = transport or self._http_post
        self._catalog: tuple[dict[str, Any], ...] = ()
        self._catalog_ttl_seconds = catalog_ttl_seconds
        self._catalog_loaded_at: float | None = None
        self._catalog_lock = threading.Lock()
        # Keep-alive connection per worker thread (http.client connections are not thread-safe).
        self._connections = threading.local()
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
        return (
            self._catalog_ttl_seconds > 0 and self._catalog_loaded_at is not None
            and time.monotonic() - self._catalog_loaded_at < self._catalog_ttl_seconds
        )

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
        if not isinstance(arguments, dict) or not _matches_schema(arguments, tool["inputSchema"]):
            raise McpGatewayProtocolError("tool arguments do not match the Gateway input schema")

    def _batch(self, calls: Sequence[tuple[str, dict[str, Any]]], request_id: str | None) -> list[dict[str, Any] | McpGatewayError] | None:
        """POST the batch; ``None`` means this Gateway has no batch endpoint."""
        trace_id = request_id or str(uuid.uuid4())
        body = json.dumps({"calls": [{"name": name, "arguments": arguments} for name, arguments in calls]}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "X-Request-ID": trace_id}
        try:
            status, response_headers, raw = self._transport(self._batch_url, body, headers, self._timeout_seconds)
        except (OSError, TimeoutError, http.client.HTTPException):
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
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
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
        """POST over this thread's keep-alive connection, reconnecting once if the Gateway closed it.

        Gateway tools are read-only, so resending after a dropped idle connection is safe.
        """
        parts = urlsplit(url)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        for attempt in range(2):
            try:
                connection = self._connection(parts.scheme, parts.hostname or "", parts.port, timeout)
                connection.request("POST", path, body=body, headers=dict(headers))
                response = connection.getresponse()
                payload = response.read()
            except TimeoutError:
                # A slow Gateway is not retried: that would double the caller's wait.
                self._drop_connection()
                raise
            except (OSError, http.client.RemoteDisconnected, http.client.CannotSendRequest):
                self._drop_connection()
                if attempt:
                    raise
                continue
            except Exception:
                self._drop_connection()
                raise
            if response.will_close:
                self._drop_connection()
            return response.status, dict(response.getheaders()), payload
        raise McpGatewayConnectionError("Gateway connection failed")

    def _connection(self, scheme: str, host: str, port: int | None, timeout: float) -> http.client.HTTPConnection:
        connection = getattr(self._connections, "connection", None)
        if connection is None:
            factory = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            connection = factory(host, port, timeout=timeout)
            self._connections.connection = connection
        elif connection.sock is not None:
            connection.sock.settimeout(timeout)
        return connection

    def _drop_connection(self) -> None:
        connection = getattr(self._connections, "connection", None)
        if connection is not None:
            connection.close()
            self._connections.connection = None


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


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Small deterministic JSON Schema subset used by Gateway tool input schemas."""
    if "anyOf" in schema:
        return isinstance(schema["anyOf"], list) and any(_matches_schema(value, item) for item in schema["anyOf"] if isinstance(item, dict))
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool), "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool), "null": value is None,
    }
    if isinstance(expected, str) and not type_matches.get(expected, True):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list) or any(key not in value for key in required):
            return False
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            return False
        return all(not isinstance(properties.get(key), dict) or _matches_schema(item, properties[key]) for key, item in value.items())
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_matches_schema(item, schema["items"]) for item in value)
    return True


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
