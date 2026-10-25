"""Minimal HTTP JSON-RPC client for the read-only MCP Gateway."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MCP_PROTOCOL_VERSION = "2025-06-18"
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
    """Call only the configured Gateway endpoint and cache its tool catalog in memory."""

    def __init__(
        self,
        url: str | None = None,
        timeout_seconds: float | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        # Imported here to keep this transport module usable without app startup.
        from n_note_ai_service_layers.settings import mcp_gateway_timeout_seconds, mcp_gateway_url
        self._url = url or mcp_gateway_url()
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else mcp_gateway_timeout_seconds()
        self._transport = transport or self._http_post
        self._catalog: tuple[dict[str, Any], ...] = ()

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
        """Refresh the in-memory catalog after a successful initialize handshake."""
        self.initialize(request_id)
        self._catalog = self.list_tools(request_id)
        return self._catalog

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
        tool = next((item for item in self._catalog if item["name"] == name), None)
        if tool is None:
            raise McpGatewayProtocolError("tool is not in the current Gateway catalog")
        if not isinstance(arguments, dict) or not _matches_schema(arguments, tool["inputSchema"]):
            raise McpGatewayProtocolError("tool arguments do not match the Gateway input schema")
        result = self._request("tools/call", {"name": name, "arguments": arguments}, request_id)
        if result.get("isError") is True:
            code, retryable = _tool_error_details(result)
            raise McpGatewayToolError(code, retryable)
        return result

    def _request(self, method: str, params: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        trace_id = request_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        body = json.dumps({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "X-Request-ID": trace_id}
        try:
            status, response_headers, raw = self._transport(self._url, body, headers, self._timeout_seconds)
        except (OSError, URLError, TimeoutError) as exc:
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

    @staticmethod
    def _http_post(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, Mapping[str, str], bytes]:
        request = Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.status, dict(response.headers.items()), response.read()
        except HTTPError as exc:
            return exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read()


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
