"""MCP JSON-RPC 호출, 서비스 catalog, 도구 중계의 결정적 로직을 구현한다."""

from dataclasses import dataclass, field
import json
import logging
from threading import RLock
from time import monotonic
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import UpstreamConfig


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "2025-06-18"
GATEWAY_CLIENT_INFO = {"name": "mcp-gateway", "version": "0.1.0"}
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 5.0


class Transport(Protocol):
    """HTTP 구현을 교체할 수 있도록 MCP 및 health 요청을 추상화한다."""

    def request_json(
        self, url: str, payload: dict[str, Any], request_id: str | None = None, session_id: str | None = None
    ) -> "McpResponse": ...

    def send_notification(self, url: str, payload: dict[str, Any], session_id: str | None = None) -> None: ...

    def healthy(self, url: str) -> bool: ...


@dataclass(frozen=True)
class McpResponse:
    """JSON-RPC 본문과 Streamable HTTP가 할당한 선택적 MCP 세션 ID다."""

    body: dict[str, Any]
    session_id: str | None = None


class HttpTransport:
    """표준 라이브러리만 사용해 Streamable HTTP MCP 서버에 요청한다."""

    def __init__(self, timeout: float = DEFAULT_UPSTREAM_TIMEOUT_SECONDS) -> None:
        """upstream MCP·health 요청에 공통으로 적용할 timeout(초)을 설정한다."""
        self._timeout = timeout

    def request_json(
        self, url: str, payload: dict[str, Any], request_id: str | None = None, session_id: str | None = None
    ) -> McpResponse:
        """JSON-RPC payload를 POST하고 JSON object 응답을 반환한다.

        url은 upstream의 `/mcp` 주소여야 하며, 통신 또는 JSON 형식 오류는 OSError로 바꾼다.
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if request_id:
            headers["X-Request-ID"] = request_id
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                parsed = _decode_mcp_response(response.read().decode("utf-8"))
                response_session_id = response.headers.get("Mcp-Session-Id")
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise OSError("upstream MCP 요청에 실패했습니다") from error
        if not isinstance(parsed, dict):
            raise OSError("upstream MCP 응답이 객체가 아닙니다")
        return McpResponse(parsed, response_session_id)

    def send_notification(self, url: str, payload: dict[str, Any], session_id: str | None = None) -> None:
        """세션을 유지해 MCP notification을 전송한다.

        Streamable HTTP 서버는 notification에 빈 202 응답을 반환할 수 있으므로 본문을 JSON으로 해석하지 않는다.
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self._timeout):
                pass
        except (URLError, TimeoutError) as error:
            raise OSError("upstream MCP notification 요청에 실패했습니다") from error

    def healthy(self, url: str) -> bool:
        """비민감 health endpoint가 HTTP 200을 반환하는지 확인한다."""
        try:
            with urlopen(Request(url, method="GET"), timeout=self._timeout) as response:
                return response.status == 200
        except (URLError, TimeoutError):
            return False


@dataclass
class ServiceState:
    """서비스별 discovery 결과와 현재 health 상태를 보관한다."""

    config: UpstreamConfig
    status: str = "inactive"
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_id: str | None = None


class GatewayCatalog:
    """upstream discovery 결과를 단일 MCP 도구 이름 공간으로 합친다."""

    def __init__(self, upstreams: tuple[UpstreamConfig, ...], transport: Transport | None = None) -> None:
        self._transport = transport or HttpTransport()
        self._services = {item.service_id: ServiceState(item) for item in upstreams}
        self._lock = RLock()

    def discover_all(self) -> None:
        """각 upstream을 독립적으로 health 검사하고 MCP session 및 tools/list를 검증한다.

        `2025-06-18` protocol 합의, initialized 알림, name/description/inputSchema 도구 계약이 모두
        성립한 서비스만 active catalog에 넣는다. 하나가 실패해도 다른 서비스 상태에는 영향을 주지 않는다.
        """
        with self._lock:
            for state in self._services.values():
                state.tools.clear()
                state.session_id = None
                if not self._transport.healthy(state.config.health_url):
                    state.status = "unhealthy"
                    LOGGER.warning("gateway_discovery_failed service_id=%s stage=health", state.config.service_id)
                    continue
                try:
                    initialize = self._transport.request_json(
                        state.config.mcp_url,
                        _request("initialize", _initialize_params()),
                    )
                    if initialize.body.get("result", {}).get("protocolVersion") != PROTOCOL_VERSION:
                        raise OSError("upstream MCP protocol version이 일치하지 않습니다")
                    state.session_id = initialize.session_id
                    self._transport.send_notification(state.config.mcp_url, _notification("notifications/initialized"), state.session_id)
                    response = self._transport.request_json(state.config.mcp_url, _request("tools/list", {}), session_id=state.session_id)
                    tools = response.body.get("result", {}).get("tools", [])
                    if not isinstance(tools, list):
                        raise OSError("tools/list 형식이 올바르지 않습니다")
                    state.tools = {tool["name"]: tool for tool in tools if _is_valid_tool(tool)}
                    if len(state.tools) != len(tools):
                        raise OSError("tools/list 도구 계약이 올바르지 않습니다")
                    state.status = "active"
                except (OSError, KeyError, TypeError) as error:
                    state.session_id = None
                    state.status = "unhealthy"
                    LOGGER.warning(
                        "gateway_discovery_failed service_id=%s stage=mcp_handshake reason=%s",
                        state.config.service_id,
                        error,
                    )

    def list_tools(self) -> list[dict[str, Any]]:
        """현재 active 서비스 도구를 `<service_id>__<name>` 형태로 반환한다."""
        with self._lock:
            exposed: list[dict[str, Any]] = []
            for service_id, state in self._services.items():
                if state.status != "active":
                    continue
                for name, tool in state.tools.items():
                    exposed.append({**tool, "name": f"{service_id}__{name}"})
            return exposed

    def route(self, external_name: str) -> tuple[ServiceState, str] | None:
        """외부 도구 이름을 정확한 service 상태와 upstream 도구명으로 역해석한다."""
        for service_id, state in self._services.items():
            prefix = f"{service_id}__"
            if external_name.startswith(prefix):
                return state, external_name.removeprefix(prefix)
        return None

    def call(self, external_name: str, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """등록된 도구 호출을 한 upstream에만 전달하고 MCP result 또는 오류를 만든다.

        알 수 없는 이름은 upstream 요청 없이 not_found가 되며, health 실패는 catalog를 유지한 채
        upstream_unavailable을 반환한다.
        """
        # tools/list의 재-discovery가 진행 중일 때 세션 ID가 바뀌지 않게 한다.
        with self._lock:
            return self._call(external_name, arguments, request_id)

    def _call(self, external_name: str, arguments: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        """catalog lock을 보유한 상태에서 하나의 upstream 도구 호출을 실행한다."""
        started_at = monotonic()
        target = self.route(external_name)
        if target is None:
            return self._finish_call(None, external_name, request_id, started_at, _tool_error("not_found", "등록되지 않은 Gateway 도구입니다", False))
        state, upstream_name = target
        if upstream_name not in state.tools:
            return self._finish_call(state, upstream_name, request_id, started_at, _tool_error("not_found", "등록되지 않은 Gateway 도구입니다", False))
        if not self._transport.healthy(state.config.health_url):
            state.status = "unhealthy"
            return self._finish_call(state, upstream_name, request_id, started_at, _tool_error("upstream_unavailable", "upstream 서비스가 준비되지 않았습니다", True))
        try:
            response = self._transport.request_json(
                state.config.mcp_url,
                _request("tools/call", {"name": upstream_name, "arguments": arguments}),
                request_id,
                state.session_id,
            )
        except OSError:
            state.status = "unhealthy"
            return self._finish_call(state, upstream_name, request_id, started_at, _tool_error("upstream_unavailable", "upstream MCP 호출에 실패했습니다", True))
        result = response.body.get("result")
        if not _is_valid_tool_result(result):
            result = _tool_error("internal_error", "upstream MCP 도구 응답 계약이 올바르지 않습니다", True)
        return self._finish_call(state, upstream_name, request_id, started_at, result)

    def _finish_call(self, state: ServiceState | None, tool_name: str, request_id: str | None, started_at: float, result: dict[str, Any]) -> dict[str, Any]:
        """도구 호출 결과를 민감 정보 없이 감사 로그에 기록하고 그대로 반환한다."""
        LOGGER.info(
            "gateway_tool service_id=%s tool_name=%s request_id=%s result_code=%s duration_ms=%d",
            state.config.service_id if state else "unknown",
            tool_name,
            request_id or "generated-by-client-unavailable",
            _result_code(result),
            (monotonic() - started_at) * 1000,
        )
        return result


def _request(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": "gateway-discovery", "method": method, "params": params}


def _initialize_params() -> dict[str, Any]:
    """MCP Streamable HTTP upstream과 호환되는 initialize 요청 인수를 생성한다."""
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": GATEWAY_CLIENT_INFO,
    }


def _notification(method: str) -> dict[str, Any]:
    """응답 ID 없이 upstream MCP notification payload를 생성한다."""
    return {"jsonrpc": "2.0", "method": method, "params": {}}


def _decode_mcp_response(raw: str) -> dict[str, Any]:
    """JSON 응답 또는 한 개의 SSE JSON-RPC event에서 객체를 추출한다."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        data_lines = [line.removeprefix("data:").lstrip() for line in raw.splitlines() if line.startswith("data:")]
        if not data_lines:
            raise
        parsed = json.loads("\n".join(data_lines))
    if not isinstance(parsed, dict):
        raise OSError("upstream MCP 응답이 객체가 아닙니다")
    return parsed


def _is_valid_tool(tool: Any) -> bool:
    """Gateway catalog에 등록 가능한 최소 MCP tool schema인지 판별한다."""
    return (
        isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
        and bool(tool["name"])
        and isinstance(tool.get("description"), str)
        and isinstance(tool.get("inputSchema"), dict)
    )


def _is_valid_tool_result(result: Any) -> bool:
    """성공 또는 예상 실패 MCP tool result가 Gateway 안전 계약을 만족하는지 판별한다."""
    if not isinstance(result, dict):
        return False
    if result.get("isError") is False:
        return "structuredContent" in result
    if result.get("isError") is not True:
        return False
    error = _tool_result_error(result)
    return isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str) and isinstance(error.get("retryable"), bool)


def _tool_result_error(result: dict[str, Any]) -> dict[str, Any] | None:
    """구조화 필드 또는 MCP text content에서 안전한 오류 object를 추출한다."""
    structured_content = result.get("structuredContent")
    if isinstance(structured_content, dict) and isinstance(structured_content.get("error"), dict):
        return structured_content["error"]
    content = result.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return None
    text = content[0].get("text")
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed.get("error") if isinstance(parsed, dict) else None


def _result_code(result: dict[str, Any]) -> str:
    """감사 로그에 남길 성공 또는 안전한 오류 코드를 반환한다."""
    error = _tool_result_error(result)
    return error.get("code", "ok") if isinstance(error, dict) else "ok"


def _tool_error(code: str, message: str, retryable: bool) -> dict[str, Any]:
    error = {"code": code, "message": message, "retryable": retryable}
    return {
        "content": [{"type": "text", "text": json.dumps({"error": error}, ensure_ascii=False)}],
        "structuredContent": {"error": error},
        "isError": True,
    }
