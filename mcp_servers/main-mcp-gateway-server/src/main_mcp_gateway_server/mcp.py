"""MCP JSON-RPC 호출, 서비스 catalog, 도구 중계의 결정적 로직을 구현한다."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import http.client
import json
import logging
from threading import Lock
from time import monotonic
from typing import Any, Protocol
from urllib.parse import urlsplit

from .config import UpstreamConfig


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "2025-06-18"
GATEWAY_CLIENT_INFO = {"name": "mcp-gateway", "version": "0.1.0"}
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_IDLE_CONNECTIONS = 16
DEFAULT_BATCH_MAX_CONCURRENCY = 8
# 연결되지 않은 서비스로 온 호출이 그 서비스를 다시 discovery하는 최소 간격(초)이다.
REDISCOVERY_INTERVAL_SECONDS = 5.0


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


class _ConnectionPool:
    """upstream host별 keep-alive 연결을 스레드 간에 공유한다.

    Gateway는 요청마다 새 handler 스레드를 쓰므로 스레드 로컬이 아닌 공용 유휴 연결 목록을 둔다.
    ``http.client`` 연결은 스레드 안전하지 않으므로 한 번에 한 스레드만 빌려 쓴다.
    """

    def __init__(self, max_idle: int = DEFAULT_MAX_IDLE_CONNECTIONS) -> None:
        self._idle: dict[tuple[str, str, int | None], list[http.client.HTTPConnection]] = {}
        self._max_idle = max_idle
        self._lock = Lock()

    def acquire(self, key: tuple[str, str, int | None], timeout: float) -> tuple[http.client.HTTPConnection, bool]:
        """유휴 연결을 꺼내거나 새로 만든다. 두 번째 값은 재사용 여부다."""
        with self._lock:
            idle = self._idle.get(key)
            connection = idle.pop() if idle else None
        if connection is not None:
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            return connection, True
        scheme, host, port = key
        factory = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        return factory(host, port, timeout=timeout), False

    def release(self, key: tuple[str, str, int | None], connection: http.client.HTTPConnection) -> None:
        with self._lock:
            idle = self._idle.setdefault(key, [])
            if len(idle) < self._max_idle:
                idle.append(connection)
                return
        connection.close()

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, {}
        for connections in idle.values():
            for connection in connections:
                connection.close()


class HttpTransport:
    """표준 라이브러리만 사용해 Streamable HTTP MCP 서버에 요청한다.

    upstream 연결은 keep-alive로 재사용해 호출마다 TCP 연결을 새로 맺지 않는다.
    """

    def __init__(self, timeout: float = DEFAULT_UPSTREAM_TIMEOUT_SECONDS, max_idle_connections: int = DEFAULT_MAX_IDLE_CONNECTIONS) -> None:
        """upstream MCP·health 요청에 공통으로 적용할 timeout(초)을 설정한다."""
        self._timeout = timeout
        self._pool = _ConnectionPool(max_idle_connections)

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
        status, response_headers, raw = self._send("POST", url, json.dumps(payload).encode("utf-8"), headers)
        if status >= 400:
            raise OSError(f"upstream MCP가 HTTP {status}를 반환했습니다")
        try:
            parsed = _decode_mcp_response(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OSError("upstream MCP 응답 형식이 올바르지 않습니다") from error
        return McpResponse(parsed, response_headers.get("mcp-session-id"))

    def send_notification(self, url: str, payload: dict[str, Any], session_id: str | None = None) -> None:
        """세션을 유지해 MCP notification을 전송한다.

        Streamable HTTP 서버는 notification에 빈 202 응답을 반환할 수 있으므로 본문을 JSON으로 해석하지 않는다.
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        status, _, _ = self._send("POST", url, json.dumps(payload).encode("utf-8"), headers)
        if status >= 400:
            raise OSError("upstream MCP notification 요청에 실패했습니다")

    def healthy(self, url: str) -> bool:
        """비민감 health endpoint가 HTTP 200을 반환하는지 확인한다."""
        try:
            status, _, _ = self._send("GET", url, None, {})
        except OSError:
            return False
        return status == 200

    def close(self) -> None:
        """유휴 upstream 연결을 모두 닫는다."""
        self._pool.close()

    def _send(self, method: str, url: str, body: bytes | None, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        """풀의 연결로 요청을 보내고 (상태, 소문자 헤더, 본문)을 반환한다.

        재사용한 유휴 연결을 upstream이 이미 닫았다면 새 연결로 한 번만 다시 보낸다. 시간 초과는
        호출자의 대기 시간을 두 배로 늘리므로 재시도하지 않는다. 통신 실패는 ``OSError``다.
        """
        parts = urlsplit(url)
        key = (parts.scheme, parts.hostname or "", parts.port)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        for _ in range(2):
            connection, reused = self._pool.acquire(key, self._timeout)
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read()
            except TimeoutError as error:
                connection.close()
                raise OSError("upstream 요청 시간이 초과되었습니다") from error
            except (OSError, http.client.HTTPException) as error:
                connection.close()
                if reused and isinstance(error, (http.client.RemoteDisconnected, http.client.CannotSendRequest, ConnectionResetError, BrokenPipeError)):
                    continue
                raise OSError("upstream 연결에 실패했습니다") from error
            if response.will_close:
                connection.close()
            else:
                self._pool.release(key, connection)
            return response.status, {name.lower(): value for name, value in response.getheaders()}, raw
        raise OSError("upstream 연결에 실패했습니다")


@dataclass
class ServiceState:
    """서비스별 discovery 결과와 현재 health 상태를 보관한다."""

    config: UpstreamConfig
    status: str = "inactive"
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_id: str | None = None
    # 이 시각(monotonic)까지는 최근 health 확인 결과를 재사용한다.
    healthy_until: float = 0.0
    # 마지막 discovery 시각(monotonic)이다. 0이면 아직 시도하지 않았다.
    discovered_at: float = 0.0


class GatewayCatalog:
    """upstream discovery 결과를 단일 MCP 도구 이름 공간으로 합친다.

    도구 호출은 catalog 잠금 밖에서 upstream을 호출하므로 서로 다른 요청의 호출이 동시에 진행된다.
    discovery는 서비스별로 새 ``ServiceState``를 만든 뒤 교체하므로, 진행 중인 호출은 교체 전 상태(세션)를
    끝까지 사용하고 discovery 중에도 호출이 막히지 않는다.
    """

    def __init__(
        self,
        upstreams: tuple[UpstreamConfig, ...],
        transport: Transport | None = None,
        health_ttl_seconds: float = 0.0,
        batch_max_concurrency: int = DEFAULT_BATCH_MAX_CONCURRENCY,
    ) -> None:
        self._transport = transport or HttpTransport()
        self._services = {item.service_id: ServiceState(item) for item in upstreams}
        # _services 사전 교체·조회만 보호한다. upstream 요청 중에는 잡지 않는다.
        self._lock = Lock()
        # 동시에 들어온 tools/list가 discovery를 중복 실행하지 않게 한다.
        self._discovery_lock = Lock()
        self._health_ttl_seconds = health_ttl_seconds
        self._batch_executor = ThreadPoolExecutor(max_workers=max(batch_max_concurrency, 1), thread_name_prefix="gateway-batch")

    def discover_all(self) -> None:
        """각 upstream을 독립적으로 health 검사하고 MCP session 및 tools/list를 검증한다.

        `2025-06-18` protocol 합의, initialized 알림, name/description/inputSchema 도구 계약이 모두
        성립한 서비스만 active catalog에 넣는다. 하나가 실패해도 다른 서비스 상태에는 영향을 주지 않는다.
        """
        with self._discovery_lock:
            for state in self._snapshot().values():
                discovered = self._discover(state.config)
                with self._lock:
                    self._services[state.config.service_id] = discovered

    def _discover(self, config: UpstreamConfig) -> ServiceState:
        state = ServiceState(config, discovered_at=monotonic())
        if not self._transport.healthy(config.health_url):
            state.status = "unhealthy"
            LOGGER.warning("gateway_discovery_failed service_id=%s stage=health", config.service_id)
            return state
        try:
            initialize = self._transport.request_json(config.mcp_url, _request("initialize", _initialize_params()))
            if initialize.body.get("result", {}).get("protocolVersion") != PROTOCOL_VERSION:
                raise OSError("upstream MCP protocol version이 일치하지 않습니다")
            session_id = initialize.session_id
            self._transport.send_notification(config.mcp_url, _notification("notifications/initialized"), session_id)
            response = self._transport.request_json(config.mcp_url, _request("tools/list", {}), session_id=session_id)
            tools = response.body.get("result", {}).get("tools", [])
            if not isinstance(tools, list):
                raise OSError("tools/list 형식이 올바르지 않습니다")
            valid = {tool["name"]: tool for tool in tools if _is_valid_tool(tool)}
            if len(valid) != len(tools):
                raise OSError("tools/list 도구 계약이 올바르지 않습니다")
        except (OSError, KeyError, TypeError) as error:
            state.status = "unhealthy"
            LOGGER.warning("gateway_discovery_failed service_id=%s stage=mcp_handshake reason=%s", config.service_id, error)
            return state
        return replace(state, status="active", tools=valid, session_id=session_id, healthy_until=monotonic() + self._health_ttl_seconds)

    def _snapshot(self) -> dict[str, ServiceState]:
        with self._lock:
            return dict(self._services)

    def list_tools(self) -> list[dict[str, Any]]:
        """현재 active 서비스 도구를 `<service_id>__<name>` 형태로 반환한다."""
        exposed: list[dict[str, Any]] = []
        for service_id, state in self._snapshot().items():
            if state.status != "active":
                continue
            for name, tool in state.tools.items():
                exposed.append({**tool, "name": f"{service_id}__{name}"})
        return exposed

    def route(self, external_name: str) -> tuple[ServiceState, str] | None:
        """외부 도구 이름을 정확한 service 상태와 upstream 도구명으로 역해석한다."""
        for service_id, state in self._snapshot().items():
            prefix = f"{service_id}__"
            if external_name.startswith(prefix):
                return state, external_name.removeprefix(prefix)
        return None

    def call(self, external_name: str, arguments: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        """등록된 도구 호출을 한 upstream에만 전달하고 MCP result 또는 오류를 만든다.

        알 수 없는 이름은 upstream 요청 없이 not_found가 되며, health 실패는 catalog를 유지한 채
        upstream_unavailable을 반환한다.
        """
        started_at = monotonic()
        target = self.route(external_name)
        if target is None:
            return self._finish_call(None, external_name, request_id, started_at, _tool_error("not_found", "등록되지 않은 Gateway 도구입니다", False))
        state, upstream_name = target
        if upstream_name not in state.tools and state.status != "active":
            # Gateway보다 늦게 준비됐거나 복구된 upstream은 tools/list를 기다리지 않고 여기서 다시 찾는다.
            state = self._rediscover_if_stale(state)
        if upstream_name not in state.tools:
            return self._finish_call(state, upstream_name, request_id, started_at, _tool_error("not_found", "등록되지 않은 Gateway 도구입니다", False))
        if not self._healthy(state):
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
            state.healthy_until = 0.0
            return self._finish_call(state, upstream_name, request_id, started_at, _tool_error("upstream_unavailable", "upstream MCP 호출에 실패했습니다", True))
        result = response.body.get("result")
        if not _is_valid_tool_result(result):
            result = _tool_error("internal_error", "upstream MCP 도구 응답 계약이 올바르지 않습니다", True)
        return self._finish_call(state, upstream_name, request_id, started_at, result)

    def call_many(self, calls: Sequence[Any], request_id: str | None = None) -> list[dict[str, Any]]:
        """여러 도구 호출을 동시에 upstream으로 보내고 입력 순서대로 MCP result를 반환한다.

        각 항목은 ``{"name": str, "arguments": object}``이며, 형식이 틀린 항목은 upstream 없이
        ``invalid_params`` 오류 result가 된다. 한 호출의 실패는 다른 호출에 영향을 주지 않는다.
        """
        def run(item: Any) -> dict[str, Any]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("arguments", {}), dict):
                return _tool_error("invalid_params", "batch 항목에는 name 문자열과 arguments 객체가 필요합니다", False)
            return self.call(item["name"], item.get("arguments", {}), request_id)
        if len(calls) <= 1:
            return [run(item) for item in calls]
        return list(self._batch_executor.map(run, calls))

    def _rediscover_if_stale(self, state: ServiceState) -> ServiceState:
        """연결되지 않은 서비스 하나를 ``REDISCOVERY_INTERVAL_SECONDS``에 한 번까지만 다시 discovery한다."""
        if monotonic() - state.discovered_at < REDISCOVERY_INTERVAL_SECONDS:
            return state
        with self._discovery_lock:
            # 기다리는 동안 다른 스레드가 이미 다시 찾았을 수 있다.
            current = self._snapshot()[state.config.service_id]
            if monotonic() - current.discovered_at < REDISCOVERY_INTERVAL_SECONDS:
                return current
            discovered = self._discover(current.config)
            with self._lock:
                self._services[current.config.service_id] = discovered
            return discovered

    def _healthy(self, state: ServiceState) -> bool:
        """health 확인 결과를 ``health_ttl_seconds`` 동안 재사용해 호출마다 왕복하지 않는다."""
        if monotonic() < state.healthy_until:
            return True
        if not self._transport.healthy(state.config.health_url):
            return False
        state.healthy_until = monotonic() + self._health_ttl_seconds
        return True

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
