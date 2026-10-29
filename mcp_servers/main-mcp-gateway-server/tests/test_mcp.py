"""Gateway catalog의 discovery, 이름 변환, 장애 격리를 검증한다."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from main_mcp_gateway_server.config import UpstreamConfig, configured_upstreams
from main_mcp_gateway_server.mcp import GatewayCatalog, HttpTransport, McpResponse, PROTOCOL_VERSION


class FakeTransport:
    """외부 네트워크 없이 catalog 계약을 검증하는 결정적 transport다."""

    def __init__(self) -> None:
        self.health = {"http://a/healthz": True, "http://b/healthz": True}
        self.calls: list[tuple[str, dict, str | None, str | None]] = []
        self.protocol_version = PROTOCOL_VERSION
        self.tools = [{"name": "lookup", "description": "lookup tool", "inputSchema": {"type": "object"}}]
        self.tool_result: dict = {"structuredContent": {"source": "fake"}, "isError": False}

        self.health_checks = 0
        # tools/call마다 호출되는 선택 hook이다(동시성 검증용).
        self.on_tool_call = None

    def healthy(self, url: str) -> bool:
        self.health_checks += 1
        return self.health[url]

    def request_json(self, url: str, payload: dict, request_id: str | None = None, session_id: str | None = None) -> McpResponse:
        self.calls.append((url, payload, request_id, session_id))
        if payload["method"] == "tools/call" and self.on_tool_call:
            self.on_tool_call(payload)
        if payload["method"] == "initialize":
            return McpResponse({"result": {"protocolVersion": self.protocol_version}}, "session-from-initialize")
        if payload["method"] == "tools/list":
            return McpResponse({"result": {"tools": self.tools}})
        if payload["method"] == "tools/call":
            return McpResponse({"result": self.tool_result})
        return McpResponse({"result": {}})

    def send_notification(self, url: str, payload: dict, session_id: str | None = None) -> None:
        self.calls.append((url, payload, None, session_id))


class GatewayCatalogTest(unittest.TestCase):
    """두 service catalog가 서로 독립적이며 올바르게 route되는지 확인한다."""

    def setUp(self) -> None:
        self.transport = FakeTransport()
        self.catalog = GatewayCatalog((UpstreamConfig("a", "http://a/mcp", "http://a/healthz"), UpstreamConfig("b", "http://b/mcp", "http://b/healthz")), self.transport)
        self.catalog.discover_all()

    def test_lists_namespaced_tools(self) -> None:
        """active upstream 도구는 service ID 접두어와 함께 노출된다."""
        self.assertEqual({tool["name"] for tool in self.catalog.list_tools()}, {"a__lookup", "b__lookup"})

    def test_discovery_negotiates_protocol_and_sends_initialized_notification(self) -> None:
        """discovery는 완전한 MCP initialize metadata와 initialized 알림을 upstream에 보낸다."""
        initialize = [payload for _, payload, _, _ in self.transport.calls if payload["method"] == "initialize"]
        notifications = [payload for _, payload, _, _ in self.transport.calls if payload["method"] == "notifications/initialized"]
        self.assertEqual([item["params"]["protocolVersion"] for item in initialize], [PROTOCOL_VERSION, PROTOCOL_VERSION])
        self.assertTrue(all(item["params"]["capabilities"] == {} for item in initialize))
        self.assertTrue(all(item["params"]["clientInfo"] == {"name": "mcp-gateway", "version": "0.1.0"} for item in initialize))
        self.assertEqual(len(notifications), 2)
        self.assertTrue(all("id" not in item for item in notifications))

    def test_discovery_and_calls_reuse_mcp_session(self) -> None:
        """실제 Streamable HTTP 서버가 요구하는 session ID를 후속 요청에 유지한다."""
        notification = [item for item in self.transport.calls if item[1]["method"] == "notifications/initialized"][-1]
        listed = [item for item in self.transport.calls if item[1]["method"] == "tools/list"][-1]
        self.assertEqual(notification[3], "session-from-initialize")
        self.assertEqual(listed[3], "session-from-initialize")

        self.catalog.call("a__lookup", {})
        tool_call = [item for item in self.transport.calls if item[1]["method"] == "tools/call"][-1]
        self.assertEqual(tool_call[3], "session-from-initialize")

    def test_invalid_protocol_or_tool_contract_is_not_registered(self) -> None:
        """버전 불일치와 description 없는 도구는 해당 서비스만 unhealthy로 만든다."""
        self.transport.protocol_version = "2024-11-05"
        self.catalog.discover_all()
        self.assertEqual(self.catalog.list_tools(), [])
        self.transport.protocol_version = PROTOCOL_VERSION
        self.transport.tools = [{"name": "lookup", "inputSchema": {"type": "object"}}]
        self.catalog.discover_all()
        self.assertEqual(self.catalog.list_tools(), [])

    def test_unknown_tool_does_not_call_upstream(self) -> None:
        """알 수 없는 도구는 upstream 호출 없이 not_found MCP tool error가 된다."""
        before = len(self.transport.calls)
        result = self.catalog.call("missing__tool", {})
        self.assertEqual(len(self.transport.calls), before)
        self.assertEqual(json.loads(result["content"][0]["text"])["error"]["code"], "not_found")

    def test_one_unhealthy_service_does_not_block_other_service(self) -> None:
        """한 서비스 health 실패는 해당 route에만 upstream_unavailable을 반환한다."""
        self.transport.health["http://a/healthz"] = False
        failed = self.catalog.call("a__lookup", {})
        successful = self.catalog.call("b__lookup", {})
        self.assertTrue(failed["isError"])
        self.assertFalse(successful.get("isError", False))

    def test_call_forwards_request_id_and_normalizes_invalid_result(self) -> None:
        """도구 호출 request ID를 upstream으로 전달하고 계약 위반 result는 안전하게 바꾼다."""
        successful = self.catalog.call("a__lookup", {}, "request-123")
        tool_call = [item for item in self.transport.calls if item[1]["method"] == "tools/call"][-1]
        self.assertEqual(tool_call[2], "request-123")
        self.assertEqual(successful["structuredContent"], {"source": "fake"})

        self.transport.tool_result = {"content": [{"type": "text", "text": "unsafe"}]}
        failed = self.catalog.call("a__lookup", {})
        self.assertTrue(failed["isError"])
        self.assertEqual(json.loads(failed["content"][0]["text"])["error"]["code"], "internal_error")


class GatewayConcurrencyTest(unittest.TestCase):
    """도구 호출이 catalog 잠금에 줄 서지 않고, health 확인과 일괄 호출이 계약대로 동작하는지 확인한다."""

    def _catalog(self, transport: FakeTransport, **options) -> GatewayCatalog:
        catalog = GatewayCatalog((UpstreamConfig("a", "http://a/mcp", "http://a/healthz"), UpstreamConfig("b", "http://b/mcp", "http://b/healthz")), transport, **options)
        catalog.discover_all()
        return catalog

    def test_tool_calls_run_concurrently(self) -> None:
        """두 호출이 동시에 upstream에 머물 수 있어야 한다(직렬화되면 barrier가 깨진다)."""
        transport = FakeTransport()
        catalog = self._catalog(transport)
        barrier = threading.Barrier(2, timeout=2)
        transport.on_tool_call = lambda _: barrier.wait()
        results: list[dict] = []
        threads = [threading.Thread(target=lambda: results.append(catalog.call("a__lookup", {}))) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.get("isError") is False for result in results))

    def test_discovery_does_not_block_calls(self) -> None:
        """discovery가 upstream을 기다리는 동안에도 기존 catalog로 도구를 호출할 수 있다."""
        transport = FakeTransport()
        catalog = self._catalog(transport)
        entered, release = threading.Event(), threading.Event()
        original = transport.healthy

        def slow_health(url: str) -> bool:
            entered.set()
            release.wait(2)
            return original(url)

        transport.healthy = slow_health
        discovery = threading.Thread(target=catalog.discover_all)
        discovery.start()
        self.assertTrue(entered.wait(2))
        transport.healthy = original
        result = catalog.call("a__lookup", {})
        release.set()
        discovery.join()
        self.assertFalse(result["isError"])

    def test_health_result_is_cached_until_failure(self) -> None:
        """health TTL 안에서는 호출마다 health를 확인하지 않고, upstream 호출이 실패하면 캐시를 버린다."""
        transport = FakeTransport()
        catalog = self._catalog(transport, health_ttl_seconds=60)
        checks = transport.health_checks
        catalog.call("a__lookup", {})
        catalog.call("a__lookup", {})
        self.assertEqual(transport.health_checks, checks)

        def fail(_: dict) -> None:
            raise OSError("down")

        transport.on_tool_call = fail
        self.assertTrue(catalog.call("a__lookup", {})["isError"])
        transport.on_tool_call = None
        catalog.call("a__lookup", {})
        self.assertEqual(transport.health_checks, checks + 1)

    def test_call_many_keeps_order_and_isolates_invalid_items(self) -> None:
        """일괄 호출은 입력 순서를 유지하고 형식이 틀린 항목만 invalid_params로 만든다."""
        transport = FakeTransport()
        catalog = self._catalog(transport)
        results = catalog.call_many(
            [{"name": "a__lookup", "arguments": {}}, {"name": 3}, {"name": "missing__tool", "arguments": {}}, {"name": "b__lookup"}],
            "batch-request",
        )
        codes = [result["structuredContent"].get("error", {}).get("code", "ok") for result in results]
        self.assertEqual(codes, ["ok", "invalid_params", "not_found", "ok"])
        forwarded = [item[2] for item in transport.calls if item[1]["method"] == "tools/call"]
        self.assertEqual(forwarded, ["batch-request", "batch-request"])


class _KeepAliveUpstream(BaseHTTPRequestHandler):
    """연결 수를 세는 HTTP/1.1 upstream이다."""

    protocol_version = "HTTP/1.1"
    connections = 0
    status = 200
    body = b""
    content_type = "application/json"

    def setup(self) -> None:
        type(self).connections += 1
        super().setup()

    def do_GET(self) -> None:
        self._reply(200, b"ok", "text/plain")

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._reply(self.status, self.body, self.content_type)

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Mcp-Session-Id", "s-1")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class HttpTransportTest(unittest.TestCase):
    """upstream keep-alive 재사용과 응답 해석을 실제 HTTP로 검증한다."""

    def setUp(self) -> None:
        handler = type("Upstream", (_KeepAliveUpstream,), {"connections": 0, "status": 200, "body": json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode()})
        self.handler = handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.transport = HttpTransport(timeout=2)

    def tearDown(self) -> None:
        self.transport.close()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def test_reuses_connection_and_reads_session(self) -> None:
        for _ in range(3):
            response = self.transport.request_json(f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
            self.assertEqual(response.body["result"], {"ok": True})
            self.assertEqual(response.session_id, "s-1")
        self.assertTrue(self.transport.healthy(f"{self.url}/healthz"))
        self.assertEqual(self.handler.connections, 1)

    def test_decodes_sse_response(self) -> None:
        self.handler.content_type = "text/event-stream"
        self.handler.body = b'event: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": {"sse": true}}\n\n'
        response = self.transport.request_json(f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
        self.assertEqual(response.body["result"], {"sse": True})

    def test_http_error_and_unreachable_upstream_raise_os_error(self) -> None:
        self.handler.status = 404
        with self.assertRaises(OSError):
            self.transport.request_json(f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
        self.assertFalse(HttpTransport(timeout=0.5).healthy("http://127.0.0.1:9/healthz"))


class UpstreamConfigTest(unittest.TestCase):
    """외부 YAML 등록 파일이 Gateway 시작 전에 검증되는지 확인한다."""

    def test_loads_multiple_upstreams_from_yaml(self) -> None:
        """YAML의 서비스 목록은 코드 변경 없이 Gateway 등록 정보가 된다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upstreams.yaml"
            path.write_text(
                "upstreams:\n"
                "  - service_id: added-server\n"
                "    mcp_url: http://added-server:8000/mcp\n"
                "    health_url: http://added-server:8000/healthz\n",
                encoding="utf-8",
            )
            previous = os.environ.get("UPSTREAM_CONFIG_PATH")
            os.environ["UPSTREAM_CONFIG_PATH"] = str(path)
            try:
                self.assertEqual(configured_upstreams()[0].service_id, "added-server")
            finally:
                if previous is None:
                    os.environ.pop("UPSTREAM_CONFIG_PATH", None)
                else:
                    os.environ["UPSTREAM_CONFIG_PATH"] = previous

    def test_rejects_duplicate_service_ids(self) -> None:
        """중복 service_id는 이름 공간 충돌을 막기 위해 Gateway 시작을 실패시킨다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upstreams.yaml"
            path.write_text(
                "upstreams:\n"
                "  - {service_id: same, mcp_url: http://one/mcp, health_url: http://one/healthz}\n"
                "  - {service_id: same, mcp_url: http://two/mcp, health_url: http://two/healthz}\n",
                encoding="utf-8",
            )
            previous = os.environ.get("UPSTREAM_CONFIG_PATH")
            os.environ["UPSTREAM_CONFIG_PATH"] = str(path)
            try:
                with self.assertRaises(ValueError):
                    configured_upstreams()
            finally:
                if previous is None:
                    os.environ.pop("UPSTREAM_CONFIG_PATH", None)
                else:
                    os.environ["UPSTREAM_CONFIG_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
