"""Gateway catalog의 discovery, 이름 변환, 장애 격리를 검증한다."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from time import monotonic
import unittest
from unittest.mock import patch

import httpx2
import uvicorn

from demo_upstream import create_demo_upstream_app
from main_mcp_gateway_server.config import UpstreamConfig, configured_upstreams
from main_mcp_gateway_server.mcp import PROTOCOL_VERSION, GatewayCatalog, McpSessionExpired, check_health, post_json, upstream_client


class FakeUpstream:
    """외부 네트워크 없이 catalog 계약을 검증하는 결정적 upstream이다. ``httpx2.MockTransport``의 비동기 handler로 쓴다."""

    def __init__(self) -> None:
        self.health = {"http://a/healthz": True, "http://b/healthz": True}
        self.calls: list[tuple[str, dict, str | None, str | None]] = []
        self.protocol_version = PROTOCOL_VERSION
        self.tools = [{"name": "lookup", "description": "lookup tool", "inputSchema": {"type": "object"}}]
        self.tool_result: dict = {"structuredContent": {"source": "fake"}, "isError": False}
        self.next_session_id = lambda: "session-from-initialize"
        self.health_checks = 0
        # health 확인·tools/call마다 기다리는 선택 hook이다(동시성 검증용). tools/call hook이 응답을 주면 그대로 반환한다.
        self.on_health = None
        self.on_tool_call = None

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if request.method == "GET":
            self.health_checks += 1
            if self.on_health:
                await self.on_health(url)
            return httpx2.Response(200 if self.health[url] else 503)
        payload = json.loads(request.content)
        session_id = request.headers.get("mcp-session-id")
        self.calls.append((url, payload, request.headers.get("x-request-id"), session_id))
        method = payload["method"]
        headers: dict[str, str] = {}
        if method == "notifications/initialized":
            return httpx2.Response(202)
        if method == "initialize":
            result: dict = {"protocolVersion": self.protocol_version}
            headers["mcp-session-id"] = self.next_session_id()
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            if self.on_tool_call:
                response = await self.on_tool_call(payload, session_id)
                if response is not None:
                    return response
            result = self.tool_result
        else:
            result = {}
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result}, headers=headers)


UPSTREAMS = (UpstreamConfig("a", "http://a/mcp", "http://a/healthz"), UpstreamConfig("b", "http://b/mcp", "http://b/healthz"))


class CatalogTestCase(unittest.IsolatedAsyncioTestCase):
    async def _catalog(self, upstream: FakeUpstream, discover: bool = True, **options) -> GatewayCatalog:
        catalog = GatewayCatalog(UPSTREAMS, upstream.client(), **options)
        self.addAsyncCleanup(catalog.aclose)
        if discover:
            await catalog.discover_all()
        return catalog


class GatewayCatalogTest(CatalogTestCase):
    """두 service catalog가 서로 독립적이며 올바르게 route되는지 확인한다."""

    async def asyncSetUp(self) -> None:
        self.upstream = FakeUpstream()
        self.catalog = await self._catalog(self.upstream)

    def test_lists_namespaced_tools(self) -> None:
        """active upstream 도구는 service ID 접두어와 함께 노출된다."""
        self.assertEqual({tool["name"] for tool in self.catalog.list_tools()}, {"a__lookup", "b__lookup"})

    def test_discovery_negotiates_protocol_and_sends_initialized_notification(self) -> None:
        """discovery는 완전한 MCP initialize metadata와 initialized 알림을 upstream에 보낸다."""
        initialize = [payload for _, payload, _, _ in self.upstream.calls if payload["method"] == "initialize"]
        notifications = [payload for _, payload, _, _ in self.upstream.calls if payload["method"] == "notifications/initialized"]
        self.assertEqual([item["params"]["protocolVersion"] for item in initialize], [PROTOCOL_VERSION, PROTOCOL_VERSION])
        self.assertTrue(all(item["params"]["capabilities"] == {} for item in initialize))
        self.assertTrue(all(item["params"]["clientInfo"] == {"name": "mcp-gateway", "version": "0.1.0"} for item in initialize))
        self.assertEqual(len(notifications), 2)
        self.assertTrue(all("id" not in item for item in notifications))

    async def test_discovery_and_calls_reuse_mcp_session(self) -> None:
        """실제 Streamable HTTP 서버가 요구하는 session ID를 후속 요청에 유지한다."""
        notification = [item for item in self.upstream.calls if item[1]["method"] == "notifications/initialized"][-1]
        listed = [item for item in self.upstream.calls if item[1]["method"] == "tools/list"][-1]
        self.assertEqual(notification[3], "session-from-initialize")
        self.assertEqual(listed[3], "session-from-initialize")

        await self.catalog.call("a__lookup", {})
        tool_call = [item for item in self.upstream.calls if item[1]["method"] == "tools/call"][-1]
        self.assertEqual(tool_call[3], "session-from-initialize")

    async def test_expired_upstream_session_is_reinitialized_and_the_call_retried_once(self) -> None:
        """upstream이 재시작해 세션을 잊으면(404) 그 서비스만 다시 초기화하고 같은 호출을 한 번 다시 보낸다."""
        sessions = iter(["session-2", "session-3"])
        self.upstream.next_session_id = lambda: next(sessions)

        async def restarted(_payload: dict, session_id: str | None) -> httpx2.Response | None:
            return httpx2.Response(404) if session_id == "session-from-initialize" else None

        self.upstream.on_tool_call = restarted
        result = await self.catalog.call("a__lookup", {}, "request-1")
        self.assertFalse(result["isError"])
        tool_calls = [item for item in self.upstream.calls if item[1]["method"] == "tools/call"]
        self.assertEqual([item[3] for item in tool_calls], ["session-from-initialize", "session-2"])
        self.assertTrue(all(item[2] == "request-1" for item in tool_calls))
        # b는 다시 초기화하지 않는다.
        self.assertEqual(sum(1 for item in self.upstream.calls if item[1]["method"] == "initialize" and item[0] == "http://b/mcp"), 1)

    async def test_invalid_protocol_or_tool_contract_is_not_registered(self) -> None:
        """버전 불일치와 description 없는 도구는 해당 서비스만 unhealthy로 만든다."""
        self.upstream.protocol_version = "2024-11-05"
        await self.catalog.discover_all()
        self.assertEqual(self.catalog.list_tools(), [])
        self.upstream.protocol_version = PROTOCOL_VERSION
        self.upstream.tools = [{"name": "lookup", "inputSchema": {"type": "object"}}]
        await self.catalog.discover_all()
        self.assertEqual(self.catalog.list_tools(), [])

    async def test_unknown_tool_does_not_call_upstream(self) -> None:
        """알 수 없는 도구는 upstream 호출 없이 not_found MCP tool error가 된다."""
        before = len(self.upstream.calls)
        result = await self.catalog.call("missing__tool", {})
        self.assertEqual(len(self.upstream.calls), before)
        self.assertEqual(json.loads(result["content"][0]["text"])["error"]["code"], "not_found")

    async def test_one_unhealthy_service_does_not_block_other_service(self) -> None:
        """한 서비스 health 실패는 해당 route에만 upstream_unavailable을 반환한다."""
        self.upstream.health["http://a/healthz"] = False
        failed = await self.catalog.call("a__lookup", {})
        successful = await self.catalog.call("b__lookup", {})
        self.assertTrue(failed["isError"])
        self.assertFalse(successful.get("isError", False))

    async def test_call_forwards_request_id_and_normalizes_invalid_result(self) -> None:
        """도구 호출 request ID를 upstream으로 전달하고 계약 위반 result는 안전하게 바꾼다."""
        successful = await self.catalog.call("a__lookup", {}, "request-123")
        tool_call = [item for item in self.upstream.calls if item[1]["method"] == "tools/call"][-1]
        self.assertEqual(tool_call[2], "request-123")
        self.assertEqual(successful["structuredContent"], {"source": "fake"})

        self.upstream.tool_result = {"content": [{"type": "text", "text": "unsafe"}]}
        failed = await self.catalog.call("a__lookup", {})
        self.assertTrue(failed["isError"])
        self.assertEqual(json.loads(failed["content"][0]["text"])["error"]["code"], "internal_error")


class GatewayConcurrencyTest(CatalogTestCase):
    """도구 호출이 discovery 잠금에 줄 서지 않고, health 확인과 일괄 호출이 계약대로 동작하는지 확인한다."""

    async def test_tool_calls_run_concurrently(self) -> None:
        """두 호출이 동시에 upstream에 머물 수 있어야 한다(직렬화되면 barrier가 깨진다)."""
        upstream = FakeUpstream()
        catalog = await self._catalog(upstream)
        barrier = asyncio.Barrier(2)

        async def meet(_payload: dict, _session_id: str | None) -> None:
            await asyncio.wait_for(barrier.wait(), 2)

        upstream.on_tool_call = meet
        results = await asyncio.gather(catalog.call("a__lookup", {}), catalog.call("a__lookup", {}))
        self.assertTrue(all(result.get("isError") is False for result in results))

    async def test_discovery_contacts_upstreams_in_parallel(self) -> None:
        """두 upstream의 health 확인이 동시에 진행돼야 한다(직렬이면 barrier가 깨진다)."""
        upstream = FakeUpstream()
        barrier = asyncio.Barrier(2)

        async def meet(_url: str) -> None:
            await asyncio.wait_for(barrier.wait(), 2)

        upstream.on_health = meet
        catalog = await self._catalog(upstream)
        self.assertEqual(sorted(tool["name"] for tool in catalog.list_tools()), ["a__lookup", "b__lookup"])

    async def test_batch_concurrency_is_bounded(self) -> None:
        """일괄 호출은 동시에 진행하되 ``batch_max_concurrency``를 넘지 않는다."""
        upstream = FakeUpstream()
        catalog = await self._catalog(upstream, batch_max_concurrency=2)
        active, peak = [0], [0]

        async def track(_payload: dict, _session_id: str | None) -> None:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1

        upstream.on_tool_call = track
        results = await catalog.call_many([{"name": "a__lookup", "arguments": {}}] * 5)
        self.assertTrue(all(result["isError"] is False for result in results))
        self.assertEqual(peak[0], 2)

    async def test_refresh_reuses_active_catalog_until_the_ttl(self) -> None:
        """tools/list(refresh)는 TTL 안에서 upstream에 다시 묻지 않고, 연결되지 않은 서비스만 짧은 간격으로 다시 찾는다."""
        upstream = FakeUpstream()
        upstream.health["http://b/healthz"] = False
        now = [0.0]
        with patch("main_mcp_gateway_server.mcp.monotonic", side_effect=lambda: now[0]):
            catalog = await self._catalog(upstream, catalog_ttl_seconds=30)
            lists = lambda: sum(1 for item in upstream.calls if item[1]["method"] == "tools/list")
            self.assertEqual(lists(), 1)  # a만 active
            now[0] = 2.0
            await catalog.refresh()
            self.assertEqual((lists(), upstream.health_checks), (1, 2))  # 둘 다 아직 새 discovery 아님
            upstream.health["http://b/healthz"] = True
            now[0] = 6.0
            await catalog.refresh()  # b(연결 안 됨)는 5초가 지나 다시 찾고, a는 TTL 안이라 그대로다
            self.assertEqual(lists(), 2)
            self.assertEqual(sorted(tool["name"] for tool in catalog.list_tools()), ["a__lookup", "b__lookup"])
            now[0] = 31.0
            await catalog.refresh()  # a의 TTL이 지났다
            self.assertEqual(lists(), 3)

    async def test_discovery_does_not_block_calls(self) -> None:
        """discovery가 upstream을 기다리는 동안에도 기존 catalog로 도구를 호출할 수 있다."""
        upstream = FakeUpstream()
        catalog = await self._catalog(upstream)
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_health(_url: str) -> None:
            entered.set()
            await asyncio.wait_for(release.wait(), 2)

        upstream.on_health = slow_health
        discovery = asyncio.create_task(catalog.discover_all())
        await asyncio.wait_for(entered.wait(), 2)
        upstream.on_health = None
        result = await asyncio.wait_for(catalog.call("a__lookup", {}), 2)
        release.set()
        await discovery
        self.assertFalse(result["isError"])

    async def test_health_result_is_cached_until_failure(self) -> None:
        """health TTL 안에서는 호출마다 health를 확인하지 않고, upstream 호출이 실패하면 캐시를 버린다."""
        upstream = FakeUpstream()
        catalog = await self._catalog(upstream, health_ttl_seconds=60)
        checks = upstream.health_checks
        await catalog.call("a__lookup", {})
        await catalog.call("a__lookup", {})
        self.assertEqual(upstream.health_checks, checks)

        async def fail(_payload: dict, _session_id: str | None) -> httpx2.Response:
            return httpx2.Response(500)

        upstream.on_tool_call = fail
        self.assertTrue((await catalog.call("a__lookup", {}))["isError"])
        upstream.on_tool_call = None
        await catalog.call("a__lookup", {})
        self.assertEqual(upstream.health_checks, checks + 1)

    async def test_call_rediscovers_a_service_that_was_not_ready_at_startup(self) -> None:
        """Gateway보다 늦게 뜬 upstream은 tools/list 없이 다음 호출에서 연결된다."""
        upstream = FakeUpstream()
        upstream.health["http://a/healthz"] = False
        catalog = await self._catalog(upstream)
        self.assertEqual({tool["name"] for tool in catalog.list_tools()}, {"b__lookup"})
        upstream.health["http://a/healthz"] = True
        with patch("main_mcp_gateway_server.mcp.monotonic", return_value=monotonic() + 10):
            result = await catalog.call("a__lookup", {})
        self.assertFalse(result["isError"])
        self.assertEqual({tool["name"] for tool in catalog.list_tools()}, {"a__lookup", "b__lookup"})

    async def test_rediscovery_is_rate_limited(self) -> None:
        """연결되지 않은 서비스로 호출이 몰려도 간격 안에서는 다시 discovery하지 않는다."""
        upstream = FakeUpstream()
        upstream.health["http://a/healthz"] = False
        catalog = await self._catalog(upstream)
        checks = upstream.health_checks
        for _ in range(3):
            self.assertTrue((await catalog.call("a__lookup", {}))["isError"])
        self.assertEqual(upstream.health_checks, checks)

    async def test_call_many_keeps_order_and_isolates_invalid_items(self) -> None:
        """일괄 호출은 입력 순서를 유지하고 형식이 틀린 항목만 invalid_params로 만든다."""
        upstream = FakeUpstream()
        catalog = await self._catalog(upstream)
        results = await catalog.call_many(
            [{"name": "a__lookup", "arguments": {}}, {"name": 3}, {"name": "missing__tool", "arguments": {}}, {"name": "b__lookup"}],
            "batch-request",
        )
        codes = [result["structuredContent"].get("error", {}).get("code", "ok") for result in results]
        self.assertEqual(codes, ["ok", "invalid_params", "not_found", "ok"])
        forwarded = [item[2] for item in upstream.calls if item[1]["method"] == "tools/call"]
        self.assertEqual(forwarded, ["batch-request", "batch-request"])
        # 같은 세션의 동시 요청이 응답을 서로 가로채지 않도록 JSON-RPC id는 요청마다 다르다(MCP 요구 사항).
        ids = [item[1]["id"] for item in upstream.calls if "id" in item[1]]
        self.assertEqual(len(ids), len(set(ids)))


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


class UpstreamHttpTest(unittest.IsolatedAsyncioTestCase):
    """upstream keep-alive 재사용과 응답 해석을 실제 HTTP로 검증한다."""

    def setUp(self) -> None:
        handler = type("Upstream", (_KeepAliveUpstream,), {"connections": 0, "status": 200, "body": json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode()})
        self.handler = handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    async def asyncSetUp(self) -> None:
        self.client = upstream_client(timeout=2)

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    async def test_reuses_connection_and_reads_session(self) -> None:
        for _ in range(3):
            body, session_id = await post_json(self.client, f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
            self.assertEqual(body["result"], {"ok": True})
            self.assertEqual(session_id, "s-1")
        self.assertTrue(await check_health(self.client, f"{self.url}/healthz"))
        self.assertEqual(self.handler.connections, 1)

    async def test_decodes_sse_response(self) -> None:
        self.handler.content_type = "text/event-stream"
        self.handler.body = b'event: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": {"sse": true}}\n\n'
        body, _ = await post_json(self.client, f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
        self.assertEqual(body["result"], {"sse": True})

    async def test_sse_stream_with_notifications_picks_the_matching_response(self) -> None:
        self.handler.content_type = "text/event-stream"
        self.handler.body = (
            b'event: message\ndata: {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}\n\n'
            b'event: message\ndata: {"jsonrpc": "2.0", "id": 7, "result": {"sse": true}}\n\n'
        )
        body, _ = await post_json(self.client, f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 7, "method": "tools/call"})
        self.assertEqual(body["result"], {"sse": True})

    async def test_response_for_another_request_id_is_rejected(self) -> None:
        with self.assertRaises(OSError):
            await post_json(self.client, f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 99, "method": "tools/call"})

    async def test_http_error_and_unreachable_upstream_raise_os_error(self) -> None:
        self.handler.status = 404
        with self.assertRaises(OSError):
            await post_json(self.client, f"{self.url}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
        async with upstream_client(timeout=0.5) as client:
            self.assertFalse(await check_health(client, "http://127.0.0.1:9/healthz"))


class DemoUpstreamEndToEndTest(unittest.IsolatedAsyncioTestCase):
    """세션·SSE를 쓰는 SDK upstream을 실제 HTTP로 discovery하고 호출한다."""

    def setUp(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(create_demo_upstream_app("korean"), host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)

    def tearDown(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)

    async def test_discovers_and_calls_session_upstream(self) -> None:
        base = f"http://127.0.0.1:{self.port}"
        catalog = GatewayCatalog((UpstreamConfig("korean", f"{base}/mcp", f"{base}/healthz"),), timeout=2)
        try:
            await catalog.discover_all()
            self.assertEqual([tool["name"] for tool in catalog.list_tools()], ["korean__review_standard_korean_text"])
            results = await catalog.call_many([{"name": "korean__review_standard_korean_text", "arguments": {}}] * 2, "e2e")
        finally:
            await catalog.aclose()
        self.assertEqual([result["structuredContent"] for result in results], [{"service_id": "korean", "tool": "review_standard_korean_text"}] * 2)


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


class UpstreamSessionExpiryTest(unittest.IsolatedAsyncioTestCase):
    async def test_404_with_a_session_is_session_expiry_and_without_one_is_a_plain_error(self) -> None:
        payload = {"jsonrpc": "2.0", "id": "1", "method": "tools/call", "params": {}}
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(404))) as client:
            with self.assertRaises(McpSessionExpired):
                await post_json(client, "http://a/mcp", payload, session_id="old")
            with self.assertRaises(OSError) as raised:
                await post_json(client, "http://a/mcp", payload)
        self.assertNotIsInstance(raised.exception, McpSessionExpired)


class UpstreamErrorEnvelopeTest(unittest.TestCase):
    """medic의 새 봉투(최상위 error)·예전 봉투(content text만 최상위 error)를 모두 같은 오류로 읽는다."""

    def test_error_code_is_read_from_current_and_legacy_medic_results(self) -> None:
        from main_mcp_gateway_server.mcp import _is_valid_tool_result, _result_code

        error = {"code": "dictionary_unavailable", "message": "x", "retryable": False}
        text = [{"type": "text", "text": json.dumps({"error": error})}]
        current = {"isError": True, "structuredContent": {"error": error}, "content": text}
        legacy = {"isError": True, "structuredContent": {"result": {"error": error}}, "content": text}
        for result in (current, legacy):
            self.assertTrue(_is_valid_tool_result(result))
            self.assertEqual(_result_code(result), "dictionary_unavailable")
        self.assertEqual(_result_code({"isError": False, "structuredContent": {"results": []}}), "ok")


if __name__ == "__main__":
    unittest.main()
