"""Gateway HTTP endpoint의 MCP protocol 및 요청 크기 제한을 검증한다."""

import http.client
import json
import os
import socket
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import uvicorn

from main_mcp_gateway_server.app import GatewayService, _upstream_timeout_seconds, configure_logging, create_app


class RecordingCatalog:
    """과대 요청이 catalog 또는 upstream으로 전달되지 않는지 확인하는 test double이다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.discovery_calls = 0

    def refresh(self) -> None:
        self.discovery_calls += 1

    def list_tools(self) -> list[dict]:
        """테스트용 catalog다."""
        return [{"name": "a__lookup", "description": "lookup", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]

    def call(self, name: str, arguments: dict, request_id: str | None = None) -> dict:
        """호출 기록을 남기고 계약에 맞는 성공 tool result를 반환한다(content 없이 구조화 결과만)."""
        self.calls.append((name, arguments, request_id))
        return {"structuredContent": {"ok": True}, "isError": False}

    def call_many(self, calls: list, request_id: str | None = None) -> list[dict]:
        return [self.call(item["name"], item.get("arguments", {}), request_id) for item in calls]


BATCH = "/v1/tools/batch-call"


class GatewayHandlerTest(unittest.TestCase):
    """HTTP 수준에서 protocol version 및 최대 본문 보호를 확인한다."""

    def setUp(self) -> None:
        self.catalog = RecordingCatalog()
        self.service = GatewayService(self.catalog, max_request_body_bytes=256)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(create_app(self.service), host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)

    def tearDown(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def _post(self, payload: dict, request_id: str = "test-request") -> tuple[int, dict]:
        """테스트 서버로 JSON-RPC POST를 보내고 HTTP status와 JSON 응답을 반환한다."""
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self.port}/mcp",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "X-Request-ID": request_id},
            method="POST",
        )
        try:
            with urlopen(request) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            return error.code, {}

    def test_initialize_negotiates_the_requested_protocol_version(self) -> None:
        """SDK 서버는 지원하는 버전이면 그대로, 모르는 버전이면 자기 최신 버전으로 답한다(MCP 버전 협상)."""
        params = {"capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}
        status, response = self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {**params, "protocolVersion": "2025-06-18"}})
        self.assertEqual((status, response["result"]["protocolVersion"]), (200, "2025-06-18"))
        self.assertEqual(response["result"]["serverInfo"]["name"], "mcp-gateway")
        _, response = self._post({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {**params, "protocolVersion": "old"}})
        self.assertNotEqual(response["result"]["protocolVersion"], "old")

    def test_rejects_large_body_without_catalog_call(self) -> None:
        """본문 한도를 넘는 tools/call은 upstream route 전에 request_too_large가 된다."""
        status, _response = self._post(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "a__lookup", "arguments": {"text": "x" * 1_000}}}
        )
        self.assertEqual(status, 413)
        self.assertEqual(self.catalog.calls, [])

    def test_forwards_received_request_id_to_catalog(self) -> None:
        """Gateway가 받은 request ID는 tools/call catalog 경계까지 변형 없이 유지된다."""
        status, response = self._post(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "a__lookup", "arguments": {}}},
            "request-from-client",
        )
        self.assertEqual(status, 200)
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"], {"ok": True})
        # 구조화 결과만 준 upstream에도 MCP 필수 필드인 content(같은 JSON 텍스트)를 채운다.
        self.assertEqual(json.loads(response["result"]["content"][0]["text"]), {"ok": True})
        self.assertEqual(self.catalog.calls, [("a__lookup", {}, "request-from-client")])

    def _post_path(self, path: str, payload: dict) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", path, body=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "X-Request-ID": "batch-1"})
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    def test_batch_call_returns_results_in_order(self) -> None:
        """일괄 호출은 각 도구 결과를 입력 순서대로 돌려주고 request ID를 유지한다."""
        status, response = self._post_path("/v1/tools/batch-call", {"calls": [{"name": "a__one", "arguments": {}}, {"name": "b__two", "arguments": {"text": "x"}}]})
        self.assertEqual(status, 200)
        self.assertEqual(len(response["results"]), 2)
        self.assertEqual(self.catalog.calls, [("a__one", {}, "batch-1"), ("b__two", {"text": "x"}, "batch-1")])

    def test_batch_call_rejects_empty_or_oversized_batches(self) -> None:
        """빈 batch와 상한을 넘는 batch는 upstream 호출 없이 400이다."""
        self.service.max_batch_calls = 2
        self.assertEqual(self._post_path("/v1/tools/batch-call", {"calls": []})[0], 400)
        calls = [{"name": "a__one", "arguments": {}}] * 3
        self.assertEqual(self._post_path("/v1/tools/batch-call", {"calls": calls})[0], 400)
        self.assertEqual(self.catalog.calls, [])

    def test_keep_alive_serves_multiple_requests_on_one_connection(self) -> None:
        """호출자는 한 연결로 여러 MCP 요청을 보낼 수 있다."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            for message_id in (1, 2):
                connection.request("POST", "/mcp", body=json.dumps({"jsonrpc": "2.0", "id": message_id, "method": "tools/list", "params": {}}).encode(), headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
                response = connection.getresponse()
                self.assertEqual(json.loads(response.read())["id"], message_id)
                self.assertFalse(response.will_close)
        finally:
            connection.close()

    def test_health_unknown_path_and_invalid_json(self) -> None:
        """`/healthz`는 200, 모르는 경로는 404, 깨진 JSON은 upstream 호출 없이 400이다."""
        with urlopen(f"http://127.0.0.1:{self.port}/healthz") as response:
            self.assertEqual((response.status, response.read()), (200, b"ok"))
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", "/other", body=b"{}", headers={"Content-Type": "application/json"})
            self.assertEqual(connection.getresponse().status, 404)
        finally:
            connection.close()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", BATCH, body=b"{broken", headers={"Content-Type": "application/json", "X-Request-ID": "bad-1"})
            response = connection.getresponse()
            self.assertEqual((response.status, response.getheader("X-Request-ID")), (400, "bad-1"))
        finally:
            connection.close()
        self.assertEqual(self.catalog.calls, [])

    def test_batch_body_over_limit_is_413(self) -> None:
        status, response = self._post_path("/v1/tools/batch-call", {"calls": [{"name": "a__one", "arguments": {"text": "x" * 1_000}}]})
        self.assertEqual((status, response), (413, {"error": "request_too_large"}))
        self.assertEqual(self.catalog.calls, [])

    def test_official_sdk_client_can_list_and_call_tools(self) -> None:
        """표준 MCP 클라이언트(SDK ClientSession)가 Gateway에 그대로 붙는다."""
        import anyio
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async def run() -> tuple[list[str], dict]:
            async with streamable_http_client(f"http://127.0.0.1:{self.port}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    result = await session.call_tool("a__lookup", {"text": "x"})
                    return [tool.name for tool in tools.tools], result.structuredContent

        names, structured = anyio.run(run)
        self.assertEqual((names, structured), (["a__lookup"], {"ok": True}))

    def test_tools_list_refreshes_upstreams(self) -> None:
        """Gateway보다 늦게 준비된 upstream도 다음 tools/list에서 발견한다."""
        status, _ = self._post({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        self.assertEqual(status, 200)
        self.assertEqual(self.catalog.discovery_calls, 1)


class LoggingConfigurationTest(unittest.TestCase):
    """컨테이너 로그 레벨 설정을 검증한다."""

    def test_defaults_to_info(self) -> None:
        """환경 변수가 없으면 운영 추적에 필요한 INFO 로그를 활성화한다."""
        with patch.dict(os.environ, {}, clear=True), patch("main_mcp_gateway_server.app.logging.basicConfig") as basic_config:
            configure_logging()
        self.assertEqual(basic_config.call_args.kwargs["level"], 20)

    def test_rejects_invalid_log_level(self) -> None:
        """오타 난 로그 레벨로 조용히 로그가 사라지지 않게 시작을 중단한다."""
        with patch.dict(os.environ, {"GATEWAY_LOG_LEVEL": "verbose"}, clear=True):
            with self.assertRaisesRegex(ValueError, "GATEWAY_LOG_LEVEL"):
                configure_logging()


class UpstreamTimeoutConfigurationTest(unittest.TestCase):
    """upstream 요청 timeout 설정을 검증한다."""

    def test_defaults_to_five_seconds(self) -> None:
        """환경 변수가 없으면 기존 5초 timeout을 유지한다."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_upstream_timeout_seconds(), 5.0)

    def test_reads_configured_timeout(self) -> None:
        """느린 배포 환경에서는 환경 변수로 timeout을 늘릴 수 있다."""
        with patch.dict(os.environ, {"GATEWAY_UPSTREAM_TIMEOUT_SECONDS": "12.5"}, clear=True):
            self.assertEqual(_upstream_timeout_seconds(), 12.5)

    def test_rejects_non_positive_timeout(self) -> None:
        """0 이하 또는 숫자가 아닌 값으로 Gateway를 시작하지 않는다."""
        for value in ("0", "-1", "slow"):
            with patch.dict(os.environ, {"GATEWAY_UPSTREAM_TIMEOUT_SECONDS": value}, clear=True):
                with self.assertRaisesRegex(ValueError, "GATEWAY_UPSTREAM_TIMEOUT_SECONDS"):
                    _upstream_timeout_seconds()
