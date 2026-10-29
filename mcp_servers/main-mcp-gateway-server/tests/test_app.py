"""Gateway HTTP endpoint의 MCP protocol 및 요청 크기 제한을 검증한다."""

from http.server import ThreadingHTTPServer
import json
import os
import threading
import unittest
from urllib.request import Request, urlopen
from unittest.mock import patch

from main_mcp_gateway_server.app import GatewayHandler, _upstream_timeout_seconds, configure_logging


class RecordingCatalog:
    """과대 요청이 catalog 또는 upstream으로 전달되지 않는지 확인하는 test double이다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.discovery_calls = 0

    def discover_all(self) -> None:
        self.discovery_calls += 1

    def list_tools(self) -> list[dict]:
        """테스트용 빈 catalog를 반환한다."""
        return []

    def call(self, name: str, arguments: dict, request_id: str | None = None) -> dict:
        """호출 기록을 남기고 계약에 맞는 성공 tool result를 반환한다."""
        self.calls.append((name, arguments, request_id))
        return {"structuredContent": {}, "isError": False}


class GatewayHandlerTest(unittest.TestCase):
    """HTTP 수준에서 protocol version 및 최대 본문 보호를 확인한다."""

    def setUp(self) -> None:
        self.catalog = RecordingCatalog()
        self.previous_catalog = getattr(GatewayHandler, "catalog", None)
        self.previous_limit = GatewayHandler.max_request_body_bytes
        GatewayHandler.catalog = self.catalog
        GatewayHandler.max_request_body_bytes = 256
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        if self.previous_catalog is None:
            del GatewayHandler.catalog
        else:
            GatewayHandler.catalog = self.previous_catalog
        GatewayHandler.max_request_body_bytes = self.previous_limit

    def _post(self, payload: dict, request_id: str = "test-request") -> tuple[int, dict]:
        """테스트 서버로 JSON-RPC POST를 보내고 HTTP status와 JSON 응답을 반환한다."""
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/mcp",
            data=data,
            headers={"Content-Type": "application/json", "X-Request-ID": request_id},
            method="POST",
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_rejects_unsupported_initialize_version(self) -> None:
        """Gateway MCP server는 지원하지 않는 initialize version을 JSON-RPC 오류로 거부한다."""
        status, response = self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "old"}})
        self.assertEqual(status, 200)
        self.assertEqual(response["error"]["code"], -32602)

    def test_rejects_large_body_without_catalog_call(self) -> None:
        """본문 한도를 넘는 tools/call은 upstream route 전에 request_too_large가 된다."""
        status, response = self._post(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "a__lookup", "arguments": {"text": "x" * 1_000}}}
        )
        self.assertEqual(status, 200)
        self.assertTrue(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["error"]["code"], "request_too_large")
        self.assertEqual(self.catalog.calls, [])

    def test_forwards_received_request_id_to_catalog(self) -> None:
        """Gateway가 받은 request ID는 tools/call catalog 경계까지 변형 없이 유지된다."""
        status, response = self._post(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "a__lookup", "arguments": {}}},
            "request-from-client",
        )
        self.assertEqual(status, 200)
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(self.catalog.calls, [("a__lookup", {}, "request-from-client")])

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
