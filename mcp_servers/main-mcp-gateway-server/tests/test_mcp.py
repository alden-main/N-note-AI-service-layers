"""Gateway catalog의 discovery, 이름 변환, 장애 격리를 검증한다."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from main_mcp_gateway_server.config import UpstreamConfig, configured_upstreams
from main_mcp_gateway_server.mcp import GatewayCatalog, McpResponse, PROTOCOL_VERSION


class FakeTransport:
    """외부 네트워크 없이 catalog 계약을 검증하는 결정적 transport다."""

    def __init__(self) -> None:
        self.health = {"http://a/healthz": True, "http://b/healthz": True}
        self.calls: list[tuple[str, dict, str | None, str | None]] = []
        self.protocol_version = PROTOCOL_VERSION
        self.tools = [{"name": "lookup", "description": "lookup tool", "inputSchema": {"type": "object"}}]
        self.tool_result: dict = {"structuredContent": {"source": "fake"}, "isError": False}

    def healthy(self, url: str) -> bool:
        return self.health[url]

    def request_json(self, url: str, payload: dict, request_id: str | None = None, session_id: str | None = None) -> McpResponse:
        self.calls.append((url, payload, request_id, session_id))
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
