from __future__ import annotations

import json
import unittest

from n_note_ai_service_layers.services.mcp_gateway_client import (
    McpGatewayClient,
    McpGatewayConnectionError,
    McpGatewayProtocolError,
    McpGatewayToolError,
    _batch_url,
)
from support import FakeGateway as _Gateway


class McpGatewayClientTests(unittest.TestCase):
    def test_discovery_follows_the_streamable_http_handshake(self):
        """initialize → notifications/initialized → tools/list, 두 응답 형식을 받는 Accept와 프로토콜 버전 헤더."""
        gateway = _Gateway()
        client = McpGatewayClient(transport=gateway)
        catalog = client.discover("trace-1")
        self.assertEqual(catalog, tuple(gateway.tools))
        self.assertEqual([item[0]["method"] for item in gateway.requests], ["initialize", "notifications/initialized", "tools/list"])
        params = gateway.requests[0][0]["params"]
        self.assertEqual((params["protocolVersion"], params["capabilities"], params["clientInfo"]["name"]), ("2025-06-18", {}, "n-note-ai-service"))
        self.assertNotIn("id", gateway.requests[1][0])
        self.assertEqual(gateway.requests[0][1]["Content-Type"], "application/json")
        self.assertEqual(gateway.requests[0][1]["Accept"], "application/json, text/event-stream")
        self.assertNotIn("MCP-Protocol-Version", gateway.requests[0][1])
        self.assertEqual(gateway.requests[2][1]["MCP-Protocol-Version"], "2025-06-18")
        self.assertEqual(gateway.requests[2][1]["X-Request-ID"], "trace-1")
        self.assertEqual(client.model_tools()[0]["name"], "dictionary__lookup")
        self.assertEqual(client.model_tools()[0]["parameters"], gateway.tools[0]["inputSchema"])

    def test_call_rejects_unknown_or_schema_invalid_tools_before_gateway(self):
        gateway = _Gateway()
        client = McpGatewayClient(transport=gateway)
        client.discover()
        with self.assertRaises(McpGatewayProtocolError):
            client.call_tool("unknown", {"term": "x"})
        with self.assertRaises(McpGatewayProtocolError):
            client.call_tool("dictionary__lookup", {"unexpected": "x"})
        self.assertEqual(len(gateway.requests), 3)  # 핸드셰이크 3건뿐, 거부한 호출은 나가지 않는다.

    def test_session_id_is_sent_and_an_expired_session_is_reinitialized_once(self):
        class Stateful(_Gateway):
            def __init__(self):
                super().__init__(); self.sessions = 0; self.expired = False
            def __call__(self, url, body, headers, timeout):
                request = json.loads(body)
                if request.get("method") == "initialize":
                    self.requests.append((request, dict(headers)))
                    self.sessions += 1
                    response = {"jsonrpc": "2.0", "id": request["id"], "result": {"protocolVersion": "2025-06-18"}}
                    return 200, {"mcp-session-id": f"s-{self.sessions}"}, json.dumps(response).encode()
                if self.expired and headers.get("Mcp-Session-Id") == "s-1":
                    self.requests.append((request, dict(headers)))
                    return 404, {}, b""
                return super().__call__(url, body, headers, timeout)

        gateway = Stateful()
        client = McpGatewayClient(transport=gateway)
        client.discover()
        self.assertTrue(all(headers.get("Mcp-Session-Id") == "s-1" for _request, headers in gateway.requests[1:]))
        gateway.expired = True
        client.call_tool("dictionary__lookup", {"term": "x"})
        self.assertEqual(gateway.sessions, 2)
        self.assertEqual(gateway.requests[-1][1]["Mcp-Session-Id"], "s-2")
        self.assertEqual(gateway.requests[-1][0]["method"], "tools/call")

    def test_sse_response_picks_the_message_for_the_request(self):
        class Sse(_Gateway):
            def __call__(self, url, body, headers, timeout):
                status, response_headers, raw = super().__call__(url, body, headers, timeout)
                if status != 200:
                    return status, response_headers, raw
                notice = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
                stream = f"event: message\ndata: {notice}\n\nevent: message\ndata: {raw.decode()}\n\n".encode()
                return 200, {**response_headers, "Content-Type": "text/event-stream"}, stream

        client = McpGatewayClient(transport=Sse())
        self.assertEqual(client.discover()[0]["name"], "dictionary__lookup")

    def test_tool_result_error_is_not_treated_as_success(self):
        gateway = _Gateway()
        gateway.tool_result = {"isError": True, "content": [{"type": "text", "text": '{"error":{"code":"not_found","retryable":false}}'}]}
        client = McpGatewayClient(transport=gateway)
        client.discover()
        with self.assertRaises(McpGatewayToolError) as caught:
            client.call_tool("dictionary__lookup", {"term": "x"})
        self.assertEqual(caught.exception.code, "not_found")


def _batch_result(_name: str, arguments: dict) -> dict:
    """``term="down"``이면 upstream 장애 tool result, 아니면 받은 term을 구조화 결과로 돌려준다."""
    if arguments["term"] == "down":
        error = {"code": "upstream_unavailable", "message": "down", "retryable": True}
        return {"isError": True, "content": [{"type": "text", "text": json.dumps({"error": error})}]}
    return {"structuredContent": {"term": arguments["term"]}, "isError": False}


class McpGatewayBatchTests(unittest.TestCase):
    def test_batch_url_keeps_host_and_path_prefix(self):
        self.assertEqual(_batch_url("http://gateway:8080/mcp"), "http://gateway:8080/v1/tools/batch-call")
        self.assertEqual(_batch_url("https://proxy/gw/mcp/"), "https://proxy/gw/v1/tools/batch-call")

    def test_call_tools_sends_valid_calls_once_and_keeps_order(self):
        gateway = _Gateway(handler=_batch_result)
        client = McpGatewayClient(transport=gateway)
        client.discover()
        results = client.call_tools([("dictionary__lookup", {"term": "a"}), ("missing__tool", {}), ("dictionary__lookup", {"term": "down"})], "trace-9")
        self.assertEqual(len(gateway.batches), 1)
        self.assertEqual([item["arguments"]["term"] for item in gateway.batches[0]["calls"]], ["a", "down"])
        self.assertEqual(results[0]["structuredContent"], {"term": "a"})
        self.assertIsInstance(results[1], McpGatewayProtocolError)
        self.assertIsInstance(results[2], McpGatewayToolError)
        self.assertTrue(results[2].retryable)

    def test_batch_connection_failure_marks_every_call_unavailable(self):
        client = McpGatewayClient(transport=_Gateway(batch_error=OSError("refused")))
        client.discover()
        results = client.call_tools([("dictionary__lookup", {"term": "a"}), ("dictionary__lookup", {"term": "b"})])
        self.assertTrue(all(isinstance(item, McpGatewayConnectionError) for item in results))


class CatalogRefreshTests(unittest.TestCase):
    def test_incomplete_catalog_is_refreshed_sooner_than_the_full_ttl(self):
        from unittest.mock import patch
        gateway = _Gateway()
        client = McpGatewayClient(transport=gateway, catalog_ttl_seconds=300, expected_tools=("dictionary__lookup", "report__validate"), incomplete_catalog_ttl_seconds=15)
        with patch("n_note_ai_service_layers.services.mcp_gateway_client.time.monotonic", side_effect=[0.0, 10.0, 20.0, 20.0]):
            client.discover()  # 0초: 조회(report__validate 없음)
            client.discover()  # 10초: 불완전 TTL(15초) 안이라 재사용
            gateway.tools.append({"name": "report__validate", "description": "v", "inputSchema": {"type": "object"}})
            catalog = client.discover()  # 20초: 불완전 TTL이 지나 다시 조회
        self.assertEqual(sum(1 for request, _ in gateway.requests if request["method"] == "tools/list"), 2)
        self.assertIn("report__validate", [tool["name"] for tool in catalog])

    def test_complete_catalog_uses_the_full_ttl(self):
        from unittest.mock import patch
        gateway = _Gateway()
        client = McpGatewayClient(transport=gateway, catalog_ttl_seconds=300, expected_tools=("dictionary__lookup",), incomplete_catalog_ttl_seconds=15)
        with patch("n_note_ai_service_layers.services.mcp_gateway_client.time.monotonic", side_effect=[0.0, 0.0, 200.0]):
            client.discover()
            client.discover()
        self.assertEqual(sum(1 for request, _ in gateway.requests if request["method"] == "tools/list"), 1)

    def test_failed_discovery_is_not_retried_until_the_backoff_passes(self):
        """Gateway가 멈췄을 때 catalog lock을 기다리던 호출이 차례로 timeout까지 기다리지 않는다."""
        from unittest.mock import patch

        class Down(_Gateway):
            def __init__(self):
                super().__init__(); self.down = True; self.attempts = 0
            def __call__(self, url, body, headers, timeout):
                if self.down:
                    self.attempts += 1
                    raise TimeoutError("timed out")
                return super().__call__(url, body, headers, timeout)

        gateway = Down()
        client = McpGatewayClient(transport=gateway, catalog_ttl_seconds=300, failed_discovery_retry_seconds=5)
        now = [0.0]
        with patch("n_note_ai_service_layers.services.mcp_gateway_client.time.monotonic", side_effect=lambda: now[0]):
            with self.assertRaises(McpGatewayConnectionError):
                client.discover()
            now[0] = 3.0
            with self.assertRaises(McpGatewayConnectionError):
                client.discover()  # 5초 안: Gateway에 다시 묻지 않고 바로 실패한다
            self.assertEqual(gateway.attempts, 1)
            gateway.down = False
            now[0] = 6.0
            self.assertEqual(client.discover(), tuple(gateway.tools))  # 5초가 지나 다시 조회해 복구한다

    def test_failed_discovery_is_retried_immediately_without_a_catalog_ttl(self):
        class Flaky(_Gateway):
            def __init__(self):
                super().__init__(); self.failures = 1
            def __call__(self, url, body, headers, timeout):
                if self.failures:
                    self.failures -= 1
                    raise OSError("refused")
                return super().__call__(url, body, headers, timeout)

        gateway = Flaky()
        client = McpGatewayClient(transport=gateway)
        with self.assertRaises(McpGatewayConnectionError):
            client.discover()
        self.assertEqual(client.discover(), tuple(gateway.tools))



class SharedClientTests(unittest.TestCase):
    def test_client_is_shared_until_settings_change_and_the_replaced_one_is_closed(self):
        import os
        from unittest.mock import patch

        from n_note_ai_service_layers.api.deps import get_gateway_client

        with patch.dict(os.environ, {"MCP_GATEWAY_URL": "http://gateway-a:8080/mcp"}):
            first = get_gateway_client()
            self.assertIs(get_gateway_client(), first)
        with patch.dict(os.environ, {"MCP_GATEWAY_URL": "http://gateway-b:8080/mcp"}):
            second = get_gateway_client()
        self.assertIsNot(first, second)
        self.assertTrue(first._http.is_closed)
        self.assertFalse(second._http.is_closed)


if __name__ == "__main__":
    unittest.main()
