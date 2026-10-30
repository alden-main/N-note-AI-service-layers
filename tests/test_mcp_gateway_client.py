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


class _Gateway:
    def __init__(self) -> None:
        self.requests: list[tuple[dict, dict]] = []
        self.tools = [{"name": "dictionary__lookup", "description": "lookup", "inputSchema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"], "additionalProperties": False}}]
        self.tool_result: dict = {"content": [{"type": "text", "text": "ok"}]}

    def __call__(self, url, body, headers, timeout):
        request = json.loads(body)
        self.requests.append((request, dict(headers)))
        if request["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        elif request["method"] == "tools/list":
            result = {"tools": self.tools}
        else:
            result = self.tool_result
        response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps(response).encode()


class McpGatewayClientTests(unittest.TestCase):
    def test_discovery_sends_minimal_json_rpc_headers_and_uses_gateway_catalog(self):
        gateway = _Gateway()
        client = McpGatewayClient(transport=gateway)
        catalog = client.discover("trace-1")
        self.assertEqual(catalog, tuple(gateway.tools))
        self.assertEqual([item[0]["method"] for item in gateway.requests], ["initialize", "tools/list"])
        self.assertEqual(gateway.requests[0][0]["params"], {"protocolVersion": "2025-06-18"})
        self.assertEqual(gateway.requests[0][1]["Content-Type"], "application/json")
        self.assertEqual(gateway.requests[0][1]["Accept"], "application/json")
        self.assertEqual(gateway.requests[1][1]["X-Request-ID"], "trace-1")
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
        self.assertEqual(len(gateway.requests), 2)

    def test_tool_result_error_is_not_treated_as_success(self):
        gateway = _Gateway()
        gateway.tool_result = {"isError": True, "content": [{"type": "text", "text": '{"error":{"code":"not_found","retryable":false}}'}]}
        client = McpGatewayClient(transport=gateway)
        client.discover()
        with self.assertRaises(McpGatewayToolError) as caught:
            client.call_tool("dictionary__lookup", {"term": "x"})
        self.assertEqual(caught.exception.code, "not_found")


class _BatchGateway(_Gateway):
    def __init__(self, fail_connection: bool = False) -> None:
        super().__init__()
        self.batches: list[dict] = []
        self.fail_connection = fail_connection

    def __call__(self, url, body, headers, timeout):
        if not url.endswith("/v1/tools/batch-call"):
            return super().__call__(url, body, headers, timeout)
        if self.fail_connection:
            raise OSError("refused")
        request = json.loads(body)
        self.batches.append(request)
        results = [
            {"isError": True, "content": [{"type": "text", "text": json.dumps({"error": {"code": "upstream_unavailable", "message": "down", "retryable": True}})}]}
            if item["arguments"]["term"] == "down" else {"structuredContent": {"term": item["arguments"]["term"]}, "isError": False}
            for item in request["calls"]
        ]
        return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps({"results": results}).encode()


class McpGatewayBatchTests(unittest.TestCase):
    def test_batch_url_keeps_host_and_path_prefix(self):
        self.assertEqual(_batch_url("http://gateway:8080/mcp"), "http://gateway:8080/v1/tools/batch-call")
        self.assertEqual(_batch_url("https://proxy/gw/mcp/"), "https://proxy/gw/v1/tools/batch-call")

    def test_call_tools_sends_valid_calls_once_and_keeps_order(self):
        gateway = _BatchGateway()
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
        client = McpGatewayClient(transport=_BatchGateway(fail_connection=True))
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


if __name__ == "__main__":
    unittest.main()
