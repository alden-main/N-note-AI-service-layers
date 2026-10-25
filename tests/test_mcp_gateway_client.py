from __future__ import annotations

import json
import unittest

from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient, McpGatewayProtocolError, McpGatewayToolError


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


if __name__ == "__main__":
    unittest.main()
