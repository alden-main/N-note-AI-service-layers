from __future__ import annotations

from types import SimpleNamespace
import unittest

from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter
from n_note_ai_service_layers.policies.pressure_injury_schema import pressure_injury_poc_schema
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient


class _Gateway:
    def __init__(self): self.called = []
    def __call__(self, url, body, headers, timeout):
        import json
        message = json.loads(body)
        method = message["method"]
        if method == "initialize": result = {"protocolVersion": "2025-06-18"}
        elif method == "tools/list": result = {"tools": [{"name": "dictionary__lookup", "description": "lookup", "inputSchema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"], "additionalProperties": False}}]}
        else:
            self.called.append(message["params"])
            result = {"content": [{"type": "text", "text": "found"}]}
        return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode()


class _Responses:
    def __init__(self): self.requests = []; self.number = 0
    def create(self, **kwargs):
        self.requests.append(kwargs); self.number += 1
        if self.number == 1:
            return SimpleNamespace(id="response-1", output=[SimpleNamespace(type="function_call", name="dictionary__lookup", arguments='{"term":"욕창"}', call_id="call-1")], output_text=None)
        return SimpleNamespace(id="response-2", output=[], output_text='{"proposed_fields":{"braden.nutrition":2},"evidence_refs":{"braden.nutrition":["tool:1"]},"needs_confirmation":true}')


class OpenAiMcpToolLoopTests(unittest.TestCase):
    def test_gateway_catalog_drives_tool_loop_and_final_draft_stays_filtered(self):
        gateway = _Gateway()
        responses = _Responses()
        client = SimpleNamespace(responses=responses)
        adapter = OpenAiDraftGenerationAdapter(client, gateway_client=McpGatewayClient(transport=gateway))
        result = adapter.generate({"facts": []}, pressure_injury_poc_schema())
        self.assertEqual(gateway.called, [{"name": "dictionary__lookup", "arguments": {"term": "욕창"}}])
        self.assertEqual(result.proposed_fields, {"braden.nutrition": 2})
        self.assertTrue(result.needs_confirmation)
        self.assertEqual(responses.requests[0]["tools"][0]["name"], "dictionary__lookup")
        self.assertFalse(responses.requests[0]["tools"][0]["strict"])
        self.assertEqual(responses.requests[1]["input"][0]["type"], "function_call_output")

    def test_only_allowlisted_gateway_tools_are_exposed_or_callable(self):
        gateway = _Gateway()
        responses = _Responses()
        responses.number = 1  # 도구가 없으므로 첫 응답부터 최종 초안을 반환한다.
        adapter = OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=gateway), allowed_tool_names=("other__search",))
        adapter.generate({"facts": []}, pressure_injury_poc_schema())
        self.assertNotIn("tools", responses.requests[0])
        self.assertEqual(gateway.called, [])

    def test_draft_is_generated_without_tools_when_gateway_is_unreachable(self):
        def unreachable(url, body, headers, timeout):
            raise OSError("down")
        responses = _Responses()
        responses.number = 1  # 첫 응답부터 최종 초안을 반환한다.
        adapter = OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=unreachable))
        result = adapter.generate({"facts": []}, pressure_injury_poc_schema())
        self.assertEqual(result.proposed_fields, {"braden.nutrition": 2})
        self.assertNotIn("tools", responses.requests[0])


if __name__ == "__main__":
    unittest.main()
