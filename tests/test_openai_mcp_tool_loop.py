from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter, model_tool_output
from support import sample_form_schema
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
            return SimpleNamespace(id="response-1", output=[SimpleNamespace(type="function_call", name="dictionary__lookup", arguments='{"term":"통증"}', call_id="call-1")], output_text=None)
        return SimpleNamespace(id="response-2", output=[], output_text='{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":{"assessment.pain_score":["tool:1"]},"needs_confirmation":true}')


class OpenAiMcpToolLoopTests(unittest.TestCase):
    def test_gateway_catalog_drives_tool_loop_and_final_draft_stays_filtered(self):
        gateway = _Gateway()
        responses = _Responses()
        client = SimpleNamespace(responses=responses)
        adapter = OpenAiDraftGenerationAdapter(client, gateway_client=McpGatewayClient(transport=gateway))
        result = adapter.generate({"facts": []}, sample_form_schema())
        self.assertEqual(gateway.called, [{"name": "dictionary__lookup", "arguments": {"term": "통증"}}])
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertTrue(result.needs_confirmation)
        self.assertEqual(responses.requests[0]["tools"][0]["name"], "dictionary__lookup")
        self.assertFalse(responses.requests[0]["tools"][0]["strict"])
        self.assertEqual(responses.requests[1]["input"][0]["type"], "function_call_output")

    def test_only_allowlisted_gateway_tools_are_exposed_or_callable(self):
        gateway = _Gateway()
        responses = _Responses()
        responses.number = 1  # 도구가 없으므로 첫 응답부터 최종 초안을 반환한다.
        adapter = OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=gateway), allowed_tool_names=("other__search",))
        adapter.generate({"facts": []}, sample_form_schema())
        self.assertNotIn("tools", responses.requests[0])
        self.assertEqual(gateway.called, [])

    def test_draft_is_generated_without_tools_when_gateway_is_unreachable(self):
        def unreachable(url, body, headers, timeout):
            raise OSError("down")
        responses = _Responses()
        responses.number = 1  # 첫 응답부터 최종 초안을 반환한다.
        adapter = OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=unreachable))
        result = adapter.generate({"facts": []}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertNotIn("tools", responses.requests[0])

    def test_tool_calls_in_one_response_run_concurrently_and_keep_order(self):
        import threading
        barrier = threading.Barrier(2, timeout=2)

        class SlowGateway(_Gateway):
            def __call__(self, url, body, headers, timeout):
                if b"tools/call" in body:
                    barrier.wait()
                return super().__call__(url, body, headers, timeout)

        class Responses(_Responses):
            def create(self, **kwargs):
                self.requests.append(kwargs); self.number += 1
                if self.number == 1:
                    calls = [SimpleNamespace(type="function_call", name="dictionary__lookup", arguments=f'{{"term":"{term}"}}', call_id=f"call-{term}") for term in ("a", "b")]
                    return SimpleNamespace(id="response-1", output=calls, output_text=None)
                return SimpleNamespace(id="response-2", output=[], output_text='{"proposed_fields":{},"evidence_refs":[],"needs_confirmation":false}')

        responses = Responses()
        adapter = OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=SlowGateway()))
        adapter.generate({"facts": []}, sample_form_schema())
        self.assertEqual([item["call_id"] for item in responses.requests[1]["input"]], ["call-a", "call-b"])

    def test_calls_beyond_the_limit_get_tool_limit_reached(self):
        class Responses(_Responses):
            def create(self, **kwargs):
                self.requests.append(kwargs); self.number += 1
                if self.number == 1:
                    calls = [SimpleNamespace(type="function_call", name="dictionary__lookup", arguments='{"term":"x"}', call_id=f"call-{index}") for index in range(3)]
                    return SimpleNamespace(id="response-1", output=calls, output_text=None)
                return SimpleNamespace(id="response-2", output=[], output_text='{"proposed_fields":{},"evidence_refs":[],"needs_confirmation":false}')

        gateway = _Gateway()
        responses = Responses()
        OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=gateway), max_tool_calls=2).generate({"facts": []}, sample_form_schema())
        outputs = [item["output"] for item in responses.requests[1]["input"]]
        self.assertEqual(len(gateway.called), 2)
        self.assertEqual(outputs[2], '{"error": "tool_limit_reached"}')

    def test_tool_rounds_are_capped_and_the_last_request_forbids_tools(self):
        """도구를 계속 요청하는 모델도 LLM 호출은 ``1 + max_tool_rounds``번에서 끝난다."""
        final = '{"proposed_fields":{},"evidence_refs":[],"needs_confirmation":false}'

        class Responses(_Responses):
            def create(self, **kwargs):
                self.requests.append(kwargs); self.number += 1
                if self.number > 10:
                    raise AssertionError("tool loop did not stop")
                call = SimpleNamespace(type="function_call", name="dictionary__lookup", arguments='{"term":"x"}', call_id=f"call-{self.number}")
                # tool_choice="none"이면 실제 API처럼 도구 없이 초안을 낸다.
                if kwargs.get("tool_choice") == "none":
                    return SimpleNamespace(id=f"r-{self.number}", output=[], output_text=final)
                return SimpleNamespace(id=f"r-{self.number}", output=[call], output_text=None)

        gateway = _Gateway()
        responses = Responses()
        OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=gateway), max_tool_rounds=2).generate({"facts": []}, sample_form_schema())
        self.assertEqual(len(responses.requests), 3)
        self.assertEqual(len(gateway.called), 2)
        self.assertEqual([request.get("tool_choice") for request in responses.requests], [None, None, "none"])

    def test_loop_stops_even_if_the_model_ignores_tool_choice_none(self):
        class Responses(_Responses):
            def create(self, **kwargs):
                self.requests.append(kwargs); self.number += 1
                if self.number > 10:
                    raise AssertionError("tool loop did not stop")
                call = SimpleNamespace(type="function_call", name="dictionary__lookup", arguments='{"term":"x"}', call_id=f"call-{self.number}")
                return SimpleNamespace(id=f"r-{self.number}", output=[call], output_text='{"proposed_fields":{},"evidence_refs":[],"needs_confirmation":false}')

        responses = Responses()
        OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=_Gateway()), max_tool_rounds=1).generate({"facts": []}, sample_form_schema())
        self.assertEqual(len(responses.requests), 2)

    def test_zero_tool_rounds_offers_no_tools(self):
        gateway = _Gateway()
        responses = _Responses()
        responses.number = 1
        OpenAiDraftGenerationAdapter(SimpleNamespace(responses=responses), gateway_client=McpGatewayClient(transport=gateway), max_tool_rounds=0).generate({"facts": []}, sample_form_schema())
        self.assertNotIn("tools", responses.requests[0])
        self.assertEqual(gateway.called, [])


class ModelToolOutputTests(unittest.TestCase):
    def test_standard_dictionary_entry_keeps_headword_pos_and_short_definition(self):
        evidence = {"headword": "통증", "pronunciation": "[통ː쯩]", "part_of_speech": "「명사」\n", "definition": "아픈 증세.\n" + "가" * 200,
                    "source": {"file_name": "a.xls", "row_number": 1, "source_hash": "h"}}
        payload = {"entries": [{"entry_id": "a.xls:1", "surface": "통증", "evidence": evidence}]}
        result = {"content": [{"type": "text", "text": json.dumps(payload)}], "structuredContent": payload, "isError": False}
        compact = model_tool_output(result)
        entry = compact["entries"][0]
        self.assertEqual((entry["surface"], entry["part_of_speech"]), ("통증", "명사"))
        self.assertTrue(entry["definition"].startswith("아픈 증세."))
        self.assertLessEqual(len(entry["definition"]), 120)
        self.assertNotIn("content", compact)

    def test_medical_search_drops_codes_and_merges_duplicates(self):
        term = lambda code, eng: {"source_id": "s", "term_code": code, "concept_code": code, "eng_name": eng, "kor_name": "고혈압", "description": None}
        payload = {"result": {"search_result_id": "x", "dictionary_version": "v", "results": [
            {"term": term("H1", "HTN"), "match_type": "exact", "score": 1.0},
            {"term": term("H2", "HTN"), "match_type": "exact", "score": 1.0},
            {"term": term("H3", "Hypertension"), "match_type": "exact", "score": 1.0},
        ]}}
        compact = model_tool_output({"structuredContent": payload, "isError": False})
        self.assertEqual(compact, {"results": [
            {"kor_name": "고혈압", "eng_name": "HTN", "match_type": "exact", "score": 1.0},
            {"kor_name": "고혈압", "eng_name": "Hypertension", "match_type": "exact", "score": 1.0},
        ]})

    def test_unknown_results_are_passed_through(self):
        plain = {"content": [{"type": "text", "text": "found"}]}
        self.assertIs(model_tool_output(plain), plain)
        self.assertEqual(model_tool_output({"structuredContent": {"other": 1}}), {"other": 1})


if __name__ == "__main__":
    unittest.main()
