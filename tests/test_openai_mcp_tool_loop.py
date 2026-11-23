"""Responses API 초안 생성의 사전 도구(Gateway) 왕복을 검증한다(가짜 HTTP 서버, ``llm_support``)."""

from __future__ import annotations

import json
import threading
import unittest

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.adapters.llm_completion import GenerationOptions, model_tool_output
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from llm_support import FakeServer, adapter, function_outputs, responses
from support import FakeGateway as _Gateway, sample_form_schema

FINAL = '{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":[{"field_id":"assessment.pain_score","refs":["tool:1"]}],"needs_confirmation":true}'
EMPTY = '{"proposed_fields":{},"evidence_refs":[],"needs_confirmation":false}'
LOOKUP = ("call-1", "dictionary__lookup", {"term": "통증"})


def generate(server: FakeServer, **options):
    return adapter(server, api="responses", **options).generate({"facts": []}, sample_form_schema())


class McpToolLoopTests(unittest.TestCase):
    def test_gateway_catalog_drives_tool_loop_and_final_draft_stays_filtered(self):
        gateway = _Gateway()
        server = FakeServer(responses(None, LOOKUP), responses(FINAL))
        result = generate(server, gateway=gateway)
        self.assertEqual(gateway.called, [{"name": "dictionary__lookup", "arguments": {"term": "통증"}}])
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertTrue(result.needs_confirmation)
        first, second = server.requests
        self.assertEqual(first["tools"][0]["name"], "dictionary__lookup")
        self.assertFalse(first["tools"][0]["strict"])
        # store=False라 서버에 앞 응답이 없다: 원래 입력 → 모델의 function_call → 그 결과를 이어 보낸다.
        self.assertTrue(all(request["store"] is False for request in server.requests))
        self.assertNotIn("previous_response_id", second)
        types = [item.get("type", item.get("role")) for item in second["input"]]
        self.assertEqual(types[-2:], ["function_call", "function_call_output"])
        self.assertEqual(second["input"][-1]["call_id"], "call-1")
        self.assertIn(prompts.get().draft.tool, first["instructions"])

    def test_instructions_do_not_mention_tools_when_none_are_offered(self):
        server = FakeServer(responses(FINAL))
        generate(server, gateway=_Gateway(), max_tool_rounds=0)
        self.assertNotIn(prompts.get().draft.tool, server.requests[0]["instructions"])
        self.assertNotIn("tools", server.requests[0])

    def test_generation_options_reach_the_responses_request(self):
        server = FakeServer(responses(FINAL))
        generate(server, options=GenerationOptions(max_tokens=512, temperature=0.0, extra_body={"seed": 1}))
        request = server.requests[0]
        self.assertEqual((request["max_output_tokens"], request["temperature"], request["seed"]), (512, 0.0, 1))

    def test_tool_not_found_refreshes_catalog_and_survives_gateway_outage(self):
        """도구가 사라졌는데 재조회까지 실패해도 초안 생성은 계속된다(그 호출만 오류로 답한다)."""
        class Gateway(_Gateway):
            def __init__(self):
                super().__init__(); self.lists = 0; self.down = False
            def __call__(self, url, body, headers, timeout):
                if self.down:
                    raise OSError("down")
                message = json.loads(body)
                if message["method"] == "tools/list":
                    self.lists += 1
                if message["method"] == "tools/call":
                    self.down = True
                    result = {"isError": True, "code": "not_found"}
                    return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode()
                return super().__call__(url, body, headers, timeout)

        server = FakeServer(responses(None, LOOKUP), responses(FINAL))
        result = generate(server, gateway=McpGatewayClient(transport=Gateway(), catalog_ttl_seconds=300))
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(json.loads(function_outputs(server.requests[1])[0]["output"]), {"error": "not_found", "retryable": False})

    def test_only_allowlisted_gateway_tools_are_exposed_or_callable(self):
        gateway = _Gateway()
        server = FakeServer(responses(FINAL))
        generate(server, gateway=gateway, allowed_tool_names=("other__search",))
        self.assertNotIn("tools", server.requests[0])
        self.assertEqual(gateway.called, [])

    def test_draft_is_generated_without_tools_when_gateway_is_unreachable(self):
        def unreachable(url, body, headers, timeout):
            raise OSError("down")
        server = FakeServer(responses(FINAL))
        result = generate(server, gateway=McpGatewayClient(transport=unreachable))
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertNotIn("tools", server.requests[0])

    def test_tool_calls_in_one_response_run_concurrently_and_keep_order(self):
        barrier = threading.Barrier(2, timeout=2)

        class SlowGateway(_Gateway):
            def __call__(self, url, body, headers, timeout):
                if b"tools/call" in body:
                    barrier.wait()  # 두 호출이 동시에 와야 통과한다
                return super().__call__(url, body, headers, timeout)

        calls = [(f"call-{term}", "dictionary__lookup", {"term": term}) for term in ("a", "b")]
        server = FakeServer(responses(None, *calls), responses(EMPTY))
        generate(server, gateway=SlowGateway())
        self.assertEqual([item["call_id"] for item in function_outputs(server.requests[1])], ["call-a", "call-b"])

    def test_calls_beyond_the_limit_get_tool_limit_reached(self):
        gateway = _Gateway()
        calls = [(f"call-{index}", "dictionary__lookup", {"term": "x"}) for index in range(3)]
        server = FakeServer(responses(None, *calls), responses(EMPTY))
        generate(server, gateway=gateway, max_tool_calls=2)
        outputs = sorted(function_outputs(server.requests[1]), key=lambda item: item["call_id"])
        self.assertEqual(len(gateway.called), 2)
        self.assertEqual(sum(json.loads(item["output"]) == {"error": "tool_limit_reached"} for item in outputs), 1)

    def test_tool_rounds_are_capped_and_the_last_request_forbids_tools(self):
        """도구를 계속 요청하는 모델도 LLM 호출은 ``1 + max_tool_rounds``번에서 끝난다."""
        count = [0]

        def reply(request):
            count[0] += 1
            if count[0] > 10:
                raise AssertionError("tool loop did not stop")
            # tool_choice="none"이면 실제 API처럼 도구 없이 초안을 낸다.
            if request.get("tool_choice") == "none":
                return responses(EMPTY)
            return responses(None, (f"call-{count[0]}", "dictionary__lookup", {"term": "x"}))

        gateway = _Gateway()
        server = FakeServer(reply)
        generate(server, gateway=gateway, max_tool_rounds=2)
        self.assertEqual(len(server.requests), 3)
        self.assertEqual(len(gateway.called), 2)
        self.assertEqual([request.get("tool_choice") for request in server.requests][-1], "none")

    def test_loop_stops_even_if_the_model_ignores_tool_choice_none(self):
        """tool_choice="none"을 무시하고 도구를 계속 요청해도 1 + max_tool_rounds번에서 멈추고 마지막 응답의 초안을 쓴다."""
        count = [0]

        def reply(request):
            count[0] += 1
            if count[0] > 10:
                raise AssertionError("tool loop did not stop")
            return responses(EMPTY, (f"call-{count[0]}", "dictionary__lookup", {"term": "x"}))

        server = FakeServer(reply)
        result = generate(server, gateway=_Gateway(), max_tool_rounds=1)
        self.assertEqual(len(server.requests), 2)
        self.assertEqual(result.proposed_fields, {})

    def test_zero_tool_rounds_offers_no_tools(self):
        gateway = _Gateway()
        server = FakeServer(responses(FINAL))
        generate(server, gateway=gateway, max_tool_rounds=0)
        self.assertNotIn("tools", server.requests[0])
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
