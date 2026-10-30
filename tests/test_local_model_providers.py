from __future__ import annotations

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.openai_chat_draft_generation import FINAL_OUTPUT_INSTRUCTION, FINAL_TURN_INSTRUCTION, OpenAiChatDraftGenerationAdapter
from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter
from support import sample_form_schema
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient

FINAL = '{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":{"assessment.pain_score":["transcript"]},"needs_confirmation":false}'


def _completion(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class _Completions:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class _Gateway:
    def __init__(self):
        self.called = []

    def __call__(self, url, body, headers, timeout):
        message = json.loads(body)
        method = message["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        elif method == "tools/list":
            result = {"tools": [{"name": "dictionary__lookup", "description": "lookup", "inputSchema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}}]}
        else:
            self.called.append(message["params"])
            result = {"content": [{"type": "text", "text": "found"}]}
        return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode()


class ModelSettingsTests(unittest.TestCase):
    def test_openai_is_default_and_legacy_variables_still_apply(self):
        with patch.dict(os.environ, {"OPENAI_DRAFT_MODEL": "gpt-legacy", "N_NOTE_LLM_BASE_URL": ""}, clear=True):
            self.assertEqual(settings.get().llm_endpoint, settings.ModelEndpoint("gpt-legacy"))
            self.assertEqual(settings.get().llm_api, "responses")
            self.assertTrue(settings.get().llm_tool_calling_enabled)
            self.assertEqual(settings.get().stt_endpoint.provider, "openai")

    def test_local_server_profile_selects_chat_api_and_parses_options(self):
        env = {
            "N_NOTE_LLM_BASE_URL": "http://local-llm:8000/v1",
            "N_NOTE_LLM_MODEL": "medgemma-27b",
            "N_NOTE_LLM_TOOL_CALLING": "false",
            "N_NOTE_LLM_EXTRA_BODY": '{"chat_template_kwargs":{"enable_thinking":false}}',
            "N_NOTE_LLM_MAX_TOKENS": "2048",
            "N_NOTE_STT_BASE_URL": "http://local-stt:8000/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(settings.get().llm_api, "chat")
            self.assertFalse(settings.get().llm_tool_calling_enabled)
            self.assertEqual(settings.get().llm_extra_body, {"chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(settings.get().llm_max_tokens, 2048)
            self.assertEqual(settings.get().stt_endpoint.provider, "openai-compatible")

    def test_factory_builds_chat_adapter_without_tools_when_disabled(self):
        env = {"N_NOTE_LLM_BASE_URL": "http://local-llm:8000/v1", "N_NOTE_LLM_MODEL": "medgemma-27b", "N_NOTE_LLM_TOOL_CALLING": "false", "OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            generator = model_providers.draft_generator(McpGatewayClient(transport=_Gateway()))
            stt = model_providers.speech_to_text_provider()
        self.assertIsInstance(generator, OpenAiChatDraftGenerationAdapter)
        self.assertIsNone(generator._gateway_client)
        self.assertEqual(str(generator._client.base_url), "http://local-llm:8000/v1/")
        # LLM만 로컬이고 STT는 OpenAI인 혼합 구성이다.
        self.assertEqual(stt._provider, "openai")

    def test_factory_keeps_responses_adapter_for_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            generator = model_providers.draft_generator(None)
        self.assertIs(type(generator), OpenAiDraftGenerationAdapter)


class ChatDraftAdapterTests(unittest.TestCase):
    def test_structured_output_without_tools_in_one_request(self):
        completions = _Completions(_completion("<think>검토</think>```json\n" + FINAL + "\n```"))
        adapter = OpenAiChatDraftGenerationAdapter(_client(completions), "medgemma-27b", extra_body={"top_k": 20}, max_tokens=1024)
        result = adapter.generate({"transcript": "식사 절반 섭취"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(result.model_version, "medgemma-27b")
        request = completions.requests[0]
        self.assertEqual(len(completions.requests), 1)
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertEqual(request["extra_body"], {"top_k": 20})
        self.assertEqual(request["max_tokens"], 1024)
        # 고정 지침·schema는 system에, 요청별 payload만 user에 둔다(prefix cache 적중, user 메시지 연속 없음).
        system, user = request["messages"]
        self.assertEqual((system["role"], user["role"]), ("system", "user"))
        self.assertIn(FINAL_OUTPUT_INSTRUCTION, system["content"])
        self.assertIn('"proposed_fields"', system["content"])
        self.assertEqual(json.loads(user["content"]), {"transcript": "식사 절반 섭취"})
        self.assertNotIn("tools", request)

    def test_system_prefix_is_identical_across_requests_for_the_same_form(self):
        completions = _Completions(_completion(FINAL), _completion(FINAL))
        adapter = OpenAiChatDraftGenerationAdapter(_client(completions), "m")
        adapter.generate({"transcript": "a"}, sample_form_schema())
        adapter.generate({"transcript": "b"}, sample_form_schema())
        first, second = (request["messages"][0] for request in completions.requests)
        self.assertEqual(first, second)

    def test_schema_valid_answer_in_tool_phase_skips_the_structured_call(self):
        answer = '```json\n{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":[{"field_id":"assessment.pain_score","refs":["transcript"]}],"needs_confirmation":false}\n```'
        completions = _Completions(_completion(answer))
        adapter = OpenAiChatDraftGenerationAdapter(_client(completions), "qwen", gateway_client=McpGatewayClient(transport=_Gateway()))
        result = adapter.generate({"transcript": "식사 절반"}, sample_form_schema())
        self.assertEqual(len(completions.requests), 1)
        self.assertIn("tools", completions.requests[0])
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(result.evidence_refs, {"assessment.pain_score": ("transcript",)})

    def test_schema_invalid_answer_in_tool_phase_falls_back_to_structured_call(self):
        invalid = '{"proposed_fields":{"assessment.pain_score":"2점"},"evidence_refs":[],"needs_confirmation":false}'
        completions = _Completions(_completion(invalid), _completion(FINAL))
        adapter = OpenAiChatDraftGenerationAdapter(_client(completions), "qwen", gateway_client=McpGatewayClient(transport=_Gateway()))
        result = adapter.generate({"transcript": "식사 절반"}, sample_form_schema())
        self.assertEqual(len(completions.requests), 2)
        self.assertIn("response_format", completions.requests[1])
        # 버린 자유 형식 답은 최종 요청 대화에 넣지 않는다.
        self.assertEqual([message["role"] for message in completions.requests[1]["messages"]], ["system", "user"])
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})

    def test_tool_phase_runs_before_structured_output(self):
        gateway = _Gateway()
        call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="dictionary__lookup", arguments='{"term":"통증"}'))
        completions = _Completions(_completion(None, [call]), _completion("확인함"), _completion(FINAL))
        adapter = OpenAiChatDraftGenerationAdapter(_client(completions), "qwen3.5-27b", gateway_client=McpGatewayClient(transport=gateway))
        result = adapter.generate({"transcript": "통증"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(gateway.called, [{"name": "dictionary__lookup", "arguments": {"term": "통증"}}])
        first, _, final = completions.requests
        self.assertEqual(first["tools"][0], {"type": "function", "function": {"name": "dictionary__lookup", "description": "lookup", "parameters": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}}})
        self.assertNotIn("response_format", first)
        self.assertNotIn("tools", final)
        roles = [message["role"] for message in final["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(final["messages"][-1]["content"], FINAL_TURN_INSTRUCTION)
        self.assertEqual(final["messages"][3]["tool_call_id"], "call-1")

    def test_server_without_tool_support_falls_back_to_hints_only(self):
        error = Exception("auto tool choice requires --enable-auto-tool-choice")
        error.status_code = 400
        completions = _Completions(error, _completion(FINAL))
        adapter = OpenAiChatDraftGenerationAdapter(_client(completions), "gemma", gateway_client=McpGatewayClient(transport=_Gateway()))
        result = adapter.generate({"transcript": "x"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertNotIn("tools", completions.requests[-1])

    def test_invalid_output_is_rejected(self):
        adapter = OpenAiChatDraftGenerationAdapter(_client(_Completions(_completion("모르겠습니다"))), "small")
        with self.assertRaises(ValueError):
            adapter.generate({"transcript": "x"}, sample_form_schema())

    def test_json_in_an_invented_structure_is_not_an_empty_draft(self):
        invented = '{"간호일지": {"활력징후": {"체온": "38.2도"}}}'
        adapter = OpenAiChatDraftGenerationAdapter(_client(_Completions(_completion(invented))), "small")
        with self.assertRaisesRegex(ValueError, "did not follow the draft schema"):
            adapter.generate({"transcript": "x"}, sample_form_schema())


if __name__ == "__main__":
    unittest.main()
