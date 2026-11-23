from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from llm_support import FakeServer, adapter, chat
from n_note_ai_service_layers import prompts, settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.llm_completion import GenerationOptions
from n_note_ai_service_layers.adapters.pydantic_ai_completion import PydanticAiCompletion
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from support import LOOKUP_TOOL, sample_form_schema
from support import FakeGateway as _Gateway

FINAL_OUTPUT_INSTRUCTION = prompts.get().structured_output.final_output
FINAL_TURN_INSTRUCTION = prompts.get().structured_output.final_turn

FINAL = '{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":{"assessment.pain_score":["transcript"]},"needs_confirmation":false}'
LOOKUP = lambda index="1": (f"call-{index}", "dictionary__lookup", {"term": "통증"})


class ModelSettingsTests(unittest.TestCase):
    def test_settings_are_reused_until_relevant_environment_changes(self):
        with patch.dict(os.environ, {"N_NOTE_LLM_MODEL": "a"}):
            first = settings.get()
            self.assertIs(settings.get(), first)
            with patch.dict(os.environ, {"UNRELATED_VARIABLE": "x"}):
                self.assertIs(settings.get(), first)
            with patch.dict(os.environ, {"N_NOTE_LLM_MODEL": "b"}):
                self.assertEqual(settings.get().llm_model, "b")
            self.assertIs(settings.get(), first)

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
        self.assertIsInstance(generator.completion, PydanticAiCompletion)
        self.assertEqual(generator.completion.api, "chat")
        self.assertIsNone(generator.tools.gateway)
        self.assertEqual(str(generator.completion.client_factory().base_url), "http://local-llm:8000/v1/")
        # LLM만 로컬이고 STT는 OpenAI인 혼합 구성이다.
        self.assertEqual(stt.provider, "openai")
        self.assertFalse(stt.normalize)

    def test_factory_keeps_responses_api_for_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            generator = model_providers.draft_generator(None)
        self.assertEqual(generator.completion.api, "responses")


class ChatDraftAdapterTests(unittest.TestCase):
    def test_structured_output_without_tools_in_one_request(self):
        server = FakeServer(chat("<think>검토</think>```json\n" + FINAL + "\n```"))
        options = GenerationOptions(max_tokens=1024, extra_body={"top_k": 20})
        result = adapter(server, model="medgemma-27b", options=options).generate({"transcript": "식사 절반 섭취"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(result.model_version, "medgemma-27b")
        (request,) = server.requests
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertEqual((request["top_k"], request["max_tokens"]), (20, 1024))
        # 고정 지침·schema는 system에, 요청별 payload만 user에 둔다(prefix cache 적중, user 메시지 연속 없음).
        system, user = request["messages"]
        self.assertEqual((system["role"], user["role"]), ("system", "user"))
        self.assertIn(FINAL_OUTPUT_INSTRUCTION, system["content"])
        self.assertIn('"proposed_fields"', system["content"])
        self.assertEqual(json.loads(user["content"]), {"transcript": "식사 절반 섭취"})
        self.assertNotIn("tools", request)

    def test_system_prefix_is_identical_across_requests_for_the_same_form(self):
        server = FakeServer(chat(FINAL), chat(FINAL))
        drafts = adapter(server, model="m")
        drafts.generate({"transcript": "a"}, sample_form_schema())
        drafts.generate({"transcript": "b"}, sample_form_schema())
        first, second = (request["messages"][0] for request in server.requests)
        self.assertEqual(first, second)

    def test_schema_valid_answer_in_tool_phase_skips_the_structured_call(self):
        answer = '```json\n{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":[{"field_id":"assessment.pain_score","refs":["transcript"]}],"needs_confirmation":false}\n```'
        server = FakeServer(chat(answer))
        result = adapter(server, gateway=_Gateway()).generate({"transcript": "식사 절반"}, sample_form_schema())
        self.assertEqual(len(server.requests), 1)
        self.assertIn("tools", server.requests[0])
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(result.evidence_refs, {"assessment.pain_score": ("transcript",)})

    def test_schema_invalid_answer_in_tool_phase_falls_back_to_structured_call(self):
        invalid = '{"proposed_fields":{"assessment.pain_score":"2점"},"evidence_refs":[],"needs_confirmation":false}'
        server = FakeServer(chat(invalid), chat(FINAL))
        result = adapter(server, gateway=_Gateway()).generate({"transcript": "식사 절반"}, sample_form_schema())
        self.assertEqual(len(server.requests), 2)
        self.assertIn("response_format", server.requests[1])
        # 버린 자유 형식 답은 최종 요청 대화에 넣지 않는다.
        self.assertEqual([message["role"] for message in server.requests[1]["messages"]], ["system", "user"])
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})

    def test_tool_phase_runs_before_structured_output(self):
        gateway = _Gateway()
        server = FakeServer(chat(None, LOOKUP()), chat("확인함"), chat(FINAL))
        result = adapter(server, model="qwen3.5-27b", gateway=gateway).generate({"transcript": "통증"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(gateway.called, [{"name": "dictionary__lookup", "arguments": {"term": "통증"}}])
        first, _, final = server.requests
        tool = first["tools"][0]["function"]
        self.assertEqual((first["tools"][0]["type"], tool["name"], tool["description"], tool["parameters"]),
                         ("function", "dictionary__lookup", "lookup", LOOKUP_TOOL["inputSchema"]))
        self.assertNotIn("response_format", first)
        self.assertNotIn("tools", final)
        roles = [message["role"] for message in final["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(final["messages"][-1]["content"], FINAL_TURN_INSTRUCTION)
        self.assertEqual(final["messages"][3]["tool_call_id"], "call-1")

    def test_tool_phase_is_capped_by_tool_rounds(self):
        """도구를 계속 요청해도 도구 단계는 ``max_tool_rounds``번이고, 이어서 구조화 출력 호출 한 번으로 끝난다."""
        gateway = _Gateway()
        server = FakeServer(chat(None, LOOKUP("1")), chat(None, LOOKUP("2")), chat(FINAL))
        result = adapter(server, gateway=gateway, max_tool_rounds=2).generate({"transcript": "x"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(len(server.requests), 3)
        self.assertEqual(len(gateway.called), 2)
        self.assertIn("response_format", server.requests[-1])

    def test_tool_round_settings_reach_the_adapter(self):
        env = {"N_NOTE_LLM_MAX_TOOL_ROUNDS": "0", "N_NOTE_LLM_MAX_TOOL_CALLS": "3", "OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            drafts = model_providers.draft_generator(McpGatewayClient(transport=_Gateway()))
        self.assertEqual((drafts.tools.max_tool_rounds, drafts.tools.max_tool_calls), (0, 3))

    def test_server_without_tool_support_falls_back_to_hints_only(self):
        server = FakeServer(400, chat(FINAL))
        result = adapter(server, model="gemma", gateway=_Gateway()).generate({"transcript": "x"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertNotIn("tools", server.requests[-1])

    def test_invalid_output_is_rejected(self):
        with self.assertRaises(ValueError):
            adapter(FakeServer(chat("모르겠습니다")), model="small").generate({"transcript": "x"}, sample_form_schema())

    def test_json_in_an_invented_structure_is_not_an_empty_draft(self):
        invented = '{"간호일지": {"활력징후": {"체온": "38.2도"}}}'
        with self.assertRaisesRegex(ValueError, "did not follow the draft schema"):
            adapter(FakeServer(chat(invented)), model="small").generate({"transcript": "x"}, sample_form_schema())


if __name__ == "__main__":
    unittest.main()
