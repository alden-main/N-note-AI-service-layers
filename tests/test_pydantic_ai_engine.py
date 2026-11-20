"""Pydantic AI 엔진(``N_NOTE_LLM_ENGINE=pydantic_ai``)이 native와 같은 요청을 보내는지 검증한다.

실제 AsyncOpenAI SDK가 가짜 HTTP 서버(httpx2 MockTransport)로 보내는 JSON 본문을 확인한다.
native 대응 테스트: test_local_model_providers(Chat)·test_openai_mcp_tool_loop(Responses).
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest.mock import patch

import httpx2
from openai import AsyncOpenAI

from tracing_support import all_attribute_text, by_name, recorded_spans

from n_note_ai_service_layers import prompts, settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.llm_completion import GenerationOptions, ToolExecutor
from n_note_ai_service_layers.adapters.openai_draft_generation import DraftGenerationAdapter
from n_note_ai_service_layers.adapters.pydantic_ai_completion import PydanticAiCompletion
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from support import LOOKUP_TOOL, FakeGateway, sample_form_schema

FINAL = {"proposed_fields": {"assessment.pain_score": 2}, "evidence_refs": [{"field_id": "assessment.pain_score", "refs": ["transcript"]}], "needs_confirmation": False}
PRIVATE = "환자가 오늘 흉통을 호소함"


class FakeServer:
    """OpenAI 호환 서버 흉내다. 받은 요청 본문을 ``requests``에 남기고 ``replies``를 차례로 돌려준다."""

    def __init__(self, *replies: dict[str, Any] | int) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.clients = 0

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            return httpx2.Response(reply, json={"error": {"message": "tool choice is not enabled", "type": "BadRequestError"}})
        return httpx2.Response(200, json=reply)

    def client(self) -> AsyncOpenAI:
        self.clients += 1
        transport = httpx2.MockTransport(self.handle)
        return AsyncOpenAI(base_url="http://local-llm/v1", api_key="EMPTY", max_retries=0, http_client=httpx2.AsyncClient(transport=transport))


def chat(content: str | None = None, *tool_calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
            for call_id, name, arguments in tool_calls
        ]
    return {
        "id": "chatcmpl", "object": "chat.completion", "created": 0, "model": "qwen",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def responses(text: str | None = None, *calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = [
        {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name, "arguments": json.dumps(arguments, ensure_ascii=False), "status": "completed"}
        for call_id, name, arguments in calls
    ]
    if text is not None:
        output.append({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]})
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-4o-mini", "status": "completed",
        "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }


def adapter(server: FakeServer, api: str = "chat", gateway: FakeGateway | None = None, max_tool_rounds: int = 2,
            options: GenerationOptions | None = None) -> DraftGenerationAdapter:
    tools = ToolExecutor(McpGatewayClient(transport=gateway), None, max_tool_rounds=max_tool_rounds) if gateway else None
    return DraftGenerationAdapter(PydanticAiCompletion(api, server.client, "qwen", options), "qwen", tools)


def roles(request: dict[str, Any]) -> list[str]:
    return [message["role"] for message in request["messages"]]


class ChatEngineTests(unittest.TestCase):
    def test_structured_output_without_tools_in_one_request(self):
        server = FakeServer(chat(json.dumps(FINAL)))
        result = adapter(server, options=GenerationOptions(512, 0.1, {"chat_template_kwargs": {"enable_thinking": False}})).generate(
            {"transcript": "통증"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(result.prompt_version, prompts.get().draft.version)
        (request,) = server.requests
        self.assertEqual(roles(request), ["system", "user"])
        system = request["messages"][0]["content"]
        self.assertTrue(system.startswith(prompts.get().draft.instructions))
        self.assertIn(prompts.get().structured_output.final_output, system)
        self.assertNotIn(prompts.get().draft.tool, system)
        self.assertEqual(request["response_format"]["json_schema"]["name"], "nursing_draft")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertNotIn("tools", request)
        self.assertEqual((request["max_tokens"], request["temperature"]), (512, 0.1))
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_block_and_code_fence_are_ignored(self):
        server = FakeServer(chat("<think>음</think>```json\n" + json.dumps(FINAL) + "\n```"))
        self.assertEqual(adapter(server).generate({"transcript": "x"}, sample_form_schema()).proposed_fields, {"assessment.pain_score": 2})

    def test_tool_phase_runs_before_structured_output(self):
        gateway = FakeGateway()
        server = FakeServer(chat(None, ("call-1", "dictionary__lookup", {"term": "통증"})), chat("확인함"), chat(json.dumps(FINAL)))
        result = adapter(server, gateway=gateway).generate({"transcript": "통증"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(gateway.called, [{"name": "dictionary__lookup", "arguments": {"term": "통증"}}])
        first, _, final = server.requests
        tool = first["tools"][0]["function"]
        self.assertEqual((tool["name"], tool["description"], tool["parameters"]["properties"]), ("dictionary__lookup", "lookup", LOOKUP_TOOL["inputSchema"]["properties"]))
        self.assertEqual(first["tool_choice"], "auto")
        self.assertIn(prompts.get().draft.tool, first["messages"][0]["content"])
        self.assertNotIn("response_format", first)
        self.assertNotIn("tools", final)
        self.assertIn("response_format", final)
        # 같은 양식의 요청은 system 메시지(접두부)가 같아 prefix caching이 적중한다.
        self.assertEqual(first["messages"][0], final["messages"][0])
        self.assertEqual(roles(final), ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(final["messages"][3]["tool_call_id"], "call-1")
        self.assertEqual(final["messages"][-1]["content"], prompts.get().structured_output.final_turn)

    def test_tool_phase_is_capped_by_tool_rounds(self):
        call = lambda index: chat(None, (f"call-{index}", "dictionary__lookup", {"term": "x"}))
        server = FakeServer(call(0), call(1), chat(json.dumps(FINAL)))
        gateway = FakeGateway()
        result = adapter(server, gateway=gateway, max_tool_rounds=2).generate({"transcript": "x"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(len(server.requests), 3)
        self.assertEqual(len(gateway.called), 2)
        self.assertIn("response_format", server.requests[-1])
        self.assertEqual(roles(server.requests[-1]), ["system", "user", "assistant", "tool", "assistant", "tool", "user"])

    def test_schema_valid_answer_in_tool_phase_skips_the_structured_call(self):
        server = FakeServer(chat("```json\n" + json.dumps(FINAL) + "\n```"))
        result = adapter(server, gateway=FakeGateway()).generate({"transcript": "식사 절반"}, sample_form_schema())
        self.assertEqual(len(server.requests), 1)
        self.assertIn("tools", server.requests[0])
        self.assertEqual(result.evidence_refs, {"assessment.pain_score": ("transcript",)})

    def test_schema_invalid_answer_in_tool_phase_falls_back_to_structured_call(self):
        invalid = {"proposed_fields": {"assessment.pain_score": "2점"}, "evidence_refs": [], "needs_confirmation": False}
        server = FakeServer(chat(json.dumps(invalid, ensure_ascii=False)), chat(json.dumps(FINAL)))
        result = adapter(server, gateway=FakeGateway()).generate({"transcript": "식사 절반"}, sample_form_schema())
        self.assertEqual(len(server.requests), 2)
        self.assertIn("response_format", server.requests[1])
        # 버린 자유 형식 답은 최종 요청 대화에 넣지 않고, user 연속을 피해 최종 출력 요청도 붙이지 않는다.
        self.assertEqual(roles(server.requests[1]), ["system", "user"])
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})

    def test_server_without_tool_support_falls_back_to_hints_only(self):
        server = FakeServer(400, chat(json.dumps(FINAL)))
        result = adapter(server, gateway=FakeGateway()).generate({"transcript": "x"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertIn("tools", server.requests[0])
        self.assertNotIn("tools", server.requests[1])
        self.assertEqual(roles(server.requests[1]), ["system", "user"])

    def test_invalid_output_is_rejected_without_a_retry(self):
        server = FakeServer(chat("not json"))
        with self.assertRaises(ValueError):
            adapter(server).generate({"transcript": "x"}, sample_form_schema())
        self.assertEqual(len(server.requests), 1)


class ResponsesEngineTests(unittest.TestCase):
    def test_tool_loop_ends_with_tool_choice_none_and_store_false(self):
        gateway = FakeGateway()
        server = FakeServer(responses(None, ("call-1", "dictionary__lookup", {"term": "통증"})), responses(json.dumps(FINAL)))
        result = adapter(server, "responses", gateway, max_tool_rounds=1).generate({"transcript": "통증"}, sample_form_schema())
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(gateway.called, [{"name": "dictionary__lookup", "arguments": {"term": "통증"}}])
        first, final = server.requests
        for request in (first, final):
            self.assertIs(request["store"], False)
            self.assertEqual(request["text"]["format"]["name"], "nursing_draft")
            self.assertTrue(request["instructions"].startswith(prompts.get().draft.instructions))
        self.assertIn(prompts.get().draft.tool, first["instructions"])
        self.assertIn(first.get("tool_choice"), (None, "auto"))  # API 기본값과 같다.
        self.assertEqual(final["tool_choice"], "none")
        kinds = [item.get("type", item.get("role")) for item in final["input"]]
        self.assertEqual(kinds[-2:], ["function_call", "function_call_output"])

    def test_without_tools_sends_one_structured_request(self):
        server = FakeServer(responses(json.dumps(FINAL)))
        adapter(server, "responses").generate({"transcript": "x"}, sample_form_schema())
        (request,) = server.requests
        self.assertNotIn("tools", request)
        self.assertNotIn(prompts.get().draft.tool, request["instructions"])


class EngineConcurrencyTests(unittest.TestCase):
    def test_one_engine_serves_parallel_forms_on_separate_threads(self):
        """여러 양식을 동시에 만들 때(generate_many) 엔진 하나를 스레드 여러 개가 함께 쓴다."""
        from concurrent.futures import ThreadPoolExecutor

        # 요청 순서가 스레드마다 섞이므로 응답은 순서가 아니라 요청 종류(도구 단계·구조화 출력)로 정한다.
        server = FakeServer()
        replies = {"tools": chat(None, ("call-1", "dictionary__lookup", {"term": "x"})), "final": chat(json.dumps(FINAL))}
        server.handle = lambda request: httpx2.Response(  # type: ignore[method-assign]
            200, json=replies["final" if "response_format" in json.loads(request.content) else "tools"])
        shared = adapter(server, gateway=FakeGateway(), max_tool_rounds=1)
        with recorded_spans() as spans, ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: shared.generate({"transcript": "x"}, sample_form_schema()), range(4)))
        self.assertEqual([result.proposed_fields for result in results], [{"assessment.pain_score": 2}] * 4)
        completions = by_name(spans, "n_note.llm.completion")
        self.assertEqual(len(completions), 4)
        # 양식마다 자기 completion span에 자기 도구 왕복 수가 남는다(다른 스레드의 실행과 섞이지 않는다).
        self.assertEqual([item.attributes["n_note.llm.tool_rounds"] for item in completions], [1, 1, 1, 1])


class EngineWiringTests(unittest.TestCase):
    def test_engine_setting_selects_pydantic_ai_for_drafts_and_form_extraction(self):
        env = {"N_NOTE_LLM_ENGINE": "pydantic_ai", "N_NOTE_LLM_BASE_URL": "http://local-llm/v1", "N_NOTE_LLM_MODEL": "qwen"}
        with patch.dict(os.environ, env):
            draft = model_providers.draft_generator(None)
            self.assertIsInstance(draft.completion, PydanticAiCompletion)
            self.assertEqual(draft.completion.api, "chat")
            self.assertIsInstance(model_providers.structured_completion(), PydanticAiCompletion)
        with patch.dict(os.environ, {"N_NOTE_LLM_ENGINE": ""}):
            self.assertEqual(settings.get().llm_engine, "native")

    def test_async_client_is_reused_within_a_thread(self):
        endpoint = settings.ModelEndpoint("qwen", "http://local-llm/v1")
        self.assertIs(model_providers.async_openai_client(endpoint, 30.0, 0), model_providers.async_openai_client(endpoint, 30.0, 0))


class EngineTracingTests(unittest.TestCase):
    def test_spans_cover_agent_model_and_gateway_calls_without_content(self):
        server = FakeServer(chat(None, ("call-1", "dictionary__lookup", {"term": "흉통"})), chat(json.dumps(FINAL)))
        with recorded_spans() as spans:
            adapter(server, gateway=FakeGateway(), max_tool_rounds=1).generate({"transcript": PRIVATE}, sample_form_schema())
        completion = by_name(spans, "n_note.llm.completion")[0]
        self.assertEqual(completion.attributes["n_note.llm.engine"], "pydantic_ai")
        self.assertEqual(completion.attributes["n_note.llm.tool_rounds"], 1)
        tool = by_name(spans, "n_note.llm.tool")[0]
        self.assertEqual(tool.context.trace_id, completion.context.trace_id)
        # Pydantic AI가 만든 모델 요청 span도 같은 추적 안에 있다.
        model_spans = [item for item in spans if item.name.startswith("chat ")]
        self.assertEqual(len(model_spans), 2)
        self.assertTrue(all(item.context.trace_id == completion.context.trace_id for item in model_spans))
        text = all_attribute_text(spans)
        self.assertNotIn(PRIVATE, text)
        self.assertNotIn("흉통", text)

    def test_content_is_recorded_when_capture_is_enabled(self):
        server = FakeServer(chat(json.dumps(FINAL)))
        with recorded_spans(capture_content=True) as spans:
            adapter(server).generate({"transcript": PRIVATE}, sample_form_schema())
        self.assertIn(PRIVATE, all_attribute_text(spans))


if __name__ == "__main__":
    unittest.main()
