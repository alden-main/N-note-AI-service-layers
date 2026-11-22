from __future__ import annotations

import json
import logging
import unittest
import unittest.mock
from types import SimpleNamespace

from tracing_support import all_attribute_text, by_name, recorded_spans

from n_note_ai_service_layers import observability, prompts
from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiChatDraftGenerationAdapter
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from support import FakeGateway, sample_form_schema

PRIVATE = "환자가 오늘 흉통을 호소함"
FINAL = '{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":[{"field_id":"assessment.pain_score","refs":["transcript"]}],"needs_confirmation":false}'


def _completion(content=None, tool_calls=None, usage=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))], usage=usage)


class _Completions:
    def __init__(self, *responses):
        self.responses = list(responses)

    def create(self, **kwargs):
        return self.responses.pop(0)


def _tool_call():
    return SimpleNamespace(id="call-1", function=SimpleNamespace(name="dictionary__lookup", arguments='{"term":"흉통"}'))


def _generate_with_one_tool_round():
    completions = _Completions(
        _completion(None, [_tool_call()], SimpleNamespace(prompt_tokens=100, completion_tokens=10)),
        _completion("not json", None, SimpleNamespace(prompt_tokens=120, completion_tokens=5)),
        _completion(FINAL, None, SimpleNamespace(prompt_tokens=130, completion_tokens=40)),
    )
    adapter = OpenAiChatDraftGenerationAdapter(
        SimpleNamespace(chat=SimpleNamespace(completions=completions)), "local-llm",
        gateway_client=McpGatewayClient(transport=FakeGateway()), max_tool_rounds=2,
    )
    return adapter.generate({"transcript": PRIVATE}, sample_form_schema())


class StageSpanTests(unittest.TestCase):
    def test_stage_span_has_metrics_and_only_the_error_type(self):
        logger = observability.get_logger("test")
        with recorded_spans() as spans:
            with observability.stage(logger, "trace-1", "draft_generation", schema_id="form-a") as metrics:
                metrics["fields"] = 3
            with self.assertRaises(ValueError):
                with observability.stage(logger, "trace-1", "stt"):
                    raise ValueError(PRIVATE)
        ok, failed = by_name(spans, "n_note.stage.draft_generation")[0], by_name(spans, "n_note.stage.stt")[0]
        self.assertEqual(ok.attributes["n_note.trace_id"], "trace-1")
        self.assertEqual(ok.attributes["n_note.schema_id"], "form-a")
        self.assertEqual(ok.attributes["n_note.fields"], 3)
        self.assertEqual(failed.attributes["error.type"], "ValueError")
        self.assertEqual(failed.status.description, "ValueError")
        self.assertEqual(failed.events, ())  # 예외 메시지(본문)를 이벤트로 남기지 않는다.
        self.assertNotIn(PRIVATE, all_attribute_text(spans))

    def test_log_lines_carry_the_otel_trace_id_inside_a_span(self):
        records = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = observability.get_logger("test.trace_id")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            with recorded_spans() as spans:
                with observability.span("outer"):
                    observability.log_event(logger, "inside")
        finally:
            logger.removeHandler(handler)
        self.assertEqual(records[0].fields["otel_trace_id"], format(spans[0].context.trace_id, "032x"))

    def test_pool_work_stays_in_the_submitting_trace(self):
        pool = observability.ContextThreadPoolExecutor(max_workers=1)
        try:
            with recorded_spans() as spans:
                with observability.span("parent"):
                    pool.submit(self._child).result()
        finally:
            pool.shutdown()
        parent = by_name(spans, "n_note.parent")[0]
        child = by_name(spans, "n_note.pooled")[0]
        self.assertEqual(child.context.trace_id, parent.context.trace_id)
        self.assertEqual(child.parent.span_id, parent.context.span_id)

    @staticmethod
    def _child():
        with observability.span("pooled"):
            pass


class LlmSpanTests(unittest.TestCase):
    def test_chat_completion_records_each_round_and_tool_call_without_content(self):
        with recorded_spans() as spans:
            draft = _generate_with_one_tool_round()
        self.assertEqual(draft.proposed_fields, {"assessment.pain_score": 2})
        completion = by_name(spans, "n_note.llm.completion")[0]
        requests = sorted(by_name(spans, "n_note.llm.request"), key=lambda item: item.start_time)
        tool = by_name(spans, "n_note.llm.tool")[0]
        self.assertEqual(completion.attributes["n_note.llm.engine"], "native")
        self.assertEqual(completion.attributes["n_note.prompt.draft_version"], prompts.get().draft.version)
        self.assertEqual(completion.attributes["n_note.llm.tool_rounds"], 2)
        self.assertEqual(completion.attributes["n_note.llm.tool_calls"], 1)
        self.assertEqual([item.attributes["n_note.phase"] for item in requests], ["tools", "tools", "structured_output"])
        self.assertEqual(requests[0].attributes["gen_ai.usage.input_tokens"], 100)
        self.assertEqual(requests[0].attributes["n_note.llm.tool_calls_requested"], 1)
        self.assertEqual(tool.attributes["gen_ai.tool.name"], "dictionary__lookup")
        for item in (*requests, tool):
            self.assertEqual(item.parent.span_id, completion.context.span_id)
        text = all_attribute_text(spans)
        self.assertNotIn(PRIVATE, text)
        self.assertNotIn("흉통", text)
        self.assertNotIn(prompts.get().draft.instructions[:40], text)

    def test_content_is_recorded_only_when_capture_is_enabled(self):
        with recorded_spans(capture_content=True) as spans:
            _generate_with_one_tool_round()
        completion = by_name(spans, "n_note.llm.completion")[0]
        self.assertIn(PRIVATE, completion.attributes["n_note.llm.input"])
        self.assertTrue(completion.attributes["n_note.llm.instructions"].startswith(prompts.get().draft.instructions))
        self.assertEqual(json.loads(completion.attributes["n_note.llm.output"])["proposed_fields"], {"assessment.pain_score": 2})
        tool = by_name(spans, "n_note.llm.tool")[0]
        self.assertIn("흉통", tool.attributes["gen_ai.tool.call.arguments"])

    def test_tracing_stays_off_unless_enabled(self):
        self.assertFalse(observability.configure_tracing(False, capture_content=True))
        self.assertFalse(observability.capture_content())

    def test_api_requests_are_traced_without_health_checks_operation_spans_or_query_values(self):
        from fastapi.testclient import TestClient
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from n_note_ai_service_layers.api.app import create_app

        with recorded_spans() as spans, TestClient(create_app()) as client:
            client.get("/healthz")
            client.get("/openapi.json")
            client.get("/v1/templates?patient_id=P-12345")
        self.assertEqual([item.name for item in spans if item.attributes.get("url.path") in observability.UNTRACED_PATHS], [])
        self.assertFalse([item for item in spans if item.name.startswith("fastapi.")])  # operation_spans 끔
        request = [item for item in spans if item.attributes.get("url.path") == "/v1/templates"]
        self.assertEqual(len(request), 1)
        # 쿼리 값 가림은 configure_tracing이 붙이는 processor다(테스트 provider와 따로 확인).
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(observability._redact_url_query())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        provider.get_tracer("t").start_span("GET", attributes={"url.query": "patient_id=P-12345&page=2"}).end()
        self.assertEqual(exporter.get_finished_spans()[0].attributes["url.query"], "patient_id=REDACTED&page=REDACTED")

    def test_enabled_tracing_without_an_endpoint_stays_off(self):
        # compose의 ${OTEL_EXPORTER_OTLP_ENDPOINT:-}처럼 빈 값이면 켜지 않는다(localhost로 조용히 보내지 않음).
        with unittest.mock.patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_ENDPOINT": " "}):
            self.assertFalse(observability.configure_tracing(True, capture_content=True))
        self.assertFalse(observability.capture_content())


if __name__ == "__main__":
    unittest.main()
