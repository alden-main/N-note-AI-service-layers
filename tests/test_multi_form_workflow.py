"""음성 하나로 여러 양식 초안을 만드는 흐름(공통 단계 1회 + 양식별 병렬)을 검증한다."""

from __future__ import annotations

import json
import threading
import time
import unittest
from dataclasses import replace

from fastapi.testclient import TestClient

from n_note_ai_service_layers.api import app, draft_review_store, get_nursing_note_workflow, get_workflow_context_flow, store
from n_note_ai_service_layers.api.routes.workflow_jobs import requested_form_ids
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.llm_payload import LlmPayloadProjector
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow, NursingNoteWorkflowResult, WorkflowStageError
from support import SCORE_FIELD, sample_form_schema

AUDIO = AudioInput(b"a", "a.wav", "audio/wav", "trace")
BASE = sample_form_schema(SchemaStatus.ACTIVE)
SCHEMAS = {name: replace(BASE, schema_id=name, version=f"{name}/v1") for name in ("form-a", "form-b", "form-c")}


class _Stt:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def transcribe(self, audio):
        self.calls += 1
        if self.error:
            raise self.error
        return Transcript("통증 점수 2점", "fake", "fake-stt", "ko")


class _Generator:
    """``barrier``가 있으면 모든 양식이 동시에 들어와야 통과한다. 순차 실행이면 timeout으로 실패한다."""

    def __init__(self, barrier: threading.Barrier | None = None, failing: frozenset[str] = frozenset()) -> None:
        self.barrier = barrier
        self.failing = failing

    def generate(self, payload, schema):
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        if schema.schema_id in self.failing:
            raise RuntimeError("model down")
        return DraftResult({SCORE_FIELD: 2})


class _Gateway:
    class Client:
        def discover(self, request_id):
            return ()

    def __init__(self) -> None:
        self.gateway = self.Client()
        self.transcript_reviews = 0
        self.document_reviews = 0
        self._lock = threading.Lock()

    def verify(self, phase, text, trace_id, catalog=None):
        with self._lock:
            self.transcript_reviews += 1
        return GatewayTermVerificationResult(phase, "ok")

    def verify_document(self, phase, document, trace_id, catalog=None):
        with self._lock:
            self.document_reviews += 1
        return GatewayTermVerificationResult(phase, "ok")


class _Projector(LlmPayloadProjector):
    def project(self, snapshot, transcript, schema, term_review_hints=(), reference_time=None):
        return {"transcript": transcript, "term_review_hints": list(term_review_hints)}


def _workflow(stt=None, generator=None, gateway=None) -> NursingNoteWorkflow:
    return NursingNoteWorkflow(stt or _Stt(), generator or _Generator(), gateway or _Gateway(), projector=_Projector())


class GenerateManyTests(unittest.TestCase):
    def test_common_stages_run_once_and_forms_are_drafted_in_parallel(self) -> None:
        stt, gateway = _Stt(), _Gateway()
        forms = [(schema, object()) for schema in SCHEMAS.values()]
        done: list[int] = []
        stages: list[str] = []
        outcomes = _workflow(stt, _Generator(threading.Barrier(len(forms))), gateway).generate_many(
            AUDIO, forms, progress=stages.append, on_form_done=lambda index, _outcome: done.append(index), max_parallel=len(forms),
        )
        self.assertTrue(all(isinstance(item, NursingNoteWorkflowResult) for item in outcomes), outcomes)
        self.assertEqual(stt.calls, 1)
        self.assertEqual(gateway.transcript_reviews, 1)
        self.assertEqual(gateway.document_reviews, len(forms))
        self.assertEqual(sorted(done), [0, 1, 2])
        self.assertEqual(stages, ["stt", "redaction", "transcript_term_review", "draft_generation"])

    def test_one_failed_form_keeps_the_other_drafts(self) -> None:
        forms = [(schema, object()) for schema in SCHEMAS.values()]
        outcomes = _workflow(generator=_Generator(failing=frozenset({"form-b"}))).generate_many(AUDIO, forms, max_parallel=2)
        self.assertIsInstance(outcomes[0], NursingNoteWorkflowResult)
        self.assertIsInstance(outcomes[1], WorkflowStageError)
        self.assertEqual(outcomes[1].stage, "draft_generation")
        self.assertIsInstance(outcomes[2], NursingNoteWorkflowResult)

    def test_common_stage_failure_fails_every_form(self) -> None:
        done: list[int] = []
        with self.assertRaises(WorkflowStageError) as raised:
            _workflow(stt=_Stt(TimeoutError("slow"))).generate_many(
                AUDIO, [(schema, object()) for schema in SCHEMAS.values()], on_form_done=lambda index, _: done.append(index),
            )
        self.assertEqual(raised.exception.stage, "stt")
        self.assertEqual(done, [])


class RequestedFormIdsTests(unittest.TestCase):
    def test_repeated_and_comma_separated_ids_keep_order_without_duplicates(self) -> None:
        self.assertEqual(requested_form_ids(["form-a, form-b", "form-a", "form-c"]), ["form-a", "form-b", "form-c"])


class _Context:
    class Registry:
        def resolve_active(self, schema_id, purpose, version):
            if schema_id not in SCHEMAS:
                raise LookupError(schema_id)
            return SCHEMAS[schema_id]

    registry = Registry()

    def build(self, query):
        self.registry.resolve_active(query.schema_id, query.purpose, None)
        return object()


def _job_result(client: TestClient, accepted):
    assert accepted.status_code == 202, accepted.text
    for _ in range(500):
        job = client.get(accepted.headers["Location"]).json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("workflow job did not finish")


class MultiFormWorkflowApiTests(unittest.TestCase):
    def setUp(self) -> None:
        store.clear()
        self.client = TestClient(app)
        app.dependency_overrides[get_workflow_context_flow] = _Context

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def _post(self, data, workflow=None):
        if workflow is not None:
            app.dependency_overrides[get_nursing_note_workflow] = lambda: workflow
        return self.client.post(
            "/v1/nursing-note-workflows",
            files={"audio": ("note.wav", b"audio", "audio/wav")},
            data={"patient_id": "p", "encounter_id": "e", "purpose": BASE.purpose, **data},
        )

    def test_each_form_gets_its_own_draft_and_failures_stay_per_form(self) -> None:
        stt = _Stt()
        workflow = _workflow(stt, _Generator(failing=frozenset({"form-b"})))
        accepted = self._post({"schema_ids": ["form-a", "form-b"]}, workflow)
        self.assertEqual(accepted.json()["request"]["schema_ids"], ["form-a", "form-b"])
        self.assertEqual([form["status"] for form in accepted.json()["forms"]], ["queued", "queued"])

        job = _job_result(self.client, accepted)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(stt.calls, 1)
        first, second = job["forms"]
        self.assertEqual((first["schema_id"], first["status"]), ("form-a", "succeeded"))
        self.assertEqual((second["schema_id"], second["status"], second["error"]["code"]), ("form-b", "failed", "draft_generation_unavailable"))
        self.assertEqual(job["links"]["drafts"], [f"/v1/nursing-note-drafts/{first['draft_id']}"])
        self.assertIsNone(job["links"]["draft"])

        results = job["result"]["forms"]
        self.assertEqual(results[0]["result"]["draft_id"], first["draft_id"])
        self.assertEqual(results[0]["result"]["form"]["schema_id"], "form-a")
        self.assertIsNone(results[1]["result"])
        self.assertEqual(results[1]["error"]["code"], "draft_generation_unavailable")
        # 초안 조회 API는 양식 레지스트리를 거치므로(테스트 양식은 미등록) 저장소에서 직접 확인한다.
        self.assertEqual(draft_review_store.get(first["draft_id"]).schema_id, "form-a")

    def test_job_fails_when_every_form_fails(self) -> None:
        workflow = _workflow(generator=_Generator(failing=frozenset(SCHEMAS)))
        job = _job_result(self.client, self._post({"schema_ids": "form-a,form-c"}, workflow))
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["code"], "all_forms_failed")
        self.assertEqual([form["error"]["code"] for form in job["forms"]], ["draft_generation_unavailable"] * 2)

    def test_events_report_each_form_as_it_finishes(self) -> None:
        workflow = _workflow()
        accepted = self._post({"schema_ids": ["form-a", "form-b"]}, workflow)
        job = _job_result(self.client, accepted)
        with self.client.stream("GET", job["links"]["events"]) as response:
            events = [line for line in response.iter_lines() if line.startswith("data: ")]
        final = json.loads(events[-1].removeprefix("data: "))
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(len(final["links"]["drafts"]), 2)

    def test_schema_ids_cannot_be_combined_with_schema_id(self) -> None:
        response = self._post({"schema_ids": ["form-a"], "schema_id": "form-b"}, _workflow())
        self.assertEqual(response.status_code, 422)

    def test_unknown_form_is_rejected_before_the_job_is_queued(self) -> None:
        response = self._post({"schema_ids": ["form-a", "missing"]}, _workflow())
        self.assertEqual(response.status_code, 422)


class GatewayBatchSplitTests(unittest.TestCase):
    def test_calls_above_the_gateway_limit_are_split_and_keep_order(self) -> None:
        batches: list[int] = []
        tool = {"name": "t", "description": "d", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}

        def transport(url, body, headers, timeout):
            message = json.loads(body)
            if url.endswith("/v1/tools/batch-call"):
                batches.append(len(message["calls"]))
                results = [{"structuredContent": {"echo": call["arguments"]["text"]}, "isError": False} for call in message["calls"]]
                return 200, {}, json.dumps({"results": results}).encode()
            result = {"protocolVersion": "2025-06-18"} if message["method"] == "initialize" else {"tools": [tool]}
            return 200, {}, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode()

        client = McpGatewayClient(transport=transport, batch_max_calls=2)
        client.discover()
        outcomes = client.call_tools([("t", {"text": str(index)}) for index in range(5)])
        self.assertEqual(sorted(batches), [1, 2, 2])
        self.assertEqual([item["structuredContent"]["echo"] for item in outcomes], ["0", "1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()
