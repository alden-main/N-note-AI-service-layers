"""운영 안정성 개선(비차단 실행·영속화·동시 수정·인증 사용자·시간대·비식별화·Gateway 캐시) 회귀 테스트다."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.document_store import InMemoryDocumentStore, SqliteDocumentStore, VersionConflict
from n_note_ai_service_layers.api import app, draft_review_store, get_nursing_note_workflow, get_workflow_context_flow, template_registry
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import (
    GatewaySuggestion,
    GatewayTermOccurrence,
    GatewayTermVerificationResult,
)
from n_note_ai_service_layers.services.draft_review import DraftReviewStore, DraftRevisionConflict
from n_note_ai_service_layers.services.llm_payload import LlmPayloadProjector
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow, NursingNoteWorkflowResult, WorkflowStageError
from n_note_ai_service_layers.services.schema_review import FormSchemaRegistry
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor

SCHEMA = FormSchema(
    "hardening-test", "v1", "HARDENING_TEST",
    (FormSection("s", (FormField("assessment", "string", required=True),)),),
    status=SchemaStatus.ACTIVE,
)


def _ensure_schema() -> None:
    if not any(item.schema_id == SCHEMA.schema_id for item in template_registry.list_versions()):
        template_registry.add(SCHEMA)


class _Context:
    class Registry:
        def resolve_active(self, schema_id, purpose, version): return SCHEMA
    registry = Registry()
    def build(self, query): return object()


class WorkflowApiTests(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_schema()
        draft_review_store.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def _post(self, workflow, **headers):
        app.dependency_overrides[get_workflow_context_flow] = _Context
        app.dependency_overrides[get_nursing_note_workflow] = lambda: workflow
        return self.client.post(
            "/v1/nursing-note-workflows",
            files={"audio": ("note.wav", b"audio", "audio/wav")},
            data={"patient_id": "PATIENT-0001", "encounter_id": "ENC-0001", "purpose": SCHEMA.purpose, "schema_id": SCHEMA.schema_id},
            headers=headers,
        )

    def _wait(self, accepted) -> dict:
        self.assertEqual(accepted.status_code, 202, accepted.text)
        for _ in range(500):
            job = self.client.get(accepted.headers["Location"]).json()
            if job["status"] in ("succeeded", "failed"):
                return job
            time.sleep(0.01)
        self.fail("workflow job did not finish")

    def test_workflow_runs_outside_the_event_loop_and_receives_known_identifiers(self) -> None:
        seen = {}

        class Workflow:
            def generate(self, snapshot, audio, schema, known_identifiers=(), progress=None):
                try:
                    asyncio.get_running_loop()
                    seen["in_event_loop"] = True
                except RuntimeError:
                    seen["in_event_loop"] = False
                seen["identifiers"] = tuple(known_identifiers)
                return NursingNoteWorkflowResult(
                    Transcript("통증 호소", "fake", "fake"), DraftResult({"assessment": "통증 호소함"}),
                    GatewayTermVerificationResult("transcript", "ok"), GatewayTermVerificationResult("draft", "ok"),
                    llm_applied_term_hints=({"source_text": "금새", "applied_text": "금세", "category": "standard_korean"},),
                )

        accepted = self._post(Workflow())
        self.assertEqual(accepted.json()["status"], "queued")
        job = self._wait(accepted)
        self.assertEqual(job["status"], "succeeded")
        self.assertFalse(seen["in_event_loop"])
        self.assertEqual(seen["identifiers"], ("PATIENT-0001", "ENC-0001"))
        self.assertEqual(job["result"]["llm_applied_term_hints"][0]["applied_text"], "금세")
        self.assertEqual(job["links"]["draft"], f"/v1/nursing-note-drafts/{job['result']['draft_id']}")
        draft = self.client.get(job["links"]["draft"]).json()
        self.assertEqual(draft["llm_applied_term_hints"][0]["source_text"], "금새")

    def test_stage_failures_are_reported_by_stage(self) -> None:
        class Workflow:
            def generate(self, snapshot, audio, schema, known_identifiers=(), progress=None):
                raise WorkflowStageError("draft_generation")

        job = self._wait(self._post(Workflow()))
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["code"], "draft_generation_unavailable")
        self.assertIsNone(job["result"])

    def _draft(self) -> str:
        class Workflow:
            def generate(self, snapshot, audio, schema, known_identifiers=(), progress=None):
                return NursingNoteWorkflowResult(
                    Transcript("통증 호소", "fake", "fake"), DraftResult({"assessment": "통증 호소함"}),
                    GatewayTermVerificationResult("transcript", "ok"), GatewayTermVerificationResult("draft", "ok"),
                )
        return self._wait(self._post(Workflow()))["result"]["draft_id"]

    def test_progress_stages_idempotency_and_sse(self) -> None:
        release = threading.Event()

        class Workflow:
            def generate(self, snapshot, audio, schema, known_identifiers=(), progress=None):
                progress("stt")
                release.wait(5)
                progress("draft_generation")
                return NursingNoteWorkflowResult(
                    Transcript("통증 호소", "fake", "fake"), DraftResult({"assessment": "통증 호소함"}),
                    GatewayTermVerificationResult("transcript", "ok"), GatewayTermVerificationResult("draft", "ok"),
                )

        accepted = self._post(Workflow(), **{"Idempotency-Key": "upload-123"})
        job_id = accepted.json()["job_id"]
        for _ in range(200):
            running = self.client.get(f"/v1/nursing-note-jobs/{job_id}").json()
            if running["stage"] == "stt":
                break
            time.sleep(0.01)
        self.assertEqual((running["status"], running["stage"]), ("running", "stt"))
        replay = self._post(Workflow(), **{"Idempotency-Key": "upload-123"})
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(replay.json()["job_id"], job_id)
        release.set()
        with self.client.stream("GET", f"/v1/nursing-note-jobs/{job_id}/events") as stream:
            events = [line.removeprefix("event: ") for line in stream.iter_lines() if line.startswith("event: ")]
        self.assertEqual(events[-1], "succeeded")
        job = self.client.get(f"/v1/nursing-note-jobs/{job_id}").json()
        self.assertEqual(job["completed_stages"], ["stt", "draft_generation", "saving_draft"])

    def test_unknown_job_is_404(self) -> None:
        self.assertEqual(self.client.get("/v1/nursing-note-jobs/missing").status_code, 404)
        self.assertEqual(self.client.get("/v1/nursing-note-jobs/missing/events").status_code, 404)

    def test_full_queue_is_rejected_with_retry_after(self) -> None:
        from n_note_ai_service_layers.api import job_runner
        with patch.object(job_runner, "max_pending", 0):
            response = self._post(object())
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "5")

    def test_stale_revision_is_rejected_with_conflict(self) -> None:
        draft_id = self._draft()
        first = self.client.post(f"/v1/nursing-note-drafts/{draft_id}/revisions", json={"edited_by": "nurse-1", "edited_fields": {"assessment": "A"}, "expected_revision": 1})
        self.assertEqual(first.status_code, 200)
        stale = self.client.post(f"/v1/nursing-note-drafts/{draft_id}/revisions", json={"edited_by": "nurse-2", "edited_fields": {"assessment": "B"}, "expected_revision": 1})
        self.assertEqual(stale.status_code, 409)
        finalize = self.client.post(f"/v1/nursing-note-drafts/{draft_id}/finalize", json={"finalized_by": "nurse-1", "expected_revision": 1})
        self.assertEqual(finalize.status_code, 409)

    def test_authenticated_user_header_overrides_and_guards_actor(self) -> None:
        draft_id = self._draft()
        mismatch = self.client.post(
            f"/v1/nursing-note-drafts/{draft_id}/revisions",
            json={"edited_by": "someone-else", "edited_fields": {"assessment": "A"}},
            headers={"X-Authenticated-User": "nurse-1"},
        )
        self.assertEqual(mismatch.status_code, 403)
        ok = self.client.post(
            f"/v1/nursing-note-drafts/{draft_id}/revisions",
            json={"edited_fields": {"assessment": "A"}},
            headers={"X-Authenticated-User": "nurse-1"},
        )
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["edited_by"], "nurse-1")
        with patch.dict(os.environ, {"N_NOTE_REQUIRE_AUTHENTICATED_USER": "true"}):
            missing = self.client.post(f"/v1/nursing-note-drafts/{draft_id}/revisions", json={"edited_by": "nurse-1", "edited_fields": {"assessment": "B"}})
        self.assertEqual(missing.status_code, 401)


class PersistenceTests(unittest.TestCase):
    def test_sqlite_store_survives_reopen_and_rejects_stale_versions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.sqlite3"
            first = SqliteDocumentStore(path)
            self.assertEqual(first.put("c", "k", {"a": 1}, expected_version=0), 1)
            with self.assertRaises(VersionConflict):
                first.put("c", "k", {"a": 2}, expected_version=0)
            second = SqliteDocumentStore(path)  # 재시작 또는 다른 worker
            self.assertEqual(second.get("c", "k").payload, {"a": 1})
            self.assertEqual(second.put("c", "k", {"a": 2}, expected_version=1), 2)
            with self.assertRaises(VersionConflict):
                first.put("c", "k", {"a": 3}, expected_version=1)

    def test_draft_review_store_is_shared_through_the_document_store(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.sqlite3"
            created = DraftReviewStore(SqliteDocumentStore(path)).create(SCHEMA, "전사문", {"assessment": "통증"}, (), "t", patient_id="p")
            other_worker = DraftReviewStore(SqliteDocumentStore(path))
            loaded = other_worker.get(created.draft_id)
            self.assertEqual(loaded, created)
            other_worker.revise(created.draft_id, SCHEMA, {"assessment": "통증 없음"}, {}, "nurse-1", "t", expected_revision=1)
            with self.assertRaises(DraftRevisionConflict):
                DraftReviewStore(SqliteDocumentStore(path)).revise(created.draft_id, SCHEMA, {"assessment": "x"}, {}, "nurse-2", "t", expected_revision=1)

    def test_expired_drafts_are_purged(self) -> None:
        store = InMemoryDocumentStore()
        drafts = DraftReviewStore(store, retention_hours=1)
        old = drafts.create(SCHEMA, "전사문", {"assessment": "a"}, (), "t")
        stored = store.get("drafts", old.draft_id)
        store._items["drafts"][old.draft_id] = stored.__class__(stored.key, stored.version, stored.payload, stored.updated_at - 7200)
        drafts.create(SCHEMA, "전사문", {"assessment": "b"}, (), "t")
        with self.assertRaises(LookupError):
            drafts.get(old.draft_id)

    def test_template_changes_persist_over_deployed_definitions(self) -> None:
        store = InMemoryDocumentStore()
        deployed_v1 = FormSchema("daily", "v1", "DAILY", (FormSection("s", (FormField("a", "string"),)),), status=SchemaStatus.ACTIVE)
        first = FormSchemaRegistry(store)
        first.add_base(deployed_v1)
        first.add(FormSchema("daily", "v2", "DAILY", (FormSection("s", (FormField("a", "string", required=True, label="A"),)),), status=SchemaStatus.APPROVED))
        first.activate("daily", "v2")
        restarted = FormSchemaRegistry(store)
        self.assertEqual(restarted.add_base(deployed_v1).status, SchemaStatus.APPROVED)
        self.assertEqual(restarted.resolve_active("daily", "DAILY").version, "v2")


class TimezoneAndPayloadTests(unittest.TestCase):
    def test_reference_time_uses_ward_timezone_not_container_timezone(self) -> None:
        with patch.dict(os.environ, {"N_NOTE_TIMEZONE": "Asia/Seoul"}):
            self.assertEqual(datetime.now(settings.local_timezone()).utcoffset(), timedelta(hours=9))
            from n_note_ai_service_layers.services.clock import now_local
            self.assertEqual(now_local().utcoffset(), timedelta(hours=9))


class RedactionTests(unittest.TestCase):
    def test_spoken_names_phone_numbers_and_request_identifiers(self) -> None:
        text = "환자 성함은 홍길동이고 김영희님 보호자 연락처 공일공 일이삼사 오육칠팔, 등록번호 PATIENT-0001 확인"
        result = TranscriptRedactor().redact(text, ("PATIENT-0001",))
        self.assertNotIn("홍길동", result.text)
        self.assertNotIn("김영희", result.text)
        self.assertNotIn("일이삼사", result.text)
        self.assertNotIn("PATIENT-0001", result.text)
        self.assertEqual(result.replacement_count, 4)

    def test_clinical_phrases_and_titles_are_not_redacted(self) -> None:
        text = "선생님께 보고함. 이모님 보호 중. 정상 환자, 이번 환자, 고위험 환자. 이송 후 오른쪽 하지 부종"
        self.assertEqual(TranscriptRedactor().redact(text).text, text)


class WorkflowHintTrackingTests(unittest.TestCase):
    def test_llm_applied_hint_is_recorded(self) -> None:
        schema = FormSchema("t", "v1", "T", (FormSection("s", (FormField("note", "string"),)),), status=SchemaStatus.ACTIVE)
        occurrence = GatewayTermOccurrence("transcript", "korean", 0, 2, "금새", "금새", (GatewaySuggestion("금세"),), category="standard_korean")

        class Stt:
            def transcribe(self, audio): return Transcript("금새 호전", "fake", "fake")

        class Generator:
            def generate(self, payload, schema): return DraftResult({"note": "금세 호전됨"})

        class Gateway:
            class Client:
                def discover(self, request_id): return ()
            gateway = Client()
            def verify(self, phase, text, trace_id, catalog=None):
                return GatewayTermVerificationResult(phase, "ok", (occurrence,))
            def verify_document(self, phase, document, trace_id, catalog=None):
                return GatewayTermVerificationResult(phase, "ok")

        class Projector(LlmPayloadProjector):
            def project(self, snapshot, transcript, schema, term_review_hints=(), reference_time=None):
                return {"transcript": transcript, "term_review_hints": list(term_review_hints)}

        result = NursingNoteWorkflow(Stt(), Generator(), Gateway(), projector=Projector()).generate(
            object(), AudioInput(b"a", "a.wav", "audio/wav", "trace"), schema,
        )
        self.assertEqual(result.llm_applied_term_hints, ({"source_text": "금새", "applied_text": "금세", "category": "standard_korean"},))

    def test_stt_failure_is_a_stage_error(self) -> None:
        class Stt:
            def transcribe(self, audio): raise TimeoutError("slow")

        with self.assertRaises(WorkflowStageError) as raised:
            NursingNoteWorkflow(Stt(), None, None).generate(object(), AudioInput(b"a", "a.wav", "audio/wav", "trace"), SCHEMA)
        self.assertEqual(raised.exception.stage, "stt")


class GatewayCatalogCacheTests(unittest.TestCase):
    def test_catalog_is_reused_within_ttl(self) -> None:
        methods = []

        def transport(url, body, headers, timeout):
            request = json.loads(body)
            methods.append(request["method"])
            result = {"protocolVersion": "2025-06-18"} if request["method"] == "initialize" else {"tools": []}
            return 200, {}, json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()

        client = McpGatewayClient("http://gateway/mcp", 1.0, transport, catalog_ttl_seconds=60)
        client.discover("a")
        client.discover("b")
        self.assertEqual(methods, ["initialize", "tools/list"])
        client.invalidate_catalog()
        client.discover("c")
        self.assertEqual(len(methods), 4)


class ModelProviderPolicyTests(unittest.TestCase):
    def test_local_stt_can_be_required(self) -> None:
        with patch.dict(os.environ, {"N_NOTE_REQUIRE_LOCAL_STT": "true", "N_NOTE_STT_BASE_URL": ""}):
            with self.assertRaises(model_providers.LocalSttRequired):
                model_providers.speech_to_text_provider()
        with patch.dict(os.environ, {"N_NOTE_REQUIRE_LOCAL_STT": "true", "N_NOTE_STT_BASE_URL": "http://local-stt:8000/v1"}):
            self.assertEqual(model_providers.speech_to_text_provider()._provider, "openai-compatible")

    def test_model_clients_have_bounded_timeouts(self) -> None:
        with patch.dict(os.environ, {"N_NOTE_LLM_BASE_URL": "http://local-llm:8000/v1", "N_NOTE_LLM_TIMEOUT_SECONDS": "42", "N_NOTE_MODEL_MAX_RETRIES": "0"}):
            client = model_providers.openai_client(settings.llm_endpoint(), settings.llm_timeout_seconds(), settings.model_max_retries())
        self.assertEqual(client.timeout, 42)
        self.assertEqual(client.max_retries, 0)




class JobStoreTests(unittest.TestCase):
    def test_job_that_stops_updating_is_reported_as_interrupted(self) -> None:
        from n_note_ai_service_layers.services.jobs import JobStore
        store = InMemoryDocumentStore()
        jobs = JobStore(store, timeout_seconds=60)
        job, created = jobs.create("k", "t", {})
        self.assertTrue(created)
        jobs.update(job["job_id"], status="running", stage="stt")
        stored = store.get("workflow_jobs", job["job_id"])
        store._items["workflow_jobs"][job["job_id"]] = stored.__class__(stored.key, stored.version, stored.payload, stored.updated_at - 120)
        self.assertEqual(jobs.get(job["job_id"])["error"]["code"], "interrupted")


class _FakeRecognizer:
    def __init__(self, names):
        self.names = names

    def detect(self, text):
        from n_note_ai_service_layers.contracts.privacy import EntitySpan
        return [EntitySpan(text.index(name), text.index(name) + len(name), "PERSON", 0.99) for name in self.names if name in text]


class NerRedactionTests(unittest.TestCase):
    def test_ner_redacts_names_rules_cannot_see(self) -> None:
        redactor = TranscriptRedactor(_FakeRecognizer(["영희", "이민호"]))
        result = redactor.redact("영희가 밤새 열이 나서 해열제 투여함. 담당의 이민호 선생님께 보고함")
        self.assertEqual(result.text, "[[PERSON_1]]가 밤새 열이 나서 해열제 투여함. 담당의 [[PERSON_2]] 선생님께 보고함")
        self.assertEqual(result.replacement_count, 2)

    def test_ner_spans_overlapping_tokens_or_titles_are_skipped(self) -> None:
        redactor = TranscriptRedactor(_FakeRecognizer(["홍길동", "선생"]))
        result = redactor.redact("환자명: 홍길동, 선생님께 보고")
        self.assertEqual(result.text, "환자명: [[LABELED_PERSON_1]], 선생님께 보고")

    def test_ner_failure_stops_the_workflow_before_external_calls(self) -> None:
        class Broken:
            def detect(self, text): raise RuntimeError("model crashed")

        class Stt:
            def transcribe(self, audio): return Transcript("영희 열남", "fake", "fake")

        class Gateway:
            class Client:
                def discover(self, request_id): raise AssertionError("gateway must not be called")
            gateway = Client()

        workflow = NursingNoteWorkflow(Stt(), None, Gateway(), redactor=TranscriptRedactor(Broken()))
        with self.assertRaises(WorkflowStageError) as raised:
            workflow.generate(object(), AudioInput(b"a", "a.wav", "audio/wav", "t"), SCHEMA)
        self.assertEqual(raised.exception.stage, "redaction")

    @unittest.skipUnless(os.getenv("N_NOTE_TEST_NER_MODEL"), "set N_NOTE_TEST_NER_MODEL to run the real local NER model")
    def test_real_korean_ner_model(self) -> None:
        from n_note_ai_service_layers.adapters.transformers_ner import TransformersNerRecognizer
        redactor = TranscriptRedactor(TransformersNerRecognizer(os.environ["N_NOTE_TEST_NER_MODEL"]))
        text = "영희가 밤새 열이 나서 해열제 투여함. 박철수 환자 혈압 120에 80. 오른쪽 하지 부종 있어 담당의 이민호 선생님께 보고함"
        result = redactor.redact(text)
        for name in ("영희", "박철수", "이민호"):
            self.assertNotIn(name, result.text)
        for clinical in ("해열제 투여함", "혈압 120에 80", "오른쪽 하지 부종", "선생님께 보고함"):
            self.assertIn(clinical, result.text)


if __name__ == "__main__":
    unittest.main()
