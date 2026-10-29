import time
import unittest

from fastapi.testclient import TestClient

from n_note_ai_service_layers.api import app, store, get_nursing_note_workflow, get_workflow_context_flow
from n_note_ai_service_layers.contracts.audio import Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.policies.pressure_injury_schema import pressure_injury_poc_schema
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflowResult



def _job_result(client, accepted):
    """비동기 초안 생성 작업(202)이 끝날 때까지 상태를 조회해 최종 작업을 반환한다."""
    assert accepted.status_code == 202, accepted.text
    for _ in range(500):
        job = client.get(accepted.headers["Location"]).json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("workflow job did not finish")

class BridgeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        store.clear()
        self.client = TestClient(app)

    def test_healthz(self) -> None:
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})

    def test_bridge_handoff_exchange(self) -> None:
        payload = {
            "schema_id": "pressure-injury",
            "schema_version": "pressure-injury-poc/v1",
            "fields": {"braden.nutrition": 2},
            "validation": {"status": "valid"},
            "nurse_review_status": "reviewed",
        }
        created = self.client.post("/v1/bridge/handoffs", json=payload)
        self.assertEqual(created.status_code, 202)
        handoff_id = created.json()["handoff_id"]
        pending = self.client.get("/v1/bridge/handoffs")
        self.assertEqual([item["handoff_id"] for item in pending.json()], [handoff_id])
        acknowledged = self.client.post(f"/v1/bridge/handoffs/{handoff_id}/acknowledgements", json={"status": "accepted", "bridge_reference": "bridge-123"})
        self.assertEqual(acknowledged.status_code, 200)
        self.assertEqual(acknowledged.json()["delivery_status"], "accepted")
        self.assertEqual(self.client.get("/v1/bridge/handoffs?delivery_status=pending").json(), [])

    def test_nursing_note_workflow_returns_candidates_without_auto_apply(self) -> None:
        schema = pressure_injury_poc_schema(status=SchemaStatus.ACTIVE)

        class Context:
            class Registry:
                def resolve_active(self, schema_id, purpose, version): return schema
            registry = Registry()
            def build(self, query): return object()

        class Workflow:
            def generate(self, snapshot, audio, selected_schema, known_identifiers=(), progress=None):
                return NursingNoteWorkflowResult(
                    Transcript("금새 확인", "fake", "fake-stt", "ko"),
                    DraftResult({"braden.nutrition": 2}),
                    GatewayTermVerificationResult("transcript", "ok"),
                    GatewayTermVerificationResult("draft", "ok"),
                )

        app.dependency_overrides[get_workflow_context_flow] = Context
        app.dependency_overrides[get_nursing_note_workflow] = Workflow
        try:
            response = self.client.post(
                "/v1/nursing-note-workflows",
                files={"audio": ("note.wav", b"audio", "audio/wav")},
                data={"patient_id": "p", "encounter_id": "e", "purpose": schema.purpose, "schema_id": schema.schema_id, "schema_version": schema.version},
            )
        finally:
            app.dependency_overrides.clear()
        job = _job_result(self.client, response)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertFalse(job["result"]["auto_apply"])
        self.assertEqual(job["result"]["transcript"]["text"], "금새 확인")
