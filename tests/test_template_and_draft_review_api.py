from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from n_note_ai_service_layers.adapters.note_artifact_store import FileNursingNoteStore
from n_note_ai_service_layers.api import (
    app,
    draft_review_store,
    get_note_store,
    get_nursing_note_workflow,
    get_workflow_context_flow,
    template_registry,
)
from n_note_ai_service_layers.contracts.report_validation import ReportValidationCriterion, ReportValidationResult
from n_note_ai_service_layers.settings import pdf_font_path
from n_note_ai_service_layers.contracts.audio import Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import (
    GatewaySuggestion,
    GatewayTermOccurrence,
    GatewayTermVerificationResult,
)
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflowResult


class TemplateAndDraftReviewApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        draft_review_store.items.clear()

    def test_operator_can_register_approve_and_activate_a_versioned_template(self) -> None:
        payload = {
            "schema_id": "sbar-handoff",
            "version": "sbar-handoff/v1-test",
            "purpose": "SBAR_HANDOFF_TEST",
            "sections": [{
                "section_id": "situation",
                "label": "Situation",
                "display_order": 1,
                "fields": [{"field_id": "s.patient_summary", "type": "string", "label": "기본정보", "required": True, "display_order": 1}],
            }],
        }
        registered = self.client.post("/v1/form-templates", json=payload)
        self.assertEqual(registered.status_code, 201)
        self.assertEqual(registered.json()["status"], "draft")
        approved = self.client.post("/v1/form-templates/sbar-handoff/sbar-handoff/v1-test/approve", json={"operator_id": "nurse-1"})
        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.json()["status"], "approved")
        active = self.client.post("/v1/form-templates/sbar-handoff/sbar-handoff/v1-test/activate", json={"operator_id": "nurse-1"})
        self.assertEqual(active.status_code, 200)
        self.assertEqual(active.json()["status"], "active")

    def test_workflow_draft_can_be_edited_and_candidates_decided_without_auto_apply(self) -> None:
        schema_id, version, purpose = "draft-review-test", "v1", "DRAFT_REVIEW_TEST"
        if not any(item.schema_id == schema_id and item.version == version for item in template_registry.list_versions()):
            template_registry.add(FormSchema(
                schema_id, version, purpose,
                (FormSection("s", (FormField("s.note", "string"),)),),
                status=SchemaStatus.ACTIVE,
            ))

        schema = template_registry.get(schema_id, version)

        class Context:
            class Registry:
                def resolve_active(self, requested_schema_id, requested_purpose, requested_version):
                    return schema
            registry = Registry()
            def build(self, query): return object()

        occurrence = GatewayTermOccurrence("transcript", "korean__review", 0, 2, "금새", "금새", (GatewaySuggestion("금세"),))

        class Workflow:
            def generate(self, snapshot, audio, selected_schema):
                return NursingNoteWorkflowResult(
                    Transcript("금새 확인", "fake", "fake-stt", "ko"),
                    DraftResult({"s.note": "초안"}),
                    GatewayTermVerificationResult("transcript", "ok", (occurrence,), ("korean__review",)),
                    GatewayTermVerificationResult("draft", "ok"),
                )

        app.dependency_overrides[get_workflow_context_flow] = Context
        app.dependency_overrides[get_nursing_note_workflow] = Workflow
        try:
            created = self.client.post(
                "/v1/nursing-note-workflows",
                files={"audio": ("note.wav", b"audio", "audio/wav")},
                data={"patient_id": "p", "encounter_id": "e", "purpose": purpose, "schema_id": schema_id, "schema_version": version},
            )
        finally:
            app.dependency_overrides.clear()
        self.assertEqual(created.status_code, 200)
        body = created.json()
        self.assertFalse(body["auto_apply"])
        self.assertEqual(body["draft"]["sections"][0]["fields"][0]["value"], "초안")
        candidate_id = body["transcript_verification"]["occurrences"][0]["candidate_id"]
        revised = self.client.post(
            f"/v1/nursing-note-drafts/{body['draft_id']}/revisions",
            json={"edited_by": "nurse-1", "edited_fields": {"s.note": "직접 수정"}, "candidate_decisions": {candidate_id: "accepted"}},
        )
        self.assertEqual(revised.status_code, 200)
        self.assertEqual(revised.json()["revision"], 2)
        self.assertEqual(revised.json()["fields"]["s.note"], "직접 수정")
        self.assertEqual(revised.json()["candidate_decisions"][candidate_id], "accepted")

    def test_configured_form_flows_from_audio_to_corrected_draft_finalized_note_and_pdf(self) -> None:
        schema = template_registry.get("nursing-daily-note", "nursing-daily-note/v1")

        class Context:
            class Registry:
                def resolve_active(self, requested_schema_id, requested_purpose, requested_version):
                    assert (requested_schema_id, requested_purpose) == (schema.schema_id, schema.purpose)
                    return schema
            registry = Registry()
            def build(self, query): return object()

        draft_candidate = GatewayTermOccurrence("draft", "korean__review", 7, 9, "금새", "금새", (GatewaySuggestion("금세"),), field_id="assessment", category="standard_korean")
        validation = ReportValidationResult("ok", (ReportValidationCriterion("section:nursing_process", "필수 항목", "satisfied"),), {"satisfied": 1}, "report-validation__evaluate_report_against_standard")

        class Workflow:
            def generate(self, snapshot, audio, selected_schema):
                return NursingNoteWorkflowResult(
                    Transcript("통증은 없고 금새 잠듦", "fake", "fake-stt", "ko"),
                    DraftResult({"assessment": "통증 없음. 금새 입면함.", "plan_and_intervention": "수면 환경 조성함."}),
                    GatewayTermVerificationResult("transcript", "ok"),
                    GatewayTermVerificationResult("draft", "ok", (draft_candidate,), ("korean__review",)),
                    report_validation=validation,
                )

        with tempfile.TemporaryDirectory() as root:
            app.dependency_overrides[get_workflow_context_flow] = Context
            app.dependency_overrides[get_nursing_note_workflow] = Workflow
            app.dependency_overrides[get_note_store] = lambda: FileNursingNoteStore(Path(root))
            try:
                created = self.client.post(
                    "/v1/nursing-note-workflows",
                    files={"audio": ("note.wav", b"audio", "audio/wav")},
                    data={"patient_id": "p-1", "encounter_id": "e-1", "schema_id": schema.schema_id},
                )
                self.assertEqual(created.status_code, 200, created.text)
                body = created.json()
                self.assertEqual(body["form"]["title"], "간호일지")
                self.assertEqual(body["report_validation"]["criteria"][0]["status"], "satisfied")
                assessment = next(field for section in body["draft"]["sections"] for field in section["fields"] if field["field_id"] == "assessment")
                candidate_id = assessment["candidate_ids"][0]

                revised = self.client.post(f"/v1/nursing-note-drafts/{body['draft_id']}/revisions", json={"edited_by": "nurse-1", "candidate_decisions": {candidate_id: "accepted"}})
                self.assertEqual(revised.status_code, 200, revised.text)
                self.assertEqual(revised.json()["fields"]["assessment"], "통증 없음. 금세 입면함.")

                finalized = self.client.post(f"/v1/nursing-note-drafts/{body['draft_id']}/finalize", json={"finalized_by": "nurse-1"})
                self.assertEqual(finalized.status_code, 201, finalized.text)
                note = finalized.json()
                self.assertEqual(note["fields"]["assessment"], "통증 없음. 금세 입면함.")
                self.assertEqual(note["patient_id"], "p-1")
                again = self.client.post(f"/v1/nursing-note-drafts/{body['draft_id']}/revisions", json={"edited_by": "nurse-1", "edited_fields": {"assessment": "x"}})
                self.assertEqual(again.status_code, 409)

                if pdf_font_path() is not None:
                    self.assertEqual(note["pdf_status"], "ready")
                    pdf = self.client.get(note["pdf_url"])
                    self.assertEqual(pdf.status_code, 200)
                    self.assertEqual(pdf.headers["content-type"], "application/pdf")
                    self.assertTrue(pdf.content.startswith(b"%PDF"))
                self.assertEqual(self.client.get(f"/v1/nursing-notes/{note['note_id']}").json()["note_id"], note["note_id"])
            finally:
                app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
