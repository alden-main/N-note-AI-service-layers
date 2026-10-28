from __future__ import annotations

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from n_note_ai_service_layers.api import app, draft_review_store, get_nursing_note_workflow, get_workflow_context_flow, template_registry
from n_note_ai_service_layers.contracts.audio import Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.draft_review import InMemoryDraftReviewStore, NumericReviewRequired
from n_note_ai_service_layers.services.numeric_verification import NumericVerifier, source_numbers
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflowResult

REFERENCE = datetime(2026, 10, 28, 16, 44, tzinfo=timezone.utc)
SCHEMA = FormSchema(
    "numeric-test", "v1", "NUMERIC_TEST",
    (FormSection("s", (FormField("recorded_at", "datetime"), FormField("assessment", "text", required=True), FormField("evaluation", "text"))),),
    status=SchemaStatus.ACTIVE,
)


def values(text: str) -> set[float]:
    return {float(item) for item in source_numbers([text])}


class SpokenNumberTests(unittest.TestCase):
    def test_sino_korean_numbers_and_decimals(self):
        self.assertTrue({38.2, 120, 70}.issubset(values("체온 삼십팔 점 이 도, 혈압 백이십에 칠십")))
        self.assertTrue({150, 80}.issubset(values("백오십 씨씨, 시간당 팔십")))

    def test_spaced_lower_digit_is_joined(self):
        found = values("맥박 구십 이 회")
        self.assertIn(92, found)
        self.assertNotIn(90, found)

    def test_particle_after_number_is_not_a_digit(self):
        self.assertTrue({6} <= values("엔알에스 육 점이고") and 6.2 not in values("엔알에스 육 점이고"))
        self.assertTrue({60, 100} <= values("혈압은 백에 육십이었고"))

    def test_digit_by_digit_and_native_numbers_and_pm_hours(self):
        self.assertIn(751, values("칠오일호"))
        self.assertTrue({7, 19} <= values("저녁 일곱시쯤"))
        self.assertIn(14, values("오후 두 시에"))
        self.assertIn(25, values("스물다섯 살"))

    def test_arabic_numbers_in_transcript(self):
        self.assertTrue({38.2, 120, 70} <= values("체온 38.2도, 혈압 120/70"))


class NumericVerifierTests(unittest.TestCase):
    def check(self, fields, transcript, reference=REFERENCE):
        return [(item.field_id, item.text) for item in NumericVerifier().check(SCHEMA, fields, [transcript], reference)]

    def test_changed_vital_sign_is_flagged_with_field_offsets(self):
        findings = NumericVerifier().check(SCHEMA, {"evaluation": "혈압 160/100mmHg, 담당의에게 알림."}, ["혈압은 백에 육십이었고"], REFERENCE)
        self.assertEqual([(item.field_id, item.text) for item in findings], [("evaluation", "160")])
        self.assertEqual((findings[0].start_offset, findings[0].end_offset), (3, 6))

    def test_misjoined_pulse_is_flagged(self):
        self.assertEqual(self.check({"assessment": "맥박 90회"}, "맥박 구십 이 회"), [("assessment", "90")])

    def test_invented_recording_time_is_flagged_but_reference_date_is_allowed(self):
        self.assertEqual(self.check({"recorded_at": "2026-10-28 16:44"}, "통증 없음"), [("recorded_at", "16"), ("recorded_at", "44")])
        self.assertEqual(self.check({"recorded_at": "2026-10-28 14:00"}, "오후 두 시에 라운딩"), [])

    def test_date_far_from_reference_is_flagged(self):
        self.assertEqual(self.check({"recorded_at": "2023-10-02 14:00"}, "오후 두 시"), [("recorded_at", "2023-10-02")])

    def test_single_digits_privacy_tokens_and_abbreviations_are_ignored(self):
        self.assertEqual(self.check({"assessment": "Braden 3점, SpO2 확인, [[NAME_12]] 보호자 동석"}, "확인"), [])

    def test_facts_are_numeric_sources(self):
        findings = NumericVerifier().check(SCHEMA, {"assessment": "알부민 2.8"}, ["알부민 확인", "2.8"], REFERENCE)
        self.assertEqual(findings, ())


class NumericFinalizeTests(unittest.TestCase):
    def test_finalize_requires_confirmation_and_records_confirmed_values(self):
        store = InMemoryDraftReviewStore()
        record = store.create(SCHEMA, "혈압은 백에 육십", {"assessment": "혈압 160/100"}, (), "t", numeric_sources=("혈압은 백에 육십",), reference_time=REFERENCE)
        with self.assertRaises(NumericReviewRequired) as raised:
            store.finalize(record.draft_id, SCHEMA, "nurse-1", "t")
        self.assertEqual([item.text for item in raised.exception.findings], ["160"])
        note = store.finalize(record.draft_id, SCHEMA, "nurse-1", "t", numeric_review_confirmed=True)
        self.assertEqual(note.confirmed_numeric_findings, ("assessment: 160",))
        self.assertEqual(note.as_dict()["confirmed_numeric_findings"], ["assessment: 160"])

    def test_corrected_value_clears_the_finding(self):
        store = InMemoryDraftReviewStore()
        record = store.create(SCHEMA, "혈압은 백에 육십", {"assessment": "혈압 160/100"}, (), "t", reference_time=REFERENCE)
        store.revise(record.draft_id, SCHEMA, {"assessment": "혈압 100/60"}, {}, "nurse-1", "t")
        self.assertEqual(store.numeric_findings(store.get(record.draft_id), SCHEMA), ())
        self.assertEqual(store.finalize(record.draft_id, SCHEMA, "nurse-1", "t").confirmed_numeric_findings, ())


class NumericApiTests(unittest.TestCase):
    def test_workflow_reports_findings_and_finalize_requires_confirmation(self):
        if not any(item.schema_id == SCHEMA.schema_id for item in template_registry.list_versions()):
            template_registry.add(SCHEMA)
        client = TestClient(app)
        draft_review_store.items.clear()

        class Context:
            class Registry:
                def resolve_active(self, *args):
                    return SCHEMA
            registry = Registry()
            def build(self, query): return object()

        class Workflow:
            def generate(self, snapshot, audio, selected_schema):
                findings = NumericVerifier().check(SCHEMA, {"assessment": "혈압 160/100"}, ["혈압은 백에 육십"], REFERENCE)
                return NursingNoteWorkflowResult(
                    Transcript("혈압은 백에 육십", "fake", "fake-stt", "ko"),
                    DraftResult({"assessment": "혈압 160/100"}),
                    GatewayTermVerificationResult("transcript", "ok"),
                    GatewayTermVerificationResult("draft", "ok"),
                    numeric_findings=findings, numeric_sources=("혈압은 백에 육십",), reference_time=REFERENCE,
                )

        app.dependency_overrides[get_workflow_context_flow] = Context
        app.dependency_overrides[get_nursing_note_workflow] = Workflow
        try:
            created = client.post(
                "/v1/nursing-note-workflows",
                files={"audio": ("note.wav", b"audio", "audio/wav")},
                data={"patient_id": "p", "encounter_id": "e", "purpose": SCHEMA.purpose, "schema_id": SCHEMA.schema_id},
            )
        finally:
            app.dependency_overrides.clear()
        self.assertEqual(created.status_code, 200, created.text)
        body = created.json()
        self.assertEqual(body["numeric_verification"]["status"], "needs_review")
        self.assertEqual(body["numeric_verification"]["findings"][0]["text"], "160")
        draft = client.get(f"/v1/nursing-note-drafts/{body['draft_id']}").json()
        self.assertEqual(draft["numeric_verification"]["findings"][0]["field_id"], "assessment")
        blocked = client.post(f"/v1/nursing-note-drafts/{body['draft_id']}/finalize", json={"finalized_by": "nurse-1"})
        self.assertEqual(blocked.status_code, 422)
        self.assertEqual(blocked.json()["detail"]["findings"][0]["text"], "160")
        revised = client.post(f"/v1/nursing-note-drafts/{body['draft_id']}/revisions", json={"edited_by": "nurse-1", "edited_fields": {"assessment": "혈압 100/60"}})
        self.assertEqual(revised.json()["numeric_verification"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
