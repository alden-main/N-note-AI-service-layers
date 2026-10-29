from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from n_note_ai_service_layers.adapters.note_artifact_store import FileNursingNoteStore
from n_note_ai_service_layers.adapters.pdf_renderer import NursingNotePdfRenderer, PdfConfigurationError
from n_note_ai_service_layers.contracts.draft_review import DraftCandidate, FinalizedNursingNote
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, PdfPlacement, SchemaStatus
from n_note_ai_service_layers.services.draft_review import DraftAlreadyFinalized, DraftReviewStore
from n_note_ai_service_layers.settings import pdf_font_path

SCHEMA = FormSchema(
    "daily", "daily/v1", "DAILY",
    (FormSection("s", (FormField("assessment", "text", required=True, label="간호사정"), FormField("plan", "text", label="간호중재"))),),
    status=SchemaStatus.ACTIVE, title="간호일지",
)


def _candidate(candidate_id, field_id, start, end, text, *suggestions):
    return DraftCandidate(candidate_id, "draft", "korean__review", start, end, text, suggestions, field_id)


class DraftReviewTests(unittest.TestCase):
    def _store_with(self, fields, candidates):
        store = DraftReviewStore()
        record = store.create(SCHEMA, "전사문", fields, tuple(item.candidate_id for item in candidates), "trace", tuple(candidates), "patient-1", "encounter-1")
        return store, record

    def test_accepted_candidates_replace_field_text_and_shift_later_offsets(self):
        candidates = [_candidate("c1", "assessment", 0, 2, "금새", "금세"), _candidate("c2", "assessment", 3, 6, "괜찬음", "괜찮음")]
        store, record = self._store_with({"assessment": "금새 괜찬음"}, candidates)
        store.revise(record.draft_id, SCHEMA, {}, {"c1": "accepted"}, "nurse-1", "trace")
        updated = store.revise(record.draft_id, SCHEMA, {}, {"c2": "accepted"}, "nurse-1", "trace", {"c2": "괜찮음"})
        self.assertEqual(updated.revisions[-1].fields["assessment"], "금세 괜찮음")
        self.assertEqual(updated.revisions[-1].applied_suggestions, {"c2": "괜찮음"})

    def test_direct_edit_makes_candidates_of_that_field_stale(self):
        store, record = self._store_with({"assessment": "금새 확인"}, [_candidate("c1", "assessment", 0, 2, "금새", "금세")])
        store.revise(record.draft_id, SCHEMA, {"assessment": "다시 작성"}, {}, "nurse-1", "trace")
        with self.assertRaises(ValueError):
            store.revise(record.draft_id, SCHEMA, {}, {"c1": "accepted"}, "nurse-1", "trace")

    def test_suggestion_must_come_from_the_candidate(self):
        store, record = self._store_with({"assessment": "금새 확인"}, [_candidate("c1", "assessment", 0, 2, "금새", "금세")])
        with self.assertRaises(ValueError):
            store.revise(record.draft_id, SCHEMA, {}, {"c1": "accepted"}, "nurse-1", "trace", {"c1": "임의 텍스트"})

    def test_finalize_requires_required_fields_and_locks_the_draft(self):
        store, record = self._store_with({"assessment": ""}, [])
        with self.assertRaises(ValueError):
            store.finalize(record.draft_id, SCHEMA, "nurse-1", "trace")
        store.revise(record.draft_id, SCHEMA, {"assessment": "통증 없음"}, {}, "nurse-1", "trace")
        note = store.finalize(record.draft_id, SCHEMA, "nurse-1", "trace")
        self.assertEqual((note.fields["assessment"], note.patient_id, note.revision), ("통증 없음", "patient-1", 2))
        with self.assertRaises(DraftAlreadyFinalized):
            store.revise(record.draft_id, SCHEMA, {"assessment": "변경"}, {}, "nurse-1", "trace")


def _note(fields):
    return FinalizedNursingNote("6f9c2a8e-1d7b-4a43-9a43-3b8e7f0c2d11", "draft-1", 2, SCHEMA.schema_id, SCHEMA.version, SCHEMA.display_title, "patient-1", "encounter-1", fields, {}, {}, "nurse-1", datetime(2026, 10, 26, 9, 30, tzinfo=timezone.utc), "trace")


def _text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)


@unittest.skipIf(pdf_font_path() is None, "Korean TrueType font is not available")
class PdfRendererTests(unittest.TestCase):
    def test_generated_pdf_contains_form_sections_and_values(self):
        pdf = NursingNotePdfRenderer(pdf_font_path()).render(SCHEMA, _note({"assessment": "통증 호소 없음", "plan": "낙상 예방 교육함"}))
        self.assertTrue(pdf.startswith(b"%PDF"))
        text = _text(pdf)
        self.assertIn("간호일지", text)
        self.assertIn("낙상 예방 교육함", text)

    def test_values_are_drawn_on_acroform_and_coordinate_templates(self):
        from reportlab.pdfgen import canvas

        with tempfile.TemporaryDirectory() as root:
            template = Path(root) / "template.pdf"
            drawing = canvas.Canvas(str(template))
            drawing.drawString(72, 760, "NURSING FORM")
            drawing.acroForm.textfield(name="assessment_box", x=72, y=600, width=400, height=60)
            drawing.save()
            schema = FormSchema(
                SCHEMA.schema_id, SCHEMA.version, SCHEMA.purpose,
                (FormSection("s", (
                    FormField("assessment", "text", required=True, label="간호사정", pdf=PdfPlacement(field_name="assessment_box")),
                    FormField("plan", "text", label="간호중재", pdf=PdfPlacement(page=1, x=72, y=400, width=400, height=40)),
                )),),
                status=SchemaStatus.ACTIVE, title="간호일지", pdf_template=str(template),
                pdf_meta=(("patient_id", PdfPlacement(page=1, x=72, y=720, width=200, height=20)),),
            )
            pdf = NursingNotePdfRenderer(pdf_font_path()).render(schema, _note({"assessment": "통증 호소 없음", "plan": "낙상 예방 교육함"}))
        reader = PdfReader(BytesIO(pdf))
        self.assertEqual(len(reader.pages), 2)  # 배치하지 않은 메타데이터는 추가 기록 쪽으로 간다.
        first = reader.pages[0].extract_text()
        self.assertIn("NURSING FORM", first)
        self.assertIn("통증 호소 없음", first)
        self.assertIn("patient-1", first)
        self.assertFalse(any(item.get_object().get("/Subtype") == "/Widget" for item in reader.pages[0].get("/Annots") or []))

    def test_unknown_acroform_field_is_a_configuration_error(self):
        from reportlab.pdfgen import canvas

        with tempfile.TemporaryDirectory() as root:
            template = Path(root) / "template.pdf"
            canvas.Canvas(str(template)).save()
            schema = FormSchema(SCHEMA.schema_id, SCHEMA.version, SCHEMA.purpose, (FormSection("s", (FormField("assessment", "text", pdf=PdfPlacement(field_name="missing")),)),), status=SchemaStatus.ACTIVE, pdf_template=str(template))
            with self.assertRaises(PdfConfigurationError):
                NursingNotePdfRenderer(pdf_font_path()).render(schema, _note({"assessment": "x"}))


class NoteStoreTests(unittest.TestCase):
    def test_finalized_note_is_written_once_and_ids_are_validated(self):
        with tempfile.TemporaryDirectory() as root:
            store = FileNursingNoteStore(Path(root))
            note = _note({"assessment": "통증 없음"})
            store.save(note)
            self.assertEqual(store.get(note.note_id), note)
            with self.assertRaises(FileExistsError):
                store.save(note)
            with self.assertRaises(LookupError):
                store.get("../etc/passwd")


if __name__ == "__main__":
    unittest.main()
