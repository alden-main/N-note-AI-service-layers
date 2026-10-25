from __future__ import annotations

import unittest

from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.draft_generation_flow import DraftGenerationFlow
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor


class TranscriptRedactionTests(unittest.TestCase):
    def test_redacts_explicit_identifiers_without_changing_clinical_terms(self) -> None:
        result = TranscriptRedactor().redact("환자명: 홍길동, 연락처 010-1234-5678, HbA1c 검사와 헤모글로빈 확인")
        self.assertEqual(result.text, "환자명: [[LABELED_PERSON_1]], 연락처 [[PHONE_1]], HbA1c 검사와 헤모글로빈 확인")
        self.assertEqual(result.replacement_count, 2)

    def test_workflow_sends_only_redacted_transcript_to_gateway_and_draft_provider(self) -> None:
        schema = FormSchema("test", "v1", "TEST", (FormSection("note", (FormField("note.text", "string"),)),), status=SchemaStatus.ACTIVE)

        class Stt:
            def transcribe(self, audio):
                return Transcript("환자명: 홍길동 금새 확인", "fake", "fake-stt")

        class Projector:
            def __init__(self): self.transcript = None
            def project(self, snapshot, transcript, selected_schema, term_review_hints=()):
                self.transcript = transcript
                return {"transcript": transcript}

        class Generator:
            def __init__(self): self.payload = None
            def generate(self, payload, selected_schema):
                self.payload = payload
                return DraftResult({"note.text": "헤모글로빈 확인"})

        class Gateway:
            class Client:
                def discover(self, request_id): return ()
            def __init__(self): self.texts = []; self.gateway = self.Client()
            def verify(self, phase, text, trace_id, catalog=None):
                self.texts.append((phase, text))
                return GatewayTermVerificationResult(phase, "ok")
            def verify_document(self, phase, document, trace_id, catalog=None):
                self.texts.append((phase, document.text))
                return GatewayTermVerificationResult(phase, "ok")

        projector, generator, gateway = Projector(), Generator(), Gateway()
        workflow = NursingNoteWorkflow(DraftGenerationFlow(Stt(), generator, projector), gateway)
        result = workflow.generate(object(), AudioInput(b"audio", "note.wav", "audio/wav", "trace"), schema)
        expected = "환자명: [[LABELED_PERSON_1]] 금새 확인"
        self.assertEqual(result.transcript.text, expected)
        self.assertEqual(projector.transcript, expected)
        self.assertEqual(gateway.texts[0], ("transcript", expected))
        self.assertEqual(result.redaction_count, 1)
        self.assertEqual(gateway.texts[1][0], "draft")
        self.assertIn("헤모글로빈 확인", gateway.texts[1][1])


if __name__ == "__main__":
    unittest.main()
