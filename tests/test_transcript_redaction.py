from __future__ import annotations

import unittest

from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor


class TranscriptRedactionTests(unittest.TestCase):
    def test_every_generated_token_matches_the_shared_privacy_pattern(self) -> None:
        """비식별화가 만드는 모든 토큰 종류를 토큰 제거·수치 대조·용어 검토가 같은 패턴으로 알아본다."""
        import re

        from n_note_ai_service_layers.contracts.privacy import PRIVACY_TOKEN
        from n_note_ai_service_layers.services.draft_token_scrub import scrub_text

        text = (
            "MRN-0001 환자, 주민번호 900101-1234567, 메일 a@b.co, 010-1234-5678, 공일공 일이삼사 오육칠팔, "
            "환자번호: AB1234, 환자명: 홍길동, 생년월일: 1990-01-01, 이름은 김영희, 박철수님 방문"
        )
        redacted = TranscriptRedactor().redact(text, ("MRN-0001",)).text
        tokens = re.findall(r"\[\[[^\]]+\]\]", redacted)
        kinds = {token.strip("[]").rsplit("_", 1)[0] for token in tokens}
        self.assertTrue({"KNOWN_ID", "RESIDENT_ID", "EMAIL", "PHONE", "LABELED_ID", "LABELED_PERSON", "DATE_OF_BIRTH", "PERSON"} <= kinds)
        for token in tokens:
            self.assertIsNotNone(PRIVACY_TOKEN.fullmatch(token), token)
            self.assertNotIn(token, scrub_text(f"{token}님은 열이 있음"))

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
            def project(self, snapshot, transcript, selected_schema, term_review_hints=(), reference_time=None):
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
        workflow = NursingNoteWorkflow(Stt(), generator, gateway, projector=projector)
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
