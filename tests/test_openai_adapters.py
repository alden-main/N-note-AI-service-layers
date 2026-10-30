from __future__ import annotations

from types import SimpleNamespace
import unittest

from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter
from n_note_ai_service_layers.adapters.openai_stt import OpenAiSpeechToTextAdapter
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import SchemaStatus
from support import sample_form_schema
from n_note_ai_service_layers.services.context_assembler import ClinicalContextAssembler
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayConnectionError
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from n_note_ai_service_layers.contracts.clinical_context import Availability, ClinicalContextQuery, ClinicalFact
from datetime import datetime
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv


class _FakeTranscriptions:
    def __init__(self): self.request = None
    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(text="간호 기록", languages=[])


class _FakeResponses:
    def __init__(self): self.request = None
    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text='{"proposed_fields":{"assessment.pain_score":2,"summary.derived_level":"HIGH_RISK"},"evidence_refs":{"assessment.pain_score":["transcript:1"]},"needs_confirmation":false}')


class OpenAiAdapterTests(unittest.TestCase):
    def test_stt_adapter_sends_audio_to_transcriptions_and_returns_provider_neutral_transcript(self):
        transcriptions = _FakeTranscriptions()
        client = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
        result = OpenAiSpeechToTextAdapter(client).transcribe(AudioInput(b"audio", "note.wav", "audio/wav", "trace", "ko"))
        self.assertEqual(result.text, "간호 기록")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(transcriptions.request["model"], "gpt-4o-transcribe")
        self.assertEqual(transcriptions.request["language"], "ko")
        self.assertEqual(transcriptions.request["file"].name, "note.wav")

    def test_draft_adapter_uses_structured_output_and_drops_deterministic_fields(self):
        responses = _FakeResponses()
        client = SimpleNamespace(responses=responses)
        schema = sample_form_schema(SchemaStatus.ACTIVE)
        result = OpenAiDraftGenerationAdapter(client).generate({"facts": []}, schema)
        self.assertEqual(result.proposed_fields, {"assessment.pain_score": 2})
        self.assertEqual(result.evidence_refs["assessment.pain_score"], ("transcript:1",))
        self.assertEqual(responses.request["text"]["format"]["type"], "json_schema")
        self.assertNotIn("summary.derived_level", result.proposed_fields)

    def test_evidence_is_requested_only_for_filled_fields(self):
        evidence = OpenAiDraftGenerationAdapter._output_format(sample_form_schema(SchemaStatus.ACTIVE))["schema"]["properties"]["evidence_refs"]
        self.assertEqual(evidence["type"], "array")
        self.assertEqual(evidence["items"]["required"], ["field_id", "refs"])
        self.assertNotIn("summary.derived_level", evidence["items"]["properties"]["field_id"]["enum"])

    def test_array_evidence_is_merged_by_field_and_filtered(self):
        class Responses:
            def create(self, **kwargs):
                return SimpleNamespace(output_text='{"proposed_fields":{"assessment.pain_score":2},"evidence_refs":[{"field_id":"assessment.pain_score","refs":["transcript"]},{"field_id":"assessment.pain_score","refs":["fact:f1"]},{"field_id":"summary.derived_level","refs":["x"]}],"needs_confirmation":false}')
        result = OpenAiDraftGenerationAdapter(SimpleNamespace(responses=Responses())).generate({}, sample_form_schema(SchemaStatus.ACTIVE))
        self.assertEqual(result.evidence_refs, {"assessment.pain_score": ("transcript", "fact:f1")})

    def test_provider_neutral_flow_connects_stt_to_draft_generation(self):
        class FakeStt:
            def transcribe(self, audio):
                return Transcript("음성 전사", "fake", "fake")
        class FakeGenerator:
            def __init__(self): self.payload = None
            def generate(self, payload, schema):
                self.payload = payload
                return DraftResult({"assessment.pain_score": 2})
        schema = sample_form_schema(SchemaStatus.ACTIVE)
        query = ClinicalContextQuery("p", "e", schema.purpose, datetime.now().astimezone(), schema.schema_id, "trace", schema.version)
        facts = [ClinicalFact("assessment.pain_score", 2, "2", query.as_of, "test", Availability.AVAILABLE)]
        snapshot = ClinicalContextAssembler().build(query, facts, schema)
        class OfflineGateway:
            class Client:
                def discover(self, request_id): raise McpGatewayConnectionError("offline")
            gateway = Client()
        generator = FakeGenerator()
        result = NursingNoteWorkflow(FakeStt(), generator, OfflineGateway()).generate(snapshot, AudioInput(b"x", "x.wav", "audio/wav", "trace"), schema)
        self.assertEqual(result.transcript.text, "음성 전사")
        self.assertEqual(generator.payload["transcript"], "음성 전사")
        self.assertEqual(result.draft.proposed_fields["assessment.pain_score"], 2)
        self.assertEqual(result.draft_gateway_verification.status, "unavailable")

    def test_dotenv_loader_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / ".env"
            path.write_text("OPENAI_API_KEY=from-file\n")
            previous = os.environ.get("OPENAI_API_KEY")
            try:
                os.environ["OPENAI_API_KEY"] = "already-set"
                load_dotenv(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "already-set")
            finally:
                if previous is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = previous


if __name__ == "__main__":
    unittest.main()
