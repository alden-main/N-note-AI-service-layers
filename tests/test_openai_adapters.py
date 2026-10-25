from __future__ import annotations

from types import SimpleNamespace
import unittest

from n_note_ai_service_layers.adapters.openai_draft_generation import OpenAiDraftGenerationAdapter
from n_note_ai_service_layers.adapters.openai_stt import OpenAiSpeechToTextAdapter
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import SchemaStatus
from n_note_ai_service_layers.policies.pressure_injury_schema import pressure_injury_poc_schema
from n_note_ai_service_layers.services.context_assembler import ClinicalContextAssembler
from n_note_ai_service_layers.services.draft_generation_flow import DraftGenerationFlow
from n_note_ai_service_layers.contracts.clinical_context import Availability, ClinicalContextQuery, ClinicalFact
from datetime import datetime
from pathlib import Path
import os
import tempfile
from n_note_ai_service_layers.settings import load_dotenv


class _FakeTranscriptions:
    def __init__(self): self.request = None
    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(text="간호 기록", languages=[])


class _FakeResponses:
    def __init__(self): self.request = None
    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text='{"proposed_fields":{"braden.nutrition":2,"risk.level":"HIGH_RISK"},"evidence_refs":{"braden.nutrition":["transcript:1"]},"needs_confirmation":false}')


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
        schema = pressure_injury_poc_schema(SchemaStatus.ACTIVE)
        result = OpenAiDraftGenerationAdapter(client).generate({"facts": []}, schema)
        self.assertEqual(result.proposed_fields, {"braden.nutrition": 2})
        self.assertEqual(result.evidence_refs["braden.nutrition"], ("transcript:1",))
        self.assertEqual(responses.request["text"]["format"]["type"], "json_schema")
        self.assertNotIn("risk.level", result.proposed_fields)

    def test_provider_neutral_flow_connects_stt_to_draft_generation(self):
        class FakeStt:
            def transcribe(self, audio):
                return Transcript("음성 전사", "fake", "fake")
        class FakeGenerator:
            def __init__(self): self.payload = None
            def generate(self, payload, schema):
                self.payload = payload
                return DraftResult({"braden.nutrition": 2})
        schema = pressure_injury_poc_schema(SchemaStatus.ACTIVE)
        query = ClinicalContextQuery("p", "e", schema.purpose, datetime.now().astimezone(), schema.schema_id, "trace", schema.version)
        facts = [ClinicalFact("braden.nutrition", 2, "2", query.as_of, "test", Availability.AVAILABLE)]
        snapshot = ClinicalContextAssembler().build(query, facts, schema)
        generator = FakeGenerator()
        transcript, result = DraftGenerationFlow(FakeStt(), generator).generate_from_audio(snapshot, AudioInput(b"x", "x.wav", "audio/wav", "trace"), schema)
        self.assertEqual(transcript.text, "음성 전사")
        self.assertEqual(generator.payload["transcript"], "음성 전사")
        self.assertEqual(result.proposed_fields["braden.nutrition"], 2)

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
