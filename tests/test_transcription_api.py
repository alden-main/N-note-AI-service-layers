from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from n_note_ai_service_layers.api import app, get_speech_to_text_provider
from n_note_ai_service_layers.contracts.audio import Transcript


class FakeStt:
    def __init__(self) -> None:
        self.audio = None

    def transcribe(self, audio):
        self.audio = audio
        return Transcript("간호 기록", "fake", "fake-stt", "ko")


class TranscriptionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.stt = FakeStt()
        app.dependency_overrides[get_speech_to_text_provider] = lambda: self.stt

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_transcribes_uploaded_audio_without_persisting_it(self) -> None:
        response = self.client.post("/v1/transcriptions", files={"audio": ("note.wav", b"audio", "audio/wav")}, data={"language_hint": "ko"}, headers={"X-Trace-Id": "trace-123"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "간호 기록", "provider": "fake", "model": "fake-stt", "language": "ko", "trace_id": "trace-123", "redacted": True, "redaction_count": 0})
        self.assertEqual(self.stt.audio.content, b"audio")
        self.assertEqual(self.stt.audio.filename, "note.wav")

    def test_rejects_non_audio_upload(self) -> None:
        response = self.client.post("/v1/transcriptions", files={"audio": ("note.txt", b"text", "text/plain")})
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "unsupported audio content type")


if __name__ == "__main__":
    unittest.main()
