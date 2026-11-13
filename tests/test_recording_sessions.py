"""녹음 중 청크 전사(화면용 누적 전사문)와 녹음 종료 후 초안 생성 작업 시작을 검증한다."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from n_note_ai_service_layers.adapters.openai_stt import OpenAiSpeechToTextAdapter
from n_note_ai_service_layers.api import app, get_nursing_note_workflow, get_services, get_speech_to_text_provider, get_workflow_context_flow, store
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.services.recording_sessions import RecordingSessionStore
from test_multi_form_workflow import BASE, _Context, _Stt, _job_result, _workflow

PATIENT_ID = "P12345"


class _ChunkStt:
    """청크 바이트를 그대로 전사문으로 돌려준다. ``fail_once``면 첫 호출만 실패한다."""

    def __init__(self, fail_once: bool = False) -> None:
        self.inputs: list[AudioInput] = []
        self.fail_once = fail_once

    def transcribe(self, audio: AudioInput) -> Transcript:
        self.inputs.append(audio)
        if self.fail_once:
            self.fail_once = False
            raise TimeoutError("stt slow")
        return Transcript(audio.content.decode(), "fake", "fake-stt", "ko")


class RecordingSessionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        store.clear()
        self.client = TestClient(app)
        self.stt = _ChunkStt()
        app.dependency_overrides[get_speech_to_text_provider] = lambda: self.stt
        app.dependency_overrides[get_workflow_context_flow] = _Context

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def _open(self) -> str:
        created = self.client.post("/v1/recording-sessions", json={"patient_id": PATIENT_ID, "encounter_id": "E9876", "language_hint": "ko"})
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.headers["Location"], f"/v1/recording-sessions/{created.json()['session_id']}")
        return created.json()["session_id"]

    def _chunk(self, session_id: str, sequence: int, text: str, content_type: str = "audio/webm"):
        return self.client.post(
            f"/v1/recording-sessions/{session_id}/chunks",
            files={"audio": (f"chunk-{sequence}.webm", text.encode(), content_type)}, data={"sequence": str(sequence)},
        )

    def test_each_chunk_returns_redacted_running_transcript_and_passes_context_to_stt(self) -> None:
        session_id = self._open()
        first = self._chunk(session_id, 0, f"환자번호 {PATIENT_ID} 통증 호소함")
        self.assertEqual(first.status_code, 200, first.text)
        self.assertNotIn(PATIENT_ID, first.json()["text"])
        self.assertEqual(first.json()["next_sequence"], 1)
        second = self._chunk(session_id, 1, "통증 점수 2점", "audio/webm;codecs=opus").json()
        self.assertEqual(second["text"], "통증 점수 2점")
        self.assertEqual(second["transcript"], f"{first.json()['text']} 통증 점수 2점")
        self.assertIsNone(self.stt.inputs[0].preceding_text)
        # 다음 청크 STT는 앞 청크의 원본 끝문장을 문맥으로 받는다(로컬 STT 경계 안).
        self.assertTrue(self.stt.inputs[1].preceding_text.endswith("통증 호소함"))
        state = self.client.get(f"/v1/recording-sessions/{session_id}").json()
        self.assertEqual((state["status"], state["chunk_count"], state["transcript"]), ("recording", 2, second["transcript"]))

    def test_resent_chunk_is_not_transcribed_again_and_gaps_are_rejected(self) -> None:
        session_id = self._open()
        self._chunk(session_id, 0, "첫 문장")
        again = self._chunk(session_id, 0, "첫 문장")
        self.assertTrue(again.json()["duplicate"])
        self.assertEqual(len(self.stt.inputs), 1)
        gap = self._chunk(session_id, 2, "셋째 문장")
        self.assertEqual(gap.status_code, 409)
        self.assertEqual(gap.json()["detail"], "expected chunk sequence 1")

    def test_failed_chunk_can_be_retried_with_the_same_sequence(self) -> None:
        self.stt.fail_once = True
        session_id = self._open()
        self.assertEqual(self._chunk(session_id, 0, "첫 문장").status_code, 503)
        retried = self._chunk(session_id, 0, "첫 문장")
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.json()["transcript"], "첫 문장")

    def test_finishing_starts_the_workflow_once_without_transcribing_again(self) -> None:
        workflow_stt = _Stt()
        app.dependency_overrides[get_nursing_note_workflow] = lambda: _workflow(stt=workflow_stt)
        session_id = self._open()
        self._chunk(session_id, 0, "통증 점수")
        self._chunk(session_id, 1, "2점")
        url = f"/v1/recording-sessions/{session_id}/workflows"
        accepted = self.client.post(url, json={"purpose": BASE.purpose, "schema_ids": ["form-a", "form-b"]})
        job = _job_result(self.client, accepted)
        self.assertEqual(job["status"], "succeeded", job["error"])
        self.assertEqual(workflow_stt.calls, 0)
        self.assertNotIn("stt", job["completed_stages"])
        self.assertEqual(job["result"]["forms"][0]["result"]["transcript"]["text"], "통증 점수 2점")
        repeated = self.client.post(url, json={"purpose": BASE.purpose, "schema_ids": ["form-a"]})
        self.assertEqual(repeated.json()["job_id"], job["job_id"])
        self.assertEqual(self._chunk(session_id, 2, "늦은 청크").status_code, 409)
        session = get_services().recording_sessions.get(session_id)
        self.assertEqual([chunk.raw_text for chunk in session.chunks], ["", ""])
        state = self.client.get(f"/v1/recording-sessions/{session_id}").json()
        self.assertEqual((state["status"], state["job_id"]), ("workflow_started", job["job_id"]))

    def test_finishing_without_speech_is_rejected_and_sessions_can_be_discarded(self) -> None:
        app.dependency_overrides[get_nursing_note_workflow] = lambda: _workflow()
        session_id = self._open()
        self._chunk(session_id, 0, "   ")
        empty = self.client.post(f"/v1/recording-sessions/{session_id}/workflows", json={"purpose": BASE.purpose, "schema_id": "form-a"})
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(self.client.delete(f"/v1/recording-sessions/{session_id}").status_code, 204)
        self.assertEqual(self.client.get(f"/v1/recording-sessions/{session_id}").status_code, 404)


class RecordingSessionStoreTests(unittest.TestCase):
    def test_idle_sessions_expire_and_open_sessions_are_capped(self) -> None:
        now = [0.0]
        sessions = RecordingSessionStore(idle_seconds=10, max_sessions=1, max_chunks=5, clock=lambda: now[0])
        first = sessions.create("p", "e", None)
        with self.assertRaises(Exception):
            sessions.create("p", "e", None)
        now[0] = 11
        self.assertIsNone(sessions.get(first.session_id))
        self.assertIsNotNone(sessions.create("p", "e", None))


class SttContextPromptTests(unittest.TestCase):
    def test_preceding_text_follows_the_vocabulary_prompt(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.request = None
                self.audio = self
                self.transcriptions = self

            def create(self, **request):
                self.request = request
                return type("Response", (), {"text": "x"})()

        client = Client()
        OpenAiSpeechToTextAdapter(client, prompt="간호기록 구술").transcribe(AudioInput(b"a", "a.wav", "audio/wav", "t", "ko", "앞 문장"))
        self.assertEqual(client.request["prompt"], "간호기록 구술 앞 문장")


if __name__ == "__main__":
    unittest.main()
