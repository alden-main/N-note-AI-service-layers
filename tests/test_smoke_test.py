"""스모크 테스트(n-note-smoke)가 실제 HTTP 서버에 대해 끝까지 실행되고 결과를 남기는지 검증한다."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import uvicorn

from n_note_ai_service_layers.api import app, get_nursing_note_workflow, get_speech_to_text_provider, get_workflow_context_flow, store
from n_note_ai_service_layers.contracts.audio import Transcript
from n_note_ai_service_layers.run_smoke_test import main
from test_multi_form_workflow import BASE, _Context, _workflow

CASE = Path(__file__).resolve().parents[1] / "eval/cases/stt-admission-tts"


class _ReferenceStt:
    """음성 길이와 상관없이 참조 전사문의 일부를 돌려준다(청크마다 다른 문장)."""

    def __init__(self) -> None:
        self.words = (CASE / "reference_transcript.txt").read_text().split()
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return Transcript(" ".join(self.words[:20]), "fake", "fake-stt", "ko")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class SmokeTestRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        store.clear()
        app.dependency_overrides[get_speech_to_text_provider] = _ReferenceStt
        app.dependency_overrides[get_workflow_context_flow] = _Context
        app.dependency_overrides[get_nursing_note_workflow] = lambda: _workflow()
        cls.port = _free_port()
        cls.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cls.port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        app.dependency_overrides.clear()

    def _run(self, *extra: str) -> dict:
        with tempfile.TemporaryDirectory() as out:
            code = main(["--api-url", f"http://127.0.0.1:{self.port}", "--case", str(CASE), "--out", out, "--chunk-seconds", "10", *extra])
            reports = list(Path(out).rglob("*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
        self.assertEqual(code, 0 if report["ok"] else 1)
        return report

    def test_full_run_covers_transcription_chunks_and_draft(self) -> None:
        report = self._run("--schema-id", "form-a", "--purpose", BASE.purpose)
        self.assertTrue(report["ok"], json.dumps(report["steps"], ensure_ascii=False))
        steps = report["steps"]
        self.assertEqual(steps["stt_server"]["server"], "openai-api")
        self.assertIn("cer", steps["transcription"])
        # 34.6초 음성을 10초 청크로 나누면 4개다.
        self.assertEqual(steps["recording_chunks"]["count"], 4)
        self.assertEqual(steps["draft_workflow"]["schema_id"], "form-a")
        self.assertNotIn("stt", steps["draft_workflow"]["stages"])

    def test_stt_only_skips_the_draft_and_unknown_form_fails_the_run(self) -> None:
        report = self._run("--stt-only")
        self.assertTrue(report["ok"])
        self.assertEqual(report["steps"]["draft_workflow"]["status"], "skipped")
        self.assertNotIn("llm_server", report["steps"])
        failed = self._run("--schema-id", "no-such-form", "--purpose", BASE.purpose)
        self.assertFalse(failed["ok"])
        self.assertIn("HTTP 422", failed["steps"]["draft_workflow"]["error"])


if __name__ == "__main__":
    unittest.main()
