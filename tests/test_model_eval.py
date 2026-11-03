from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from n_note_ai_service_layers import run_model_eval
from n_note_ai_service_layers.contracts.draft import DraftResult

ROOT = Path(__file__).parents[1]


class _Generator:
    def generate(self, payload, schema):
        self.payload = payload
        return DraftResult(
            proposed_fields={"assessment": "오한 호소함. 체온 38.2℃.", "plan_and_intervention": "해열제 투여함. 담당의에게 보고함.", "evaluation": "체온 37.4℃로 하강함."},
            evidence_refs={"assessment": ("transcript",)},
            needs_confirmation=False,
            model_version="fake-local",
            prompt_version="draft-v1",
        )


class ModelEvalTests(unittest.TestCase):
    def test_transcript_term_recall_reports_missing_terms(self):
        scored = run_model_eval.score_transcript("강경변증 진단, 흑색변 있음", {"stt_terms": ["흑색변", "간경변증"]})
        self.assertEqual(scored, {"stt_term_recall": 0.5, "missing_stt_terms": ["간경변증"]})
        self.assertEqual(run_model_eval.score_transcript("전사문", {}), {"stt_term_recall": None, "missing_stt_terms": []})

    def test_synthetic_audio_case_has_reference_and_terms(self):
        case = next(item for item in run_model_eval.load_cases(ROOT / "eval/cases") if item.case_id == "stt-admission-tts")
        self.assertEqual(case.audio_path.suffix, ".wav")
        self.assertTrue(all(term in case.reference_transcript for term in case.expected["stt_terms"]))

    def test_character_error_rate_ignores_spacing(self):
        self.assertEqual(run_model_eval.character_error_rate("체온 38.2도", "체온38.2도"), 0.0)
        self.assertAlmostEqual(run_model_eval.character_error_rate("오한있음", "오환있음"), 0.25)

    def test_bundled_cases_are_loadable(self):
        cases = run_model_eval.load_cases(ROOT / "eval" / "cases")
        self.assertGreaterEqual(len(cases), 4)
        self.assertTrue(all(case.transcript or case.audio_path for case in cases))

    def test_run_scores_audio_case_with_cer_and_term_recall(self):
        env = {
            "N_NOTE_FORMS_DIR": str(ROOT / "forms"),
            "MCP_GATEWAY_URL": "http://127.0.0.1:9/mcp",
            "MCP_GATEWAY_TIMEOUT_SECONDS": "0.5",
        }
        case = next(item for item in run_model_eval.load_cases(ROOT / "eval/cases") if item.case_id == "stt-admission-tts")
        stt = run_model_eval.FixedTranscriptProvider(case.reference_transcript.replace("간경변증", "강경변증"))
        with patch.dict(os.environ, env), patch("n_note_ai_service_layers.adapters.model_providers.draft_generator", lambda gateway: _Generator()):
            record = run_model_eval.run_case(case, stt)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["missing_stt_terms"], ["간경변증"])
        self.assertGreater(record["cer"], 0)
        self.assertEqual(run_model_eval.summarize("fake", [record])["mean_stt_term_recall"], record["stt_term_recall"])

    def test_run_scores_transcript_case_without_gateway(self):
        env = {
            "N_NOTE_FORMS_DIR": str(ROOT / "forms"),
            "MCP_GATEWAY_URL": "http://127.0.0.1:9/mcp",
            "MCP_GATEWAY_TIMEOUT_SECONDS": "0.5",
            "N_NOTE_MODEL_PROFILE": "fake-local",
        }
        generator = _Generator()
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, env), \
                patch("n_note_ai_service_layers.adapters.model_providers.draft_generator", lambda gateway: generator):
            code = run_model_eval.main(["run", "--cases", str(ROOT / "eval" / "cases"), "--out", out, "--case", "daily-fever"])
            self.assertEqual(code, 0)
            run_dir = next((Path(out) / "fake-local").iterdir())
            record = json.loads((run_dir / "results.jsonl").read_text().splitlines()[0])
            summary = json.loads((run_dir / "summary.json").read_text())
            rows = run_model_eval.compare(Path(out))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["required_filled"], record["required_total"])
        self.assertEqual(record["keyword_recall"], 1.0)
        self.assertEqual(record["forbidden_found"], [])
        self.assertEqual(record["gateway_status"], "unavailable")
        self.assertIn("오한", json.dumps(generator.payload, ensure_ascii=False))
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(rows[0]["profile"], "fake-local")


if __name__ == "__main__":
    unittest.main()
