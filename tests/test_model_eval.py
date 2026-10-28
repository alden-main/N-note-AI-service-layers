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
    def test_character_error_rate_ignores_spacing(self):
        self.assertEqual(run_model_eval.character_error_rate("체온 38.2도", "체온38.2도"), 0.0)
        self.assertAlmostEqual(run_model_eval.character_error_rate("오한있음", "오환있음"), 0.25)

    def test_bundled_cases_are_loadable(self):
        cases = run_model_eval.load_cases(ROOT / "eval" / "cases")
        self.assertGreaterEqual(len(cases), 4)
        self.assertTrue(all(case.transcript for case in cases))

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
