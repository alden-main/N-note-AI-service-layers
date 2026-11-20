from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.adapters.openai_draft_generation import DraftGenerationAdapter, OpenAiDraftGenerationAdapter
from n_note_ai_service_layers.api.routes import internal_stt
from support import sample_form_schema


class PromptCatalogTests(unittest.TestCase):
    def tearDown(self) -> None:
        prompts.clear_cache()

    def test_every_section_has_text_and_version(self) -> None:
        catalog = prompts.get()
        self.assertEqual(catalog.source, prompts.DEFAULT_PROMPTS_FILE)
        for text in (catalog.draft.instructions, catalog.draft.tool, catalog.form_extraction.instructions,
                     catalog.structured_output.final_output, catalog.structured_output.final_turn, catalog.stt.context):
            self.assertTrue(text.strip())
            self.assertNotIn("\n", text)  # TOML 줄 이음으로 한 줄 문자열이 된다.
        for version in (catalog.draft.version, catalog.form_extraction.version, catalog.stt.version):
            self.assertRegex(version, r"^[a-z-]+-v\d+$")

    def test_render_fills_only_named_placeholders(self) -> None:
        self.assertEqual(prompts.render("at most {n} of {status, detail} {other}", n=3), "at most 3 of {status, detail} {other}")
        rendered = internal_stt._instructions()
        self.assertIn(f"at most {internal_stt._MAX_EVIDENCE} of", rendered)
        self.assertIn("{status, detail}", rendered)
        self.assertNotIn("{max_evidence}", rendered)

    def test_draft_instructions_add_form_title_and_form_instructions(self) -> None:
        schema = sample_form_schema()
        instructions = DraftGenerationAdapter._instructions(schema)
        self.assertTrue(instructions.startswith(prompts.get().draft.instructions))
        self.assertIn(f"The form is '{schema.display_title}'.", instructions)

    def test_default_prompt_version_comes_from_catalog(self) -> None:
        adapter = OpenAiDraftGenerationAdapter(client=object())
        self.assertEqual(adapter._prompt_version, prompts.get().draft.version)

    def test_prompts_file_override_and_missing_key_failure(self) -> None:
        source = prompts.DEFAULT_PROMPTS_FILE.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            variant = Path(directory) / "variant.toml"
            variant.write_text(source.replace('version = "draft-v3"', 'version = "draft-v99"'), encoding="utf-8")
            broken = Path(directory) / "broken.toml"
            broken.write_text("[draft]\nversion = \"x\"\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {prompts.PROMPTS_FILE_ENV: str(variant)}):
                self.assertEqual(prompts.get().draft.version, "draft-v99")
            with mock.patch.dict(os.environ, {prompts.PROMPTS_FILE_ENV: str(broken)}):
                with self.assertRaises(prompts.PromptCatalogError):
                    prompts.get()


if __name__ == "__main__":
    unittest.main()
