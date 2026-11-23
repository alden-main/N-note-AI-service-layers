"""초안에 남은 보호 토큰([[TYPE_n]])을 지우는 후처리를 검증한다."""

from __future__ import annotations

import unittest

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.adapters.draft_generation import DraftGenerationAdapter
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.draft_token_scrub import scrub_privacy_tokens, scrub_text
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from support import StaticCompletion

BASE_INSTRUCTIONS = prompts.get().draft.instructions


class ScrubTextTests(unittest.TestCase):
    def test_removes_patient_subject_with_honorific_and_particle(self) -> None:
        self.assertEqual(
            scrub_text("[[PERSON_1]]님은 내일 폐수술 예정임. 기침과 가래가 조금 있었고, 하얀 가래임."),
            "내일 폐수술 예정임. 기침과 가래가 조금 있었고, 하얀 가래임.",
        )

    def test_handles_various_particles_and_spacing(self) -> None:
        cases = {
            "[[PERSON_1]] 님이 통증 호소함.": "통증 호소함.",
            "[[PERSON_1]]씨가 보행 가능함.": "보행 가능함.",
            "[[PERSON_1]]께서 식사 잘 함.": "식사 잘 함.",
            "[[LABELED_PERSON_1]]의 V/S 안정적임.": "V/S 안정적임.",
            "등록번호 [[KNOWN_ID_1]] 확인함.": "등록번호 확인함.",
            "통증 호소함 [[PERSON_1]].": "통증 호소함.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(scrub_text(source), expected)

    def test_does_not_treat_following_word_as_particle(self) -> None:
        # '이틀'의 '이'는 조사가 아니므로 남긴다.
        self.assertEqual(scrub_text("[[PERSON_1]] 이틀째 발열 있음."), "이틀째 발열 있음.")

    def test_removes_identity_label_in_front_of_token(self) -> None:
        # 비식별화가 "환자명: 최라온님이"를 "환자명: [[LABELED_PERSON_1]]이"로 바꾼 전사문을 LLM이 옮긴 경우다.
        cases = {
            "환자명: [[LABELED_PERSON_1]]이 기침함.": "기침함.",
            "환자명:[[LABELED_PERSON_1]]": "",
            "환자 성함은 [[LABELED_PERSON_1]]이고 수술 예정임.": "수술 예정임.",
            "이름 [[PERSON_1]]님이 통증 호소함.": "통증 호소함.",
            "성명: [[LABELED_PERSON_1]], 수면 부족함.": ", 수면 부족함.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(scrub_text(source), expected)

    def test_replaces_guardian_name_with_role(self) -> None:
        self.assertEqual(scrub_text("보호자명: [[LABELED_PERSON_2]]에게 수술 일정 설명함."), "보호자에게 수술 일정 설명함.")
        self.assertEqual(scrub_text("보호자 [[PERSON_2]]님께 연락함."), "보호자께 연락함.")

    def test_keeps_label_words_without_token(self) -> None:
        text = "이름을 부르면 반응함."
        self.assertIs(scrub_text(text), text)

    def test_keeps_text_without_tokens_unchanged(self) -> None:
        text = "기침과 가래가 조금 있었음.  걸어다니는 것에 문제 없음."
        self.assertIs(scrub_text(text), text)

    def test_keeps_multiline_layout(self) -> None:
        self.assertEqual(scrub_text("[[PERSON_1]]님은 수술 예정임.\n[[PERSON_1]]님이 불안 호소함."), "수술 예정임.\n불안 호소함.")


class ScrubDraftTests(unittest.TestCase):
    def test_returns_same_draft_when_no_tokens(self) -> None:
        draft = DraftResult({"sbar.situation": "내일 폐수술 예정임.", "score": 3}, {"sbar.situation": ("transcript",)})
        self.assertIs(scrub_privacy_tokens(draft), draft)

    def test_scrubs_nested_values_and_flags_confirmation(self) -> None:
        draft = DraftResult(
            {
                "sbar.situation": "[[PERSON_1]]님은 내일 폐수술 예정임.",
                "notes": ["[[PERSON_1]]님이 기침 호소함.", "가래 있음."],
                "detail": {"text": "[[PERSON_1]]님은 보행 가능함."},
                "score": 2,
            },
            {"sbar.situation": ("transcript",)},
            prompt_version="draft-v2",
        )
        result = scrub_privacy_tokens(draft)
        self.assertEqual(result.proposed_fields, {
            "sbar.situation": "내일 폐수술 예정임.",
            "notes": ["기침 호소함.", "가래 있음."],
            "detail": {"text": "보행 가능함."},
            "score": 2,
        })
        self.assertTrue(result.needs_confirmation)
        self.assertEqual(result.evidence_refs, {"sbar.situation": ("transcript",)})
        self.assertEqual(result.prompt_version, "draft-v2")

    def test_drops_field_that_was_only_a_token(self) -> None:
        draft = DraftResult({"patient.name": "[[PERSON_1]]", "note": "안정적임."}, {"patient.name": ("transcript",), "note": ("transcript",)})
        result = scrub_privacy_tokens(draft)
        self.assertEqual(result.proposed_fields, {"note": "안정적임."})
        self.assertEqual(result.evidence_refs, {"note": ("transcript",)})

    def test_drops_field_left_with_only_label_or_punctuation(self) -> None:
        # 벤치마크(SBAR 개인정보 사례)에서 s.patient_summary가 "환자명:", "환자명:."으로 남던 경우다.
        draft = DraftResult({
            "s.patient_summary": "환자명: [[LABELED_PERSON_1]].",
            "a.status": "환자명: [[LABELED_PERSON_1]]이 기침함.",
        })
        result = scrub_privacy_tokens(draft)
        self.assertEqual(result.proposed_fields, {"a.status": "기침함."})


class PromptTests(unittest.TestCase):
    def test_prompt_forbids_tokens_in_output(self) -> None:
        self.assertIn("must not appear in any output field", BASE_INSTRUCTIONS)
        self.assertNotIn("immutable", BASE_INSTRUCTIONS)

    def test_default_prompt_version_is_bumped(self) -> None:
        adapter = DraftGenerationAdapter(StaticCompletion(), "m")
        self.assertEqual(adapter._prompt_version, "draft-v4")

    def test_prompt_guards_measurements_placement_and_interpretation(self) -> None:
        # 벤치마크(openai-baseline)에서 활력징후 누락·해석 추가·'담당의 보고'를 평가 항목에 넣던 문제에 대한 지침이다.
        for phrase in ("never drop a measurement", "Do not add clinical interpretations", "reporting to a physician",
                       "not evaluations", "do not drop statements", "the service fills the record time"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, BASE_INSTRUCTIONS)


class WorkflowScrubTests(unittest.TestCase):
    def test_workflow_scrubs_tokens_before_review_and_result(self) -> None:
        schema = FormSchema("test", "v1", "TEST", (FormSection("note", (FormField("note.text", "string"),)),), status=SchemaStatus.ACTIVE)

        class Stt:
            def transcribe(self, audio):
                return Transcript("김철수님 내일 폐수술 예정이에요", "fake", "fake-stt")

        class Projector:
            def project(self, snapshot, transcript, selected_schema, term_review_hints=(), reference_time=None):
                return {"transcript": transcript}

        class Generator:
            def generate(self, payload, selected_schema):
                return DraftResult({"note.text": "[[PERSON_1]]님은 내일 폐수술 예정임."})

        class Gateway:
            class Client:
                def discover(self, request_id): return ()
            def __init__(self): self.texts = []; self.gateway = self.Client()
            def verify(self, phase, text, trace_id, catalog=None):
                return GatewayTermVerificationResult(phase, "ok")
            def verify_document(self, phase, document, trace_id, catalog=None):
                self.texts.append(document.text)
                return GatewayTermVerificationResult(phase, "ok")

        gateway = Gateway()
        workflow = NursingNoteWorkflow(Stt(), Generator(), gateway, projector=Projector())
        result = workflow.generate(object(), AudioInput(b"audio", "note.wav", "audio/wav", "trace"), schema)
        self.assertEqual(result.draft.proposed_fields, {"note.text": "내일 폐수술 예정임."})
        self.assertTrue(result.draft.needs_confirmation)
        self.assertNotIn("[[", gateway.texts[0])


if __name__ == "__main__":
    unittest.main()
