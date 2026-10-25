"""음성 전사와 최소 LLM 페이로드 생성을 거쳐 기록 초안을 만드는 흐름이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.ai_provider import DraftGenerationProvider, SpeechToTextProvider
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextSnapshot
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.services.llm_payload import LlmPayloadProjector


class DraftGenerationFlow:
    """전사 제공자와 초안 생성 제공자를 독립적으로 연결하는 흐름이다.

    입력: 음성 전사기, 초안 생성기와 선택 LLM 페이로드 투영기이다.
    출력: 음성으로부터 전사문과 ``DraftResult``를 함께 생성한다.
    """
    def __init__(self, stt: SpeechToTextProvider, generator: DraftGenerationProvider, projector: LlmPayloadProjector | None = None) -> None:
        self.stt = stt
        self.generator = generator
        self.projector = projector or LlmPayloadProjector()

    def generate_from_audio(self, snapshot: ClinicalContextSnapshot, audio: AudioInput, schema: FormSchema) -> tuple[Transcript, DraftResult]:
        transcript = self.stt.transcribe(audio)
        payload = self.projector.project(snapshot, transcript.text, schema)
        return transcript, self.generator.generate(payload, schema)
