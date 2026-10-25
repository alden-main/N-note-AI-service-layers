"""음성 전사와 초안 생성을 위한 외부 AI 제공자 계약이다."""

from __future__ import annotations

from typing import Protocol

from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormSchema


class SpeechToTextProvider(Protocol):
    """음성 입력을 전사 결과로 변환하는 제공자 인터페이스이다.

    입력: ``AudioInput`` 객체이다.
    출력: 전사 텍스트와 메타데이터를 담은 ``Transcript`` 객체이다.
    """
    def transcribe(self, audio: AudioInput) -> Transcript: ...


class DraftGenerationProvider(Protocol):
    """LLM 입력으로 양식 초안을 생성하는 제공자 인터페이스이다.

    입력: 최소화된 LLM 페이로드와 ``FormSchema``이다.
    출력: 제안 필드와 근거를 담은 ``DraftResult``이다.
    """
    def generate(self, payload: dict, schema: FormSchema) -> DraftResult: ...
