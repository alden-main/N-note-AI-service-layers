"""OpenAI 음성 전사 API를 도메인 계약으로 감싸는 어댑터이다."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript


class OpenAiSpeechToTextAdapter:
    """OpenAI 전사 호출을 ``Transcript`` 도메인 객체로 변환한다.

    입력: OpenAI 클라이언트와 사용할 모델명이다.
    출력: ``transcribe`` 호출 시 전사 텍스트와 언어 정보를 담은 ``Transcript``이다.
    """
    def __init__(self, client: Any | None = None, model: str = "gpt-4o-transcribe", prompt: str | None = None) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self._client = client
        self._model = model
        # 병동 약어·의학용어 인식률을 높이기 위한 선택 문맥 힌트다(OPENAI_STT_PROMPT).
        self._prompt = prompt

    def transcribe(self, audio: AudioInput) -> Transcript:
        """음성 바이트를 전사하여 도메인 전사 결과로 반환한다.

        입력: ``audio``는 파일명, MIME 형식, 음성 바이트 및 선택 언어 힌트를 담는다.
        출력: OpenAI 전사 결과를 담은 ``Transcript`` 객체이다.
        """
        audio_file = BytesIO(audio.content)
        audio_file.name = audio.filename
        request: dict[str, Any] = {"model": self._model, "file": audio_file}
        if audio.language_hint:
            request["language"] = audio.language_hint
        if self._prompt:
            request["prompt"] = self._prompt
        response = self._client.audio.transcriptions.create(**request)
        languages = getattr(response, "languages", None) or []
        language = languages[0].code if languages and hasattr(languages[0], "code") else None
        return Transcript(text=response.text, provider="openai", model=self._model, language=language)
