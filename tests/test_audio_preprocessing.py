"""STT 전처리(16kHz mono WAV 변환·VAD·환각 필터)를 합성 음성으로 검증한다."""

from __future__ import annotations

import os
import unittest
import wave
from io import BytesIO
from unittest.mock import patch

import av
import numpy as np
from fastapi.testclient import TestClient

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.audio_preprocessing import (
    SAMPLE_RATE, AudioPreprocessingSpeechToText, HallucinationFilter, VadConfig, decode_to_pcm,
)
from n_note_ai_service_layers.api import app, get_speech_to_text_provider
from n_note_ai_service_layers.contracts.audio import AudioDecodeError, AudioInput, Transcript

RATE = 48_000


def _speech_like(seconds: float, rate: int = RATE) -> np.ndarray:
    """음절처럼 4Hz로 크기가 변하는 배음 신호다(말소리 대용)."""
    t = np.arange(int(seconds * rate)) / rate
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 540, 720), start=1))
    envelope = np.clip(np.sin(2 * np.pi * 4 * t), 0, None)
    return 0.3 * voice * envelope


def _noise(seconds: float, level: float, rate: int = RATE) -> np.ndarray:
    return np.random.default_rng(0).normal(0, level, int(seconds * rate))


def _encode(signal: np.ndarray, container: str, codec: str, rate: int = RATE, layout: str = "stereo") -> bytes:
    """float 신호를 지정한 형식으로 인코딩한다(브라우저 녹음 webm/opus 등)."""
    buffer = BytesIO()
    channels = 2 if layout == "stereo" else 1
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    with av.open(buffer, mode="w", format=container) as output:
        stream = output.add_stream(codec, rate=rate, layout=layout)
        frame_size = stream.codec_context.frame_size or 1024
        for start in range(0, pcm.size, frame_size):
            block = pcm[start:start + frame_size]
            if block.size < frame_size:
                block = np.pad(block, (0, frame_size - block.size))
            interleaved = np.repeat(block, channels).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout=layout)
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return buffer.getvalue()


def _webm(signal: np.ndarray) -> bytes:
    return _encode(signal, "webm", "libopus")


class _Inner:
    provider = "fake"
    model = "fake-stt"

    def __init__(self, text: str = "통증 호소함.") -> None:
        self.text = text
        self.inputs: list[AudioInput] = []

    def transcribe(self, audio: AudioInput) -> Transcript:
        self.inputs.append(audio)
        return Transcript(self.text, self.provider, self.model, "ko")


def _audio(content: bytes, filename: str = "chunk.webm", mime: str = "audio/webm;codecs=opus") -> AudioInput:
    return AudioInput(content, filename, mime, "trace", "ko")


class AudioNormalizationTests(unittest.TestCase):
    def test_browser_webm_opus_is_sent_as_trimmed_16k_mono_wav(self) -> None:
        signal = np.concatenate([np.zeros(3 * RATE), _speech_like(2.0), np.zeros(3 * RATE)])
        inner = _Inner()
        transcript = AudioPreprocessingSpeechToText(inner, True, VadConfig(), None, 60).transcribe(_audio(_webm(signal)))
        self.assertEqual(transcript.text, "통증 호소함.")
        sent = inner.inputs[0]
        self.assertEqual((sent.filename, sent.mime_type), ("chunk.wav", "audio/wav"))
        with wave.open(BytesIO(sent.content)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, SAMPLE_RATE))
            seconds = wav.getnframes() / SAMPLE_RATE
        # 8초 중 말소리 2초 + 앞뒤 여유만 남는다.
        self.assertLess(seconds, 4.0)
        self.assertGreater(seconds, 2.0)

    def test_other_containers_decode_to_the_same_format(self) -> None:
        for container, codec in (("wav", "pcm_s16le"), ("ogg", "libopus"), ("mp4", "aac")):
            with self.subTest(container=container):
                samples = decode_to_pcm(_encode(_speech_like(1.0), container, codec), 60)
                self.assertAlmostEqual(samples.size / SAMPLE_RATE, 1.0, delta=0.1)

    def test_undecodable_audio_fails_only_when_normalizing(self) -> None:
        with self.assertRaises(AudioDecodeError):
            AudioPreprocessingSpeechToText(_Inner(), True, VadConfig(), None, 60).transcribe(_audio(b"not audio"))
        inner = _Inner()
        # OpenAI API 경로(변환 끔)는 VAD만 건너뛰고 원본을 보낸다.
        AudioPreprocessingSpeechToText(inner, False, VadConfig(), None, 60).transcribe(_audio(b"not audio"))
        self.assertEqual(inner.inputs[0].content, b"not audio")

    def test_audio_longer_than_the_limit_is_rejected(self) -> None:
        with self.assertRaises(AudioDecodeError):
            decode_to_pcm(_webm(_speech_like(3.0)), max_seconds=1)


class VoiceActivityTests(unittest.TestCase):
    def test_silence_and_steady_noise_are_not_sent_to_stt(self) -> None:
        for name, signal in (("silence", np.zeros(5 * RATE)), ("steady noise", _noise(5, 0.05))):
            with self.subTest(name):
                inner = _Inner("시청해 주셔서 감사합니다.")
                transcript = AudioPreprocessingSpeechToText(inner, True, VadConfig(), None, 60).transcribe(_audio(_webm(signal)))
                self.assertEqual(transcript.text, "")
                self.assertEqual(inner.inputs, [])

    def test_speech_over_background_noise_is_kept(self) -> None:
        signal = _noise(4, 0.01) + np.concatenate([np.zeros(RATE), _speech_like(2.0), np.zeros(RATE)])
        inner = _Inner()
        AudioPreprocessingSpeechToText(inner, True, VadConfig(), None, 60).transcribe(_audio(_webm(signal)))
        self.assertEqual(len(inner.inputs), 1)

    def test_openai_path_skips_silent_audio_but_sends_the_original_bytes(self) -> None:
        inner = _Inner()
        silent = _webm(np.zeros(3 * RATE))
        self.assertEqual(AudioPreprocessingSpeechToText(inner, False, VadConfig(), None, 60).transcribe(_audio(silent)).text, "")
        speech = _webm(_speech_like(2.0))
        AudioPreprocessingSpeechToText(inner, False, VadConfig(), None, 60).transcribe(_audio(speech))
        self.assertEqual(inner.inputs[0].content, speech)


class HallucinationFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = HallucinationFilter()

    def test_known_broadcast_phrases_are_removed_but_clinical_sentences_stay(self) -> None:
        text = "체온 38.2도 확인함. 시청해 주셔서 감사합니다. MBC 뉴스 김철수입니다. 구독과 좋아요 부탁드립니다! 한글자막 by 한효정"
        self.assertEqual(self.filter.clean(text), ("체온 38.2도 확인함.", 4))

    def test_plain_thanks_and_twice_repeated_sentences_are_kept(self) -> None:
        text = "식사 잘 하심. 식사 잘 하심. 감사합니다."
        self.assertEqual(self.filter.clean(text), (text, 0))

    def test_decoding_loops_are_collapsed(self) -> None:
        self.assertEqual(self.filter.clean("통증 없음. 통증 없음. 통증 없음. 통증 없음."), ("통증 없음.", 3))
        self.assertEqual(self.filter.clean("네 네 네 네 네 네 네 네 알겠습니다."), ("네 알겠습니다.", 1))

    def test_filter_runs_after_stt(self) -> None:
        inner = _Inner("혈압 120/80 측정함. 다음 영상에서 만나요.")
        stt = AudioPreprocessingSpeechToText(inner, True, VadConfig(), self.filter, 60)
        self.assertEqual(stt.transcribe(_audio(_webm(_speech_like(2.0)))).text, "혈압 120/80 측정함.")


class ProviderFactoryTests(unittest.TestCase):
    def test_local_stt_normalizes_audio_by_default_and_openai_does_not(self) -> None:
        with patch.dict(os.environ, {"N_NOTE_STT_BASE_URL": "http://local-stt:8000/v1"}, clear=True):
            self.assertTrue(model_providers.speech_to_text_provider().normalize)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            stt = model_providers.speech_to_text_provider()
            self.assertFalse(stt.normalize)
            self.assertIsNotNone(stt.vad)
        env = {"OPENAI_API_KEY": "k", "N_NOTE_STT_AUDIO_NORMALIZE": "on", "N_NOTE_STT_VAD_ENABLED": "false", "N_NOTE_STT_HALLUCINATION_PATTERNS": '["끝"]'}
        with patch.dict(os.environ, env, clear=True):
            stt = model_providers.speech_to_text_provider()
            self.assertEqual((stt.normalize, stt.vad), (True, None))
            self.assertEqual(stt.hallucination_filter.clean("끝. 통증 없음."), ("통증 없음.", 1))
            self.assertEqual(settings.get().stt_hallucination_patterns, ["끝"])


class UndecodableAudioWorkflowTests(unittest.TestCase):
    def test_workflow_reports_decode_failure_separately_from_stt_outage(self) -> None:
        from n_note_ai_service_layers.services.nursing_note_workflow import WorkflowStageError
        from test_multi_form_workflow import AUDIO, SCHEMAS, _Stt, _workflow

        with self.assertRaises(WorkflowStageError) as raised:
            _workflow(stt=_Stt(AudioDecodeError("audio could not be decoded"))).generate(object(), AUDIO, SCHEMAS["form-a"])
        self.assertEqual(raised.exception.stage, "audio_decode")


class UndecodableAudioApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_transcription_api_returns_415_for_undecodable_audio(self) -> None:
        app.dependency_overrides[get_speech_to_text_provider] = lambda: AudioPreprocessingSpeechToText(_Inner(), True, VadConfig(), None, 60)
        response = TestClient(app).post("/v1/transcriptions", files={"audio": ("note.webm", b"broken", "audio/webm")})
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "audio could not be decoded")


if __name__ == "__main__":
    unittest.main()
