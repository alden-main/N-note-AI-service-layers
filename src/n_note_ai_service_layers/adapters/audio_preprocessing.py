"""STT 호출 전후의 음성 전처리다: 형식 변환(16kHz mono WAV), 음성 구간 검출(VAD), 전사 환각 문구 제거.

- 형식 변환: 브라우저 녹음(webm/opus)·m4a 등 어떤 입력이든 16kHz mono 16-bit WAV로 바꿔 보낸다. 로컬 STT 서버
  (vLLM Whisper 등)가 디코딩할 수 있는 형식이 엔진마다 달라도 입력을 하나로 고정하기 위해서다. PyAV(ffmpeg 내장)를 쓴다.
- VAD: 말소리가 없는 음성(무음·일정한 잡음)은 STT에 보내지 않는다. Whisper는 무음에서 "시청해 주셔서 감사합니다" 같은
  문장을 지어내기 때문이다. 앞뒤 무음도 잘라 보낸다(형식 변환을 할 때만). 모델 파일이 필요 없는 에너지 기반 방식이다.
- 환각 필터: 그래도 나온 알려진 환각 문장(방송 자막 문구)과 같은 문장·단어의 반복 루프를 전사문에서 지운다.
"""

from __future__ import annotations

import re
import wave
from collections.abc import Iterable
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import PurePath

import numpy as np

from n_note_ai_service_layers.contracts.ai_provider import SpeechToTextProvider
from n_note_ai_service_layers.contracts.audio import AudioDecodeError, AudioInput, Transcript
from n_note_ai_service_layers.observability import get_logger, log_event

SAMPLE_RATE = 16_000
_FRAME_SAMPLES = SAMPLE_RATE * 30 // 1000
_logger = get_logger("stt.preprocessing")

# 공백·문장부호를 지운 문장 전체가 이 패턴과 일치하면 지운다(문장 일부만 겹치는 임상 문장은 지우지 않는다).
# Whisper가 한국어 방송·유튜브 자막으로 학습해 무음·잡음 구간에서 내놓는 문구다.
DEFAULT_HALLUCINATION_PATTERNS: tuple[str, ...] = (
    r"(끝까지)?시청해(주셔서|줘서)(정말)?(감사합니다|고맙습니다)",
    r".*(구독|좋아요|알림설정).*(부탁|눌러).*",
    r"(MBC|KBS|SBS|YTN|JTBC|MBN|TV조선|채널A|연합)뉴스.*",
    r".*자막.*",
    r"다음(영상|시간)에?(서)?(또)?(만나요|뵙겠습니다|만나겠습니다)",
)
# 같은 문장이 연달아 이 횟수 이상 나오면 디코딩 루프로 보고 한 번만 남긴다.
_SENTENCE_LOOP = 3
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
_NORMALIZE = re.compile(r"[\s.,!?·~\-\"'“”‘’。…]+")
# 같은 단어가 6번 넘게 이어지면 디코딩 루프로 보고 한 번만 남긴다.
_WORD_LOOP = re.compile(r"(?<!\S)(\S+)(?:\s+\1(?!\S)){5,}")


def decode_to_pcm(content: bytes, max_seconds: float) -> np.ndarray:
    """음성 바이트를 16kHz mono int16 샘플로 디코딩한다. 실패하거나 ``max_seconds``를 넘으면 ``AudioDecodeError``다."""
    import av

    limit = int(max_seconds * SAMPLE_RATE)
    parts: list[np.ndarray] = []
    total = 0
    try:
        with av.open(BytesIO(content), mode="r") as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError("no audio stream")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            frames = (out for frame in container.decode(stream) for out in resampler.resample(frame))
            for out in [*frames, *resampler.resample(None)]:
                samples = out.to_ndarray().reshape(-1)
                total += samples.size
                if total > limit:
                    raise AudioDecodeError(f"audio is longer than {max_seconds:g} seconds")
                parts.append(samples)
    except AudioDecodeError:
        raise
    except Exception as exc:  # PyAV(FFmpeg)의 형식·손상 오류다.
        raise AudioDecodeError("audio could not be decoded") from exc
    return np.concatenate(parts).astype(np.int16, copy=False) if parts else np.zeros(0, dtype=np.int16)


def encode_wav(samples: np.ndarray) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.astype("<i2", copy=False).tobytes())
    return buffer.getvalue()


@dataclass(frozen=True)
class VadConfig:
    # 말소리로 볼 30ms 구간의 합이 이보다 짧으면 말이 없는 음성으로 본다.
    min_speech_seconds: float = 0.3
    # 이보다 작은 소리(dBFS)는 말소리로 보지 않는다.
    min_dbfs: float = -50.0
    # 배경 소음(하위 10% 구간 에너지)보다 이만큼(dB) 커야 말소리로 본다. 일정한 잡음만 있는 음성을 걸러낸다.
    margin_db: float = 8.0
    # 앞뒤 무음을 자를 때 말소리 앞뒤로 남길 여유(초)다.
    padding_seconds: float = 0.5


@dataclass(frozen=True)
class SpeechActivity:
    speech_seconds: float
    start: int
    end: int

    def has_speech(self, config: VadConfig) -> bool:
        return self.speech_seconds >= config.min_speech_seconds


def detect_speech(samples: np.ndarray, config: VadConfig) -> SpeechActivity:
    """30ms 구간별 에너지로 말소리 구간을 찾는다. ``start``·``end``는 앞뒤 무음을 잘라 낸 샘플 범위다."""
    count = samples.size // _FRAME_SAMPLES
    if count == 0:
        return SpeechActivity(0.0, 0, 0)
    frames = samples[: count * _FRAME_SAMPLES].astype(np.float64).reshape(count, _FRAME_SAMPLES)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    dbfs = 20 * np.log10(rms / 32768.0 + 1e-10)
    threshold = max(config.min_dbfs, float(np.percentile(dbfs, 10)) + config.margin_db)
    speech = np.flatnonzero(dbfs > threshold)
    if speech.size == 0:
        return SpeechActivity(0.0, 0, 0)
    padding = int(config.padding_seconds * SAMPLE_RATE)
    start = max(0, int(speech[0]) * _FRAME_SAMPLES - padding)
    end = min(samples.size, (int(speech[-1]) + 1) * _FRAME_SAMPLES + padding)
    return SpeechActivity(speech.size * _FRAME_SAMPLES / SAMPLE_RATE, start, end)


class HallucinationFilter:
    """알려진 환각 문장과 반복 루프를 지운다. 숫자·임상 문장은 문장 전체가 패턴과 같을 때만 지우므로 보존된다."""

    def __init__(self, patterns: Iterable[str] = DEFAULT_HALLUCINATION_PATTERNS) -> None:
        self._patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)

    def clean(self, text: str) -> tuple[str, int]:
        """정리한 전사문과 지운 문장(또는 루프) 수를 돌려준다."""
        removed = 0
        runs: list[list[str]] = []
        last_key: str | None = None
        for sentence in (part.strip() for part in _SENTENCE_SPLIT.split(text)):
            key = _NORMALIZE.sub("", sentence)
            if not key:
                continue
            if any(pattern.fullmatch(key) for pattern in self._patterns):
                removed += 1
                continue
            looped = _WORD_LOOP.sub(r"\1", sentence)
            removed += looped != sentence
            if key == last_key:
                runs[-1].append(looped)
            else:
                runs.append([looped])
            last_key = key
        kept: list[str] = []
        for run in runs:
            if len(run) >= _SENTENCE_LOOP:
                removed += len(run) - 1
                run = run[:1]
            kept.extend(run)
        return " ".join(kept), removed


class AudioPreprocessingSpeechToText:
    """STT provider를 감싸 형식 변환·VAD·환각 필터를 적용한다.

    ``normalize``가 꺼져 있으면(OpenAI API) 원본 바이트를 그대로 보내고, 디코딩은 VAD 판정에만 쓴다. 이때 디코딩에
    실패하면 VAD 없이 원본을 보낸다(OpenAI는 더 많은 형식을 디코딩한다). ``normalize``가 켜져 있으면 디코딩 실패는
    ``AudioDecodeError``다.
    """

    def __init__(
        self, inner: SpeechToTextProvider, normalize: bool, vad: VadConfig | None,
        hallucination_filter: HallucinationFilter | None, max_seconds: float,
    ) -> None:
        self.inner = inner
        self.normalize = normalize
        self.vad = vad
        self.hallucination_filter = hallucination_filter
        self.max_seconds = max_seconds

    @property
    def provider(self) -> str:
        return getattr(self.inner, "provider", "unknown")

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    def transcribe(self, audio: AudioInput) -> Transcript:
        samples = self._decode(audio) if self.normalize or self.vad else None
        if samples is not None and self.vad is not None:
            activity = detect_speech(samples, self.vad)
            if not activity.has_speech(self.vad):
                log_event(_logger, "stt_skipped_no_speech", trace_id=audio.trace_id, audio_seconds=round(samples.size / SAMPLE_RATE, 2))
                return Transcript("", self.provider, self.model, audio.language_hint)
            if self.normalize:
                samples = samples[activity.start:activity.end]
        if samples is not None and self.normalize:
            audio = replace(audio, content=encode_wav(samples), filename=f"{PurePath(audio.filename).stem or 'audio'}.wav", mime_type="audio/wav")
        transcript = self.inner.transcribe(audio)
        if self.hallucination_filter is not None and transcript.text:
            text, removed = self.hallucination_filter.clean(transcript.text)
            if removed:
                log_event(_logger, "stt_hallucination_removed", trace_id=audio.trace_id, removed=removed)
                transcript = replace(transcript, text=text)
        return transcript

    def _decode(self, audio: AudioInput) -> np.ndarray | None:
        try:
            return decode_to_pcm(audio.content, self.max_seconds)
        except AudioDecodeError as exc:
            if self.normalize:
                raise
            log_event(_logger, "stt_vad_skipped_undecodable", trace_id=audio.trace_id, error=str(exc))
            return None
