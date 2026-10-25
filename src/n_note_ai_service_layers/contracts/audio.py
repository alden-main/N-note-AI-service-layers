"""음성 입력과 전사 결과를 서비스 계층 사이에서 전달하는 불변 계약이다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioInput:
    """전사 제공자에 전달할 음성 파일을 표현한다.

    입력: 음성 바이트, 파일명, MIME 형식, 추적 ID와 선택 언어 힌트이다.
    출력: 전사 서비스가 소비하는 불변 입력 객체이다.
    """
    content: bytes
    filename: str
    mime_type: str
    trace_id: str
    language_hint: str | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    """전사 결과의 시간 구간과 화자 정보를 표현한다.

    입력: 구간 ID, 텍스트와 선택 시작·종료 시각 및 화자이다.
    출력: ``Transcript``에 포함되는 불변 구간 객체이다.
    """
    segment_id: str
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    """음성 전사 제공자가 반환한 전체 전사 결과를 표현한다.

    입력: 전사문, 제공자·모델 정보와 선택 언어·구간 목록이다.
    출력: 초안 생성 흐름에 전달되는 불변 전사 객체이다.
    """
    text: str
    provider: str
    model: str
    language: str | None = None
    segments: tuple[TranscriptSegment, ...] = ()
