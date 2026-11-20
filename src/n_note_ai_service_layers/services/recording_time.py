"""음성이 녹음된 시각을 정한다.

구술에 시각이 없으면 기록 시각 항목(``FormField.default_from == "recording_time"``)을 이 시각으로 채우고,
구술의 상대 시각("오후 2시", "어제")도 이 시각을 기준으로 해석한다. 우선순위는 다음과 같다.

1. 요청의 ``recorded_at``(클라이언트가 녹음을 시작한 시각, ISO 8601)
2. 음성 파일 메타데이터의 생성 시각(mp4·m4a·webm). 2000년 이전(컨테이너 기본값)이나 미래 시각은 무시한다.
3. 서버가 음성을 접수한 시각

녹음 세션은 세션을 연 시각(녹음 시작)을 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.services.clock import now_local

RecordingTimeSource = Literal["request", "audio_metadata", "received_at", "recording_session"]
# 클라이언트·서버 시계 차이로 미래가 되는 녹음 시각을 얼마나 허용할지다.
_FUTURE_TOLERANCE = timedelta(minutes=5)
# mp4는 생성 시각이 없으면 1904-01-01, 일부 기기는 1970-01-01을 넣는다.
_EARLIEST = datetime(2000, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class RecordingTime:
    value: datetime
    source: RecordingTimeSource


def parse_recorded_at(text: str) -> datetime:
    """ISO 8601 시각을 병동 시간대 시각으로 바꾼다. 시간대가 없으면 병동 시간대(``N_NOTE_TIMEZONE``)로 본다.

    잘못된 형식이면 ``ValueError``다.
    """
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    zone = settings.get().local_timezone
    return value.replace(tzinfo=zone) if value.tzinfo is None else value.astimezone(zone)


def _plausible(value: datetime, now: datetime) -> bool:
    return _EARLIEST <= value <= now + _FUTURE_TOLERANCE


def resolve_recording_time(requested: datetime | None, metadata_value: str | None, now: datetime | None = None) -> RecordingTime:
    """요청 값 → 음성 메타데이터 → 접수 시각 순으로 녹음 시각을 정한다(병동 시간대)."""
    now = now or now_local()
    if requested is not None:
        return RecordingTime(requested.astimezone(now.tzinfo), "request")
    if metadata_value:
        try:
            parsed = parse_recorded_at(metadata_value)
        except ValueError:
            parsed = None
        if parsed is not None and _plausible(parsed, now):
            return RecordingTime(parsed.astimezone(now.tzinfo), "audio_metadata")
    return RecordingTime(now, "received_at")


def format_for_field(value: datetime, field_type: str) -> str:
    """양식 항목에 넣을 표시 문자열이다. LLM 출력 형식과 같게 일시는 ``YYYY-MM-DD HH:MM``, 날짜는 ``YYYY-MM-DD``다."""
    local = value.astimezone(settings.get().local_timezone)
    return local.strftime("%Y-%m-%d" if field_type == "date" else "%Y-%m-%d %H:%M")
