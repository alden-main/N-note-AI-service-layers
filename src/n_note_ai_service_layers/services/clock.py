"""병동 시간대 기준 현재 시각을 제공한다."""

from __future__ import annotations

from datetime import datetime

from n_note_ai_service_layers import settings


def now_local() -> datetime:
    """``N_NOTE_TIMEZONE``(기본 Asia/Seoul) 기준 현재 시각이다. 컨테이너 시간대와 무관하다."""
    return datetime.now(settings.get().local_timezone)
