"""현재 시각의 유일한 출처다.

- ``now_utc``: 저장·API 시각(작업·초안·핸드오프·확정본)
- ``now_local``: 병동 시간대 시각. 구술 기준 시각과 감사 이력처럼 사람이 병동 시각으로 읽는 값에 쓴다.
"""

from __future__ import annotations

from datetime import UTC, datetime

from n_note_ai_service_layers import settings


def now_utc() -> datetime:
    """저장·API 응답에 쓰는 기준 시각(UTC)이다."""
    return datetime.now(UTC)


def now_local() -> datetime:
    """``N_NOTE_TIMEZONE``(기본 Asia/Seoul) 기준 현재 시각이다. 컨테이너 시간대와 무관하다."""
    return datetime.now(settings.get().local_timezone)
