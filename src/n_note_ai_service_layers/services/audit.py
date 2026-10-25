"""서비스 동작을 추적하기 위한 메모리 기반 감사 이력을 제공한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AuditEvent:
    """한 번의 서비스 동작에 대한 감사 기록을 표현한다.

    입력: 추적 ID, 동작명, 선택 스키마 식별자·버전과 발생 시각이다.
    출력: ``AuditTrail``에 보관되는 불변 감사 이벤트이다.
    """
    trace_id: str
    action: str
    schema_id: str | None
    version: str | None
    occurred_at: datetime


class AuditTrail:
    """감사 이벤트를 발생 순서대로 메모리에 수집한다.

    입력: ``record`` 호출의 추적 ID, 동작명, 스키마 식별자와 버전이다.
    출력: ``events`` 목록에 추가되는 ``AuditEvent``이다.
    """
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, trace_id: str, action: str, schema_id: str | None = None, version: str | None = None) -> None:
        self.events.append(AuditEvent(trace_id, action, schema_id, version, datetime.now().astimezone()))
