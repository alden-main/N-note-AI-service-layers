"""서비스 동작을 추적하기 위한 감사 이력을 제공한다."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from n_note_ai_service_layers.adapters.document_store import DocumentStore
from n_note_ai_service_layers.services.clock import now_local

AUDIT_COLLECTION = "audit_events"
# 영속 저장소가 있으면 조회용 최근 이벤트만 메모리에 남긴다.
_RECENT_EVENTS = 1000


@dataclass(frozen=True)
class AuditEvent:
    """한 번의 서비스 동작에 대한 감사 기록을 표현한다.

    입력: 추적 ID, 동작명, 선택 스키마 식별자·버전, 수행자와 발생 시각이다.
    출력: ``AuditTrail``에 보관되는 불변 감사 이벤트이다.
    """
    trace_id: str
    action: str
    schema_id: str | None
    version: str | None
    occurred_at: datetime
    actor: str | None = None
    subject_id: str | None = None


class AuditTrail:
    """감사 이벤트를 발생 순서대로 수집하고, 저장소가 있으면 영속 보관한다.

    입력: ``record`` 호출의 추적 ID, 동작명, 스키마 식별자·버전, 수행자와 대상 ID이다.
    출력: ``events``(최근 이벤트)에 추가되고 저장소 ``audit_events`` 컬렉션에 기록되는 ``AuditEvent``이다.
    """
    def __init__(self, store: DocumentStore | None = None) -> None:
        self.store = store
        self.events: deque[AuditEvent] = deque(maxlen=_RECENT_EVENTS)

    def record(
        self, trace_id: str, action: str, schema_id: str | None = None, version: str | None = None,
        actor: str | None = None, subject_id: str | None = None,
    ) -> None:
        event = AuditEvent(trace_id, action, schema_id, version, now_local(), actor, subject_id)
        self.events.append(event)
        if self.store is not None:
            self.store.put(AUDIT_COLLECTION, str(uuid4()), {
                "trace_id": trace_id, "action": action, "schema_id": schema_id, "version": version,
                "occurred_at": event.occurred_at.isoformat(), "actor": actor, "subject_id": subject_id,
            }, expected_version=0)
