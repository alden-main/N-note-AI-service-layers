"""브리지 핸드오프를 문서 저장소에 보관하는 전송 대기열이다."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from n_note_ai_service_layers.adapters.document_store import DocumentStore
from n_note_ai_service_layers.api.schemas import BridgeHandoffRequest, HandoffEnvelope

HANDOFF_COLLECTION = "bridge_handoffs"


class HandoffStore:
    """핸드오프를 저장소(SQLite 또는 메모리)에 보관한다.

    입력: ``BridgeHandoffRequest``와 브리지의 상태 회신이다.
    출력: UUID와 전송 상태가 붙은 ``HandoffEnvelope``이다.
    """
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def create(self, handoff: BridgeHandoffRequest) -> HandoffEnvelope:
        """전송 요청에 식별자와 생성 시각을 부여하여 대기 상태로 저장한다."""
        envelope = HandoffEnvelope(
            **handoff.model_dump(),
            handoff_id=str(uuid4()),
            delivery_status="pending",
            created_at=datetime.now(timezone.utc),
        )
        self.store.put(HANDOFF_COLLECTION, envelope.handoff_id, envelope.model_dump(mode="json"), expected_version=0)
        return envelope

    def get(self, handoff_id: str) -> HandoffEnvelope | None:
        stored = self.store.get(HANDOFF_COLLECTION, handoff_id)
        return HandoffEnvelope.model_validate(stored.payload) if stored else None

    def list(self, delivery_status: str) -> list[HandoffEnvelope]:
        items = (HandoffEnvelope.model_validate(item.payload) for item in self.store.list(HANDOFF_COLLECTION))
        return sorted((item for item in items if item.delivery_status == delivery_status), key=lambda item: item.created_at)

    def update(self, envelope: HandoffEnvelope) -> HandoffEnvelope:
        self.store.put(HANDOFF_COLLECTION, envelope.handoff_id, envelope.model_dump(mode="json"))
        return envelope

    def delete(self, handoff_id: str) -> bool:
        return self.store.delete(HANDOFF_COLLECTION, handoff_id)

    def clear(self) -> None:
        self.store.clear(HANDOFF_COLLECTION)
