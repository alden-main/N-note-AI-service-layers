"""검토 완료 초안을 브리지 시스템이 가져가고 결과를 회신하는 API이다."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from n_note_ai_service_layers.api.schemas import BridgeAcknowledgement, BridgeHandoffRequest, HandoffEnvelope
from n_note_ai_service_layers.api.state import get_services

router = APIRouter(prefix="/v1/bridge/handoffs", tags=["bridge"])


@router.post("", response_model=HandoffEnvelope, status_code=status.HTTP_202_ACCEPTED)
def enqueue_handoff(handoff: BridgeHandoffRequest) -> HandoffEnvelope:
    """간호사 검토가 끝난 핸드오프를 브리지 조회 및 확인 대상으로 접수한다."""
    return get_services().handoff_store.create(handoff)


@router.get("", response_model=list[HandoffEnvelope])
def list_handoffs(delivery_status: str = "pending") -> list[HandoffEnvelope]:
    return get_services().handoff_store.list(delivery_status)


@router.get("/{handoff_id}", response_model=HandoffEnvelope)
def get_handoff(handoff_id: str) -> HandoffEnvelope:
    item = get_services().handoff_store.get(handoff_id)
    if item is None:
        raise HTTPException(status_code=404, detail="handoff not found")
    return item


@router.post("/{handoff_id}/acknowledgements", response_model=HandoffEnvelope)
def acknowledge_handoff(handoff_id: str, acknowledgement: BridgeAcknowledgement) -> HandoffEnvelope:
    store = get_services().handoff_store
    item = store.get(handoff_id)
    if item is None:
        raise HTTPException(status_code=404, detail="handoff not found")
    return store.update(item.model_copy(update={"delivery_status": acknowledgement.status, "acknowledgement": acknowledgement}))


@router.delete("/{handoff_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_handoff(handoff_id: str) -> Response:
    if not get_services().handoff_store.delete(handoff_id):
        raise HTTPException(status_code=404, detail="handoff not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
