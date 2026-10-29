"""N-note HTTP API 패키지다. ``uvicorn n_note_ai_service_layers.api:app``으로 실행한다.

- ``app.py``: 앱 조립 · ``state.py``: 저장소·레지스트리 · ``deps.py``: 주입 provider
- ``schemas.py``: 요청·응답 모델 · ``mappers.py``: 응답 변환 · ``routes/``: 엔드포인트
"""

from __future__ import annotations

from typing import Any

from n_note_ai_service_layers.api.app import create_app, main
from n_note_ai_service_layers.api.deps import (
    get_gateway_client,
    get_note_store,
    get_nursing_note_workflow,
    get_pdf_renderer,
    get_speech_to_text_provider,
    get_transcript_redactor,
    get_workflow_context_flow,
)
from n_note_ai_service_layers.api.routes.workflow_jobs import resolve_workflow_form
from n_note_ai_service_layers.api.state import get_services

app = create_app()
# 평가 스크립트(run_model_eval)가 쓰던 이름이다.
_resolve_workflow_form = resolve_workflow_form

_SERVICE_ATTRIBUTES = {
    "template_registry": "template_registry",
    "template_review_service": "template_review_service",
    "draft_review_store": "draft_review_store",
    "store": "handoff_store",
    "handoff_store": "handoff_store",
    "audit": "audit",
    "jobs": "jobs",
    "job_runner": "job_runner",
}


def __getattr__(name: str) -> Any:
    """저장소·레지스트리는 처음 접근할 때 만든다(import 시 설정·양식 파일을 읽지 않는다)."""
    if name in _SERVICE_ATTRIBUTES:
        return getattr(get_services(), _SERVICE_ATTRIBUTES[name])
    raise AttributeError(name)


__all__ = [
    "app", "create_app", "main", "get_services", "resolve_workflow_form",
    "get_gateway_client", "get_note_store", "get_nursing_note_workflow", "get_pdf_renderer",
    "get_speech_to_text_provider", "get_transcript_redactor", "get_workflow_context_flow",
]
