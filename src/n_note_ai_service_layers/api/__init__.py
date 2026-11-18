"""N-note HTTP API 패키지다. ``uvicorn n_note_ai_service_layers.api:app``으로 실행한다.

- ``app.py``: 앱 조립 · ``state.py``: 저장소·레지스트리 · ``deps.py``: 주입 provider
- ``schemas.py``: 요청·응답 모델 · ``mappers.py``: 응답 변환 · ``routes/``: 엔드포인트

저장소·레지스트리는 ``get_services()``로 얻는다(처음 접근할 때 만든다).
"""

from __future__ import annotations

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

__all__ = [
    "app", "create_app", "main", "get_services", "resolve_workflow_form",
    "get_gateway_client", "get_note_store", "get_nursing_note_workflow", "get_pdf_renderer",
    "get_speech_to_text_provider", "get_transcript_redactor", "get_workflow_context_flow",
]
