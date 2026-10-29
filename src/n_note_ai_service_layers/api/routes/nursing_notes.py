"""간호기록 초안 검토·수정, 확정·PDF API이다. 초안 생성은 ``workflow_jobs``가 비동기로 수행한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from n_note_ai_service_layers.adapters.note_artifact_store import FileNursingNoteStore
from n_note_ai_service_layers.adapters.pdf_renderer import NursingNotePdfRenderer, PdfConfigurationError
from n_note_ai_service_layers.api import mappers
from n_note_ai_service_layers.api.deps import (
    get_authenticated_user,
    get_note_store,
    get_pdf_renderer,
    resolve_actor,
)
from n_note_ai_service_layers.api.schemas import DraftFinalizeRequest, DraftRevisionRequest
from n_note_ai_service_layers.api.state import get_services
from n_note_ai_service_layers.contracts.draft_review import FinalizedNursingNote
from n_note_ai_service_layers.services.draft_review import DraftAlreadyFinalized, DraftRevisionConflict, NumericReviewRequired

router = APIRouter(tags=["nursing-note"])


def _draft_response(record) -> dict[str, Any]:
    services = get_services()
    schema = services.template_registry.get(record.schema_id, record.schema_version)
    return mappers.draft_review_response(record, schema, services.draft_review_store.numeric_findings(record, schema))


@router.get("/v1/nursing-note-drafts/{draft_id}")
def get_nursing_note_draft(draft_id: str) -> dict[str, Any]:
    """확정 전 초안의 최신 revision과 후보 결정을 반환한다."""
    try:
        return _draft_response(get_services().draft_review_store.get(draft_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note draft not found") from exc


@router.post("/v1/nursing-note-drafts/{draft_id}/revisions")
def revise_nursing_note_draft(
    draft_id: str, request: DraftRevisionRequest, x_trace_id: str | None = Header(default=None),
    authenticated_user: str | None = Depends(get_authenticated_user),
) -> dict[str, Any]:
    """사용자 후보 결정·제안어 적용과 직접 편집을 새 revision으로 기록한다."""
    services = get_services()
    edited_by = resolve_actor(request.edited_by, authenticated_user, "edited_by")
    try:
        current = services.draft_review_store.get(draft_id)
        schema = services.template_registry.get(current.schema_id, current.schema_version)
        record = services.draft_review_store.revise(
            draft_id, schema, request.edited_fields, request.candidate_decisions, edited_by,
            x_trace_id or str(uuid4()), request.selected_suggestions, request.expected_revision,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note draft not found") from exc
    except (DraftAlreadyFinalized, DraftRevisionConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _draft_response(record)


def _render_pdf(note: FinalizedNursingNote, note_store: FileNursingNoteStore, renderer: NursingNotePdfRenderer) -> Path:
    schema = get_services().template_registry.get(note.schema_id, note.schema_version)
    return note_store.save_pdf(note.note_id, renderer.render(schema, note))


@router.post("/v1/nursing-note-drafts/{draft_id}/finalize", status_code=status.HTTP_201_CREATED)
def finalize_nursing_note_draft(
    draft_id: str,
    request: DraftFinalizeRequest,
    x_trace_id: str | None = Header(default=None),
    authenticated_user: str | None = Depends(get_authenticated_user),
    note_store: FileNursingNoteStore = Depends(get_note_store),
    renderer: NursingNotePdfRenderer = Depends(get_pdf_renderer),
) -> dict[str, Any]:
    """최신 revision을 불변 확정본으로 저장하고, 선택한 양식의 PDF를 생성한다."""
    services = get_services()
    finalized_by = resolve_actor(request.finalized_by, authenticated_user, "finalized_by")
    try:
        current = services.draft_review_store.get(draft_id)
        schema = services.template_registry.get(current.schema_id, current.schema_version)
        note = services.draft_review_store.finalize(
            draft_id, schema, finalized_by, x_trace_id or str(uuid4()), request.numeric_review_confirmed, request.expected_revision,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note draft not found") from exc
    except (DraftAlreadyFinalized, DraftRevisionConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NumericReviewRequired as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), **mappers.numeric_verification_response(exc.findings)}) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    note_store.save(note)
    try:
        _render_pdf(note, note_store, renderer)
        pdf_status = "ready"
    except PdfConfigurationError:
        # 확정본은 보존하고, 글꼴·양식 설정을 고친 뒤 PDF 조회 시 다시 렌더링한다.
        pdf_status = "unavailable"
    return mappers.note_response(note, pdf_status)


@router.get("/v1/nursing-notes/{note_id}")
def get_nursing_note(note_id: str, note_store: FileNursingNoteStore = Depends(get_note_store)) -> dict[str, Any]:
    """확정 간호기록과 PDF 상태를 반환한다."""
    try:
        note = note_store.get(note_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note not found") from exc
    return mappers.note_response(note, "ready" if note_store.pdf_path(note_id) else "unavailable")


@router.get("/v1/nursing-notes/{note_id}/pdf", response_class=Response)
def get_nursing_note_pdf(
    note_id: str,
    note_store: FileNursingNoteStore = Depends(get_note_store),
    renderer: NursingNotePdfRenderer = Depends(get_pdf_renderer),
) -> Response:
    """확정본 PDF를 반환한다. 저장된 PDF가 없으면 지금 렌더링해 저장한다."""
    try:
        note = note_store.get(note_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note not found") from exc
    path = note_store.pdf_path(note_id)
    if path is None:
        try:
            path = _render_pdf(note, note_store, renderer)
        except (PdfConfigurationError, KeyError) as exc:
            raise HTTPException(status_code=503, detail="PDF rendering is not configured") from exc
    filename = f"{note.schema_id}-{note.note_id}.pdf"
    return Response(path.read_bytes(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
