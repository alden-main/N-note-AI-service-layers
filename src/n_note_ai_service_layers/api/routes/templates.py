"""운영자가 양식 template 버전을 등록·승인·활성화하는 API이다."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Header, status

from n_note_ai_service_layers.api.deps import get_authenticated_user, resolve_actor
from n_note_ai_service_layers.api.mappers import request_schema, template_response
from n_note_ai_service_layers.api.schemas import TemplateActionRequest, TemplateRegistrationRequest
from n_note_ai_service_layers.api.state import get_services

router = APIRouter(prefix="/v1/form-templates", tags=["form-template"])


@router.post("", status_code=status.HTTP_201_CREATED)
def register_form_template(request: TemplateRegistrationRequest) -> dict[str, Any]:
    """운영자가 승인 전 양식 template 버전을 등록한다."""
    try:
        schema = get_services().template_registry.add(request_schema(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return template_response(schema)


@router.get("")
def list_form_templates(schema_id: str | None = None) -> list[dict[str, Any]]:
    """등록된 양식 template 버전을 반환한다."""
    return [template_response(schema) for schema in get_services().template_registry.list_versions(schema_id)]


@router.get("/{schema_id}/{version:path}")
def get_form_template(schema_id: str, version: str) -> dict[str, Any]:
    """특정 양식 template 버전을 반환한다."""
    try:
        return template_response(get_services().template_registry.get(schema_id, version))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="form template not found") from exc


@router.post("/{schema_id}/{version:path}/approve")
def approve_form_template(
    schema_id: str, version: str, request: TemplateActionRequest,
    authenticated_user: str | None = Depends(get_authenticated_user), x_trace_id: str | None = Header(default=None),
) -> dict[str, Any]:
    """구조 검증을 통과한 template 버전을 운영자 승인 상태로 전환한다."""
    services = get_services()
    operator = resolve_actor(request.operator_id, authenticated_user, "operator_id")
    try:
        schema = services.template_registry.get(schema_id, version)
        approved = services.template_review_service.approve_all(schema, operator, request.occurred_at or datetime.now(timezone.utc))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    services.audit.record(x_trace_id or "-", "template_approved", schema_id, version, actor=operator)
    return template_response(approved)


@router.post("/{schema_id}/{version:path}/activate")
def activate_form_template(
    schema_id: str, version: str, request: TemplateActionRequest,
    authenticated_user: str | None = Depends(get_authenticated_user), x_trace_id: str | None = Header(default=None),
) -> dict[str, Any]:
    """승인된 template 버전을 양식별 유일한 active 상태로 전환한다."""
    services = get_services()
    operator = resolve_actor(request.operator_id, authenticated_user, "operator_id")
    try:
        active = services.template_registry.activate(schema_id, version)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    services.audit.record(x_trace_id or "-", "template_activated", schema_id, version, actor=operator)
    return template_response(active)
