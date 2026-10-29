"""간호노트 백엔드용 동기 STT 및 동적 서식 추출 API."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.openai_chat_draft_generation import _json_text, _message
from n_note_ai_service_layers.api.deps import get_gateway_client, get_speech_to_text_provider, get_transcript_redactor
from n_note_ai_service_layers.contracts.audio import AudioInput
from n_note_ai_service_layers.services.gateway_term_verification import GatewayTermVerificationFlow
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow

class FormResult(BaseModel):
    templateType: str
    templateName: str
    fields: dict
    relatedSentences: dict[str, list[str]]


class InternalSttResponse(BaseModel):
    transcript: str
    forms: list[FormResult]


router = APIRouter(tags=["internal-stt"])
_MAX_AUDIO = 50 * 1024 * 1024
_EXTENSIONS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".webm"}


def _templates(raw: str) -> list[dict]:
    try:
        templates = json.loads(raw)
        if not isinstance(templates, list):
            raise ValueError
        for template in templates:
            if not isinstance(template, dict) or not all(isinstance(template.get(k), str) and template[k] for k in ("templateType", "templateName")):
                raise ValueError
            schema = template.get("schemaDefinition")
            if not isinstance(schema, dict) or not isinstance(schema.get("sections"), list):
                raise ValueError
            ids = set()
            for section in schema["sections"]:
                if not isinstance(section, dict) or not isinstance(section.get("fields"), list):
                    raise ValueError
                for field in section["fields"]:
                    if not isinstance(field, dict) or not isinstance(field.get("field_id"), str) or not field["field_id"] or field["field_id"] in ids:
                        raise ValueError
                    ids.add(field["field_id"])
                    if field.get("field_type") not in {"text", "radio", "checkbox"}:
                        raise ValueError
                    if field["field_type"] != "text" and (not isinstance(field.get("options"), list) or not all(isinstance(x, str) for x in field["options"])):
                        raise ValueError
        return templates
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="invalid templates JSON or schema") from None


def _extract(transcript: str, template: dict, hints: tuple[dict, ...]) -> dict:
    endpoint = settings.llm_endpoint()
    client = model_providers.openai_client(endpoint, settings.llm_timeout_seconds(), settings.model_max_retries())
    instructions = (
        "Extract fields from the transcript using the supplied schema. Return JSON with fields and relatedSentences objects. "
        "Include every field_id. text is a string; radio is an option string or empty string; checkbox is an array of options. "
        "When has_detail_text is true, radio is {status, detail} and checkbox is {selected, detail}. "
        "Use only facts supported by the transcript; do not guess. Related sentences must be exact substrings of the transcript. "
        "Term review hints are suggestions, not facts: use them only if supported by context, never introduce a diagnosis. "
        "Preserve privacy tokens exactly and do not infer identities."
    )
    payload = json.dumps({"transcript": transcript, "schemaDefinition": template["schemaDefinition"], "term_review_hints": hints}, ensure_ascii=False)
    if settings.llm_api() == "chat":
        request = {"model": endpoint.model, "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": payload}]}
        if settings.llm_extra_body():
            request["extra_body"] = settings.llm_extra_body()
        if settings.llm_max_tokens() is not None:
            request["max_tokens"] = settings.llm_max_tokens()
        if settings.llm_temperature() is not None:
            request["temperature"] = settings.llm_temperature()
        content = _message(client.chat.completions.create(**request))["content"]
    else:
        response = client.responses.create(model=endpoint.model, instructions=instructions, input=payload)
        content = response.output_text
    result = json.loads(_json_text(content))
    if not isinstance(result, dict):
        raise ValueError("invalid extraction")
    return result


def _normalize(template: dict, transcript: str, result: dict) -> dict:
    values = result.get("fields") if isinstance(result.get("fields"), dict) else {}
    evidence = result.get("relatedSentences") if isinstance(result.get("relatedSentences"), dict) else {}
    fields, related = {}, {}
    for section in template["schemaDefinition"]["sections"]:
        for field in section["fields"]:
            key, kind = field["field_id"], field["field_type"]
            value = values.get(key)
            options = field.get("options", [])
            detail_enabled = field.get("has_detail_text") is True
            if kind == "text":
                fields[key] = value if isinstance(value, str) else ""
            elif kind == "radio":
                choice = value.get("status") if detail_enabled and isinstance(value, dict) else value
                choice = choice if isinstance(choice, str) and choice in options else ""
                detail = value.get("detail") if detail_enabled and isinstance(value, dict) else ""
                fields[key] = {"status": choice, "detail": detail if choice and choice != "없음" and isinstance(detail, str) else ""} if detail_enabled else choice
            else:
                selected = value.get("selected") if detail_enabled and isinstance(value, dict) else value
                selected = [x for x in selected if isinstance(x, str) and x in options] if isinstance(selected, list) else []
                detail = value.get("detail") if detail_enabled and isinstance(value, dict) else ""
                fields[key] = {"selected": selected, "detail": detail if selected and selected != ["없음"] and isinstance(detail, str) else ""} if detail_enabled else selected
            sentences = evidence.get(key)
            if isinstance(sentences, list):
                related[key] = [s for s in sentences if isinstance(s, str) and s and s in transcript]
    return {"templateType": template["templateType"], "templateName": template["templateName"], "fields": fields, "relatedSentences": related}


@router.post(
    "/internal/v1",
    summary="음성 전사 및 서식 필드 추출",
    response_model=InternalSttResponse,
    description="음성 전사 후 비식별화하고 Gateway 용어 검토 후보를 참고해 서식을 생성합니다. templates는 JSON 문자열 또는 application/json 파일로 전달할 수 있습니다. Gateway 검토가 불가능하면 503을 반환합니다.",
    openapi_extra={
        "requestBody": {"content": {"multipart/form-data": {"schema": {
            "type": "object",
            "required": ["file", "templates"],
            "properties": {
                "file": {"type": "string", "format": "binary", "description": "음성 파일 (최대 50 MiB)"},
                "templates": {"type": "string", "description": "서식 목록을 담은 JSON 배열", "example": '[{"templateType":"HAND_OVER","templateName":"인계기록지","schemaDefinition":{"sections":[]}}]'},
            },
        }}}},
    },
)
async def internal_transcription(request: Request, provider=Depends(get_speech_to_text_provider), redactor=Depends(get_transcript_redactor), gateway=Depends(get_gateway_client)) -> dict:
    form = await request.form()
    parts = form.getlist("templates")
    if len(parts) != 1 or len(form.getlist("file")) != 1:
        raise HTTPException(status_code=422, detail="exactly one file and templates part are required")
    file = form.get("file")
    if not isinstance(file, StarletteUploadFile):
        raise HTTPException(status_code=422, detail="file must be an upload")
    part = parts[0]
    if isinstance(part, StarletteUploadFile):
        raw = await part.read()
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="templates must be UTF-8 JSON") from None
    else:
        raw = part
    templates = _templates(raw)
    if Path(file.filename or "").suffix.lower() not in _EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported audio file extension")
    content = await file.read(_MAX_AUDIO + 1)
    if not content:
        raise HTTPException(status_code=422, detail="audio file is empty")
    if len(content) > _MAX_AUDIO:
        raise HTTPException(status_code=413, detail="audio file exceeds 50 MiB limit")
    trace_id = str(uuid4())
    try:
        transcript = await run_in_threadpool(provider.transcribe, AudioInput(content, file.filename or "audio", file.content_type or "application/octet-stream", trace_id, "ko"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="speech transcription is unavailable") from exc
    try:
        redacted = await run_in_threadpool(redactor.redact, transcript.text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="transcript de-identification failed") from exc
    try:
        review = await run_in_threadpool(GatewayTermVerificationFlow(gateway).verify, "transcript", redacted.text, trace_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="term review is unavailable") from exc
    if review.status != "ok":
        raise HTTPException(status_code=503, detail="term review is unavailable")
    hints = NursingNoteWorkflow.hints(review)
    forms = []
    for template in templates:
        try:
            result = await run_in_threadpool(_extract, redacted.text, template, hints)
            forms.append(_normalize(template, redacted.text, result))
        except Exception:
            forms.append({"templateType": template["templateType"], "templateName": template["templateName"], "fields": {}, "relatedSentences": {}})
    return {"transcript": redacted.text, "forms": forms}
