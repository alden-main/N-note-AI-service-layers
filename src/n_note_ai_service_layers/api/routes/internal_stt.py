"""간호노트 백엔드용 동기 STT 및 동적 서식 추출 API."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from jsonschema import Draft202012Validator
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel

from n_note_ai_service_layers import prompts, settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.llm_completion import StructuredOutput, json_text
from n_note_ai_service_layers.api.deps import get_transcript_preparation, read_audio
from n_note_ai_service_layers.contracts.audio import AudioInput
from n_note_ai_service_layers.observability import get_logger, span, stage
from n_note_ai_service_layers.adapters.audio_preprocessing import audio_creation_time
from n_note_ai_service_layers.contracts.numeric_verification import NumericFinding
from n_note_ai_service_layers.services.clock import now_local
from n_note_ai_service_layers.services.draft_token_scrub import has_content, scrub_value as _scrub_value
from n_note_ai_service_layers.services.numeric_verification import NumericVerifier
from n_note_ai_service_layers.services.recording_time import format_for_field, parse_recorded_at, resolve_recording_time
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.nursing_note_workflow import TranscriptPreparation, WorkflowStageError

class NumericVerification(BaseModel):
    status: str
    findings: list[dict[str, Any]]


class FormResult(BaseModel):
    templateType: str
    templateName: str
    fields: dict
    relatedSentences: dict[str, list[str]]
    numericVerification: NumericVerification


class InternalSttResponse(BaseModel):
    transcript: str
    recordedAt: str
    forms: list[FormResult]


router = APIRouter(tags=["internal-stt"])
_MAX_AUDIO = 50 * 1024 * 1024
_EXTENSIONS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".webm"}
_logger = get_logger("api.internal_stt")
# 공통 단계 실패를 이 API의 HTTP 응답으로 바꾼다. 본문·예외 메시지는 노출하지 않는다.
_PREPARATION_FAILURES = {
    "audio_decode": (415, "audio could not be decoded"),
    "stt": (503, "speech transcription is unavailable"),
    "redaction": (503, "transcript de-identification failed"),
}
# 필드 하나에 돌려줄 근거 문장 수 상한이다. LLM은 근거를 문장 번호로만 답해 출력 토큰을 줄인다.
_MAX_EVIDENCE = 3
# 문장부호 없이 길게 이어진 전사 구간은 쉼표에서 더 나눠 근거가 전사문 통째가 되지 않게 한다.
_MAX_SENTENCE_CHARS = 200
_SENTENCE_BREAK = re.compile(r"(?<=[.?!。])\s+|\n+")
_CLAUSE_BREAK = re.compile(r"(?<=,)\s+")


def _instructions() -> str:
    """서식 추출 지침이다(``prompts.toml``의 ``[form_extraction]``)."""
    return prompts.render(prompts.get().form_extraction.instructions, max_evidence=_MAX_EVIDENCE)


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
                    # 구술에 값이 없을 때 채울 기본값(선택)이다. 녹음 시각은 text 필드에만 쓴다.
                    if "default_from" in field and (field["default_from"] != "recording_time" or field["field_type"] != "text"):
                        raise ValueError
                    if field["field_type"] != "text" and (not isinstance(field.get("options"), list) or not all(isinstance(x, str) for x in field["options"])):
                        raise ValueError
        return templates
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="invalid templates JSON or schema") from None


def _sentences(text: str) -> list[str]:
    """전사문을 근거 단위 문장으로 나눈다. 각 문장은 전사문의 부분 문자열 그대로다."""
    sentences = []
    for part in _SENTENCE_BREAK.split(text):
        pieces = _CLAUSE_BREAK.split(part) if len(part) > _MAX_SENTENCE_CHARS else [part]
        sentences.extend(piece.strip() for piece in pieces if piece.strip())
    return sentences


def _value_schema(field: dict) -> dict:
    """응답 계약(docs/STT_API.md 3.1)의 필드 값 모양을 JSON 스키마로 만든다."""
    kind = field["field_type"]
    if kind == "text":
        return {"type": "string"}
    if kind == "radio":
        key, value = "status", {"type": "string", "enum": list(dict.fromkeys([*field["options"], ""]))}
    else:
        options = list(dict.fromkeys(field["options"]))
        key, value = "selected", {"type": "array", "items": {"type": "string", "enum": options} if options else {"type": "string"}}
    if field.get("has_detail_text") is not True:
        return value
    return {"type": "object", "properties": {key: value, "detail": {"type": "string"}}, "required": [key, "detail"], "additionalProperties": False}


def _output_schema(template: dict, sentence_count: int) -> dict:
    """서식 스키마를 LLM 출력 JSON 스키마로 바꾼다. field_id를 ``fields`` 바로 아래에 평면으로 둔다.

    ``relatedSentences``는 문장 번호(0 ~ ``sentence_count - 1``) 배열이다. 응답에서는 문장 원문으로 바꾼다.
    """
    fields = [field for section in template["schemaDefinition"]["sections"] for field in section["fields"]]
    ids = [field["field_id"] for field in fields]
    values = {}
    for field in fields:
        label = " - ".join(str(field[key]) for key in ("label", "sub_label") if field.get(key))
        values[field["field_id"]] = {**_value_schema(field), "description": label} if label else _value_schema(field)
    sentence_id = {"type": "integer", "enum": list(range(sentence_count))} if sentence_count else {"type": "integer"}
    related = {key: {"type": "array", "items": sentence_id, "maxItems": _MAX_EVIDENCE if sentence_count else 0} for key in ids}
    return {
        "type": "object",
        "properties": {
            "fields": {"type": "object", "properties": values, "required": ids, "additionalProperties": False},
            "relatedSentences": {"type": "object", "properties": related, "required": ids, "additionalProperties": False},
        },
        "required": ["fields", "relatedSentences"],
        "additionalProperties": False,
    }


def _extract(sentences: list[str], template: dict, hints: tuple[dict, ...], reference_time: datetime | None = None) -> dict:
    schema = _output_schema(template, len(sentences))
    transcript = [{"id": index, "text": sentence} for index, sentence in enumerate(sentences)]
    request: dict[str, Any] = {"transcript_sentences": transcript, "schemaDefinition": template["schemaDefinition"], "term_review_hints": hints}
    if reference_time is not None:
        request["reference_time"] = reference_time.isoformat(timespec="minutes")
    payload = json.dumps(request, ensure_ascii=False)
    # 초안 생성과 같은 요청기다(API 형식·store=False·temperature 등 옵션·Chat의 스키마 프롬프트를 공유한다).
    content = model_providers.structured_completion().complete(_instructions(), payload, StructuredOutput("form_extraction", schema))
    result = json.loads(json_text(content))
    # 구조가 다르면 빈 서식으로 오인하지 않고 실패로 처리한다. options·문장 번호 범위 밖의 값과 근거 초과분은 _normalize가 정리한다.
    if any(error.validator not in {"enum", "maxItems"} for error in Draft202012Validator(schema).iter_errors(result)):
        raise ValueError("extraction did not follow the form schema")
    return result


def _normalize(template: dict, sentences: list[str], result: dict) -> dict:
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
            ids = evidence.get(key)
            if isinstance(ids, list):
                valid = (i for i in ids if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(sentences))
                related[key] = [sentences[i] for i in list(dict.fromkeys(valid))[:_MAX_EVIDENCE]]
    # 프롬프트로 막아도 남은 보호 토큰은 초안 생성과 같은 규칙으로 지운다. 근거 문장은 전사문 원문 그대로 둔다.
    fields = {key: _blank_without_content(_scrub_value(value)) for key, value in fields.items()}
    return {"templateType": template["templateType"], "templateName": template["templateName"], "fields": fields, "relatedSentences": related}


def _blank_without_content(value: Any) -> Any:
    """토큰을 지운 뒤 "환자명:"·"."만 남은 문자열(상세 텍스트 포함)은 빈 값으로 둔다."""
    if isinstance(value, str):
        return value if has_content(value) else ""
    if isinstance(value, dict) and isinstance(value.get("detail"), str) and not has_content(value["detail"]):
        return {**value, "detail": ""}
    return value


def _template_fields(template: dict) -> list[dict]:
    return [field for section in template["schemaDefinition"]["sections"] for field in section["fields"]]


def _fill_recording_time(template: dict, extracted: dict, recorded_at: datetime) -> set[str]:
    """``default_from: "recording_time"``인 빈 text 필드를 녹음 시각(``YYYY-MM-DD HH:MM``)으로 채우고 채운 field_id를 돌려준다."""
    filled = set()
    for field in _template_fields(template):
        key = field["field_id"]
        if field.get("default_from") == "recording_time" and not extracted["fields"].get(key):
            extracted["fields"][key] = format_for_field(recorded_at, "datetime")
            filled.add(key)
    return filled


def _finding(item: NumericFinding) -> dict[str, Any]:
    return {"findingId": item.finding_id, "fieldId": item.field_id, "startOffset": item.start_offset,
            "endOffset": item.end_offset, "text": item.text, "reason": item.reason}


def _numeric_verification(template: dict, fields: dict, transcript: str, recorded_at: datetime, defaulted: set[str]) -> dict:
    """서식 값의 수치 대조다: 출처에 없는 수치(text 값·상세 텍스트)와 전사문에 있는데 서식에서 빠진 수치.

    서비스가 녹음 시각으로 채운 필드는 출처 대조에서 뺀다. 추출에 실패한 서식(빈 ``fields``)은 ``skipped``다.
    """
    if not fields:
        return {"status": "skipped", "findings": []}
    values = []
    for field in _template_fields(template):
        key = field["field_id"]
        if key in defaulted:
            continue
        value = fields.get(key)
        text = value.get("detail") if isinstance(value, dict) else value if field["field_type"] == "text" else None
        values.append((key, "text", text))
    verifier = NumericVerifier()
    findings = verifier.values_not_in_source(values, (transcript,), recorded_at) + verifier.omissions(transcript, fields)
    return {"status": "needs_review" if findings else "ok", "findings": [_finding(item) for item in findings]}


def _recorded_at_part(form) -> datetime | None:
    """선택 part ``recorded_at``(녹음 시작 시각, ISO 8601)이다. 시간대가 없으면 병동 시간대로 본다."""
    parts = form.getlist("recorded_at")
    if len(parts) > 1 or (parts and not isinstance(parts[0], str)):
        raise HTTPException(status_code=422, detail="recorded_at must be a single ISO 8601 text part")
    if not parts or not parts[0].strip():
        return None
    try:
        return parse_recorded_at(parts[0])
    except ValueError:
        raise HTTPException(status_code=422, detail="recorded_at must be an ISO 8601 date-time") from None


def _is_filled(value: object) -> bool:
    return any(value.values()) if isinstance(value, dict) else bool(value)


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
async def internal_transcription(request: Request, preparation: TranscriptPreparation = Depends(get_transcript_preparation)) -> dict:
    # 요청 하나가 추적 하나다. 공통 단계·양식별 추출 span이 이 아래에 놓인다(run_in_threadpool은 문맥을 넘긴다).
    with span("internal_stt"):
        return await _internal_transcription(request, preparation)


async def _internal_transcription(request: Request, preparation: TranscriptPreparation) -> dict:
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
    received_at = now_local()
    requested_time = _recorded_at_part(form)
    if Path(file.filename or "").suffix.lower() not in _EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported audio file extension")
    # 백엔드 계약(docs/STT_API.md)은 확장자와 50 MiB 상한이다.
    content = await read_audio(file, _MAX_AUDIO)
    trace_id = str(uuid4())
    # 녹음 시각: 요청 recorded_at → 음성 메타데이터 생성 시각 → 접수 시각(초안 작업 API와 같은 규칙).
    metadata_time = None if requested_time else await run_in_threadpool(audio_creation_time, content)
    recording = resolve_recording_time(requested_time, metadata_time, received_at)
    audio = AudioInput(content, file.filename or "audio", file.content_type or "application/octet-stream", trace_id, "ko", recorded_at=recording.value)
    # STT → 비식별화 → 전사문 검토(힌트 대기 상한까지)는 비동기 초안 작업과 같은 공통 단계다.
    try:
        prepared = await run_in_threadpool(preparation.prepare, audio, (), lambda _stage: None)
    except WorkflowStageError as exc:
        status_code, detail = _PREPARATION_FAILURES.get(exc.stage, (503, "speech transcription is unavailable"))
        raise HTTPException(status_code=status_code, detail=detail) from exc
    # 이 API는 검토가 성공해야 결과를 돌려준다(docs/STT_API.md). 상한 안에 끝난 검토는 추출 전에 확인하고,
    # 늦은 검토는 추출을 먼저 시작한 뒤 확인한다(실패면 그 추출 결과는 버린다).
    reviewed_in_time = prepared.review.done()
    if reviewed_in_time:
        _require_review(prepared.review.result())
    forms = await _extract_forms(
        templates, prepared.transcript.text, prepared.hints, trace_id, settings.get().form_concurrency,
        hints_deferred=not reviewed_in_time, recorded_at=prepared.reference_time,
    )
    if not reviewed_in_time:
        _require_review(await asyncio.wrap_future(prepared.review))
    return {"transcript": prepared.transcript.text, "recordedAt": prepared.reference_time.isoformat(), "forms": forms}


def _require_review(review: GatewayTermVerificationResult) -> None:
    """전사문 검토가 성공하지 않았으면(장애·미지원) 503이다."""
    if review.status != "ok":
        raise HTTPException(status_code=503, detail="term review is unavailable")


async def _extract_forms(
    templates: list[dict], transcript: str, hints: tuple[dict, ...], trace_id: str, concurrency: int, hints_deferred: bool,
    recorded_at: datetime | None = None,
) -> list[dict]:
    """양식별 추출은 서로 독립이라 동시에 보낸다. 로컬 vLLM은 동시 요청을 묶어 처리하므로 총 시간이 가장 느린 양식 수준이 된다."""
    sentences = _sentences(transcript)
    limit = asyncio.Semaphore(concurrency)
    recorded_at = recorded_at or now_local()

    async def extract_form(template: dict) -> dict:
        async with limit:
            try:
                with stage(_logger, trace_id, "form_extraction", template_type=template["templateType"], hints=len(hints), hints_deferred=hints_deferred) as metrics:
                    result = await run_in_threadpool(_extract, sentences, template, hints, recorded_at)
                    extracted = _normalize(template, sentences, result)
                    defaulted = _fill_recording_time(template, extracted, recorded_at)
                    extracted["numericVerification"] = _numeric_verification(template, extracted["fields"], transcript, recorded_at, defaulted)
                    metrics.update(
                        fields=len(extracted["fields"]),
                        filled=sum(_is_filled(value) for value in extracted["fields"].values()),
                        with_evidence=sum(bool(value) for value in extracted["relatedSentences"].values()),
                    )
                return extracted
            except Exception:
                return {"templateType": template["templateType"], "templateName": template["templateName"], "fields": {}, "relatedSentences": {},
                        "numericVerification": {"status": "skipped", "findings": []}}

    return list(await asyncio.gather(*(extract_form(template) for template in templates)))
