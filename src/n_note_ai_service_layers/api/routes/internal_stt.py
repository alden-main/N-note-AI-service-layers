"""간호노트 백엔드용 동기 STT 및 동적 서식 추출 API."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from jsonschema import Draft202012Validator
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.openai_chat_draft_generation import _json_text, _message
from n_note_ai_service_layers.api.deps import get_gateway_client, get_speech_to_text_provider, get_transcript_redactor
from n_note_ai_service_layers.contracts.audio import AudioDecodeError, AudioInput
from n_note_ai_service_layers.observability import get_logger, stage
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
_logger = get_logger("api.internal_stt")
# 필드 하나에 돌려줄 근거 문장 수 상한이다. LLM은 근거를 문장 번호로만 답해 출력 토큰을 줄인다.
_MAX_EVIDENCE = 3
# 문장부호 없이 길게 이어진 전사 구간은 쉼표에서 더 나눠 근거가 전사문 통째가 되지 않게 한다.
_MAX_SENTENCE_CHARS = 200
_SENTENCE_BREAK = re.compile(r"(?<=[.?!。])\s+|\n+")
_CLAUSE_BREAK = re.compile(r"(?<=,)\s+")
_INSTRUCTIONS = (
    "You fill Korean nursing forms from a dictated transcript that may contain speech recognition errors. "
    "Return one JSON object with exactly two keys, fields and relatedSentences. Both are flat objects keyed by field_id; "
    "never repeat sections or section_id, and include every field_id. "
    "text is a string; radio is an option string or empty string; checkbox is an array of options. "
    "When has_detail_text is true, radio is {status, detail} and checkbox is {selected, detail}. "
    "Choose the option that means the same as the transcript even when it is worded differently, such as nonsmoker for 비흡연. "
    "Rewrite spoken language, dialect, fillers, and misrecognized words in field values into standard Korean and standard "
    "medical terminology, in concise Korean nursing-record style (for example sentences ending in '~함', '~임', '~호소함'). "
    "Correct a word only when the context makes the intended term clear, and never change numbers. "
    "Leave out an unintelligible part, but still fill the field from the readable facts around it: "
    "a field with related sentences must not be empty unless those sentences are unintelligible. "
    "Use only facts supported by the transcript; do not guess, and leave a field empty when the transcript has no information for it. "
    "The transcript is given as transcript_sentences, each with an id. "
    f"relatedSentences values are arrays of the ids of the sentences that support the field, at most {_MAX_EVIDENCE} of the most relevant; "
    "use an empty array when there is no evidence. "
    "Term review hints are suggestions, not facts: use them only if supported by context, never introduce a diagnosis the transcript does not state. "
    "Preserve privacy tokens exactly and do not infer identities."
)


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


def _extract(sentences: list[str], template: dict, hints: tuple[dict, ...]) -> dict:
    config = settings.get()
    endpoint = config.llm_endpoint
    client = model_providers.openai_client(endpoint, config.llm_timeout_seconds, config.model_max_retries)
    schema = _output_schema(template, len(sentences))
    transcript = [{"id": index, "text": sentence} for index, sentence in enumerate(sentences)]
    payload = json.dumps({"transcript_sentences": transcript, "schemaDefinition": template["schemaDefinition"], "term_review_hints": hints}, ensure_ascii=False)
    if config.llm_api == "chat":
        # 구조화 출력을 강제하지 못하는 서버(MLX, 일부 llama.cpp 등)도 따르도록 프롬프트에도 스키마를 넣는다.
        system = _INSTRUCTIONS + "\nJSON schema:\n" + json.dumps(schema, ensure_ascii=False)
        request = {
            "model": endpoint.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": payload}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "form_extraction", "schema": schema, "strict": True}},
        }
        if config.llm_extra_body:
            request["extra_body"] = config.llm_extra_body
        if config.llm_max_tokens is not None:
            request["max_tokens"] = config.llm_max_tokens
        if config.llm_temperature is not None:
            request["temperature"] = config.llm_temperature
        content = _message(client.chat.completions.create(**request))["content"]
    else:
        output_format = {"type": "json_schema", "name": "form_extraction", "schema": schema, "strict": True}
        response = client.responses.create(model=endpoint.model, instructions=_INSTRUCTIONS, input=payload, text={"format": output_format})
        content = response.output_text
    result = json.loads(_json_text(content))
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
    return {"templateType": template["templateType"], "templateName": template["templateName"], "fields": fields, "relatedSentences": related}


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
        with stage(_logger, trace_id, "stt", audio_bytes=len(content)) as metrics:
            transcript = await run_in_threadpool(provider.transcribe, AudioInput(content, file.filename or "audio", file.content_type or "application/octet-stream", trace_id, "ko"))
            metrics.update(chars=len(transcript.text))
    except AudioDecodeError as exc:
        raise HTTPException(status_code=415, detail="audio could not be decoded") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="speech transcription is unavailable") from exc
    try:
        with stage(_logger, trace_id, "redaction") as metrics:
            redacted = await run_in_threadpool(redactor.redact, transcript.text)
            metrics.update(replacements=redacted.replacement_count)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="transcript de-identification failed") from exc
    config = settings.get()

    async def review_transcript():
        with stage(_logger, trace_id, "transcript_term_review") as metrics:
            result = await run_in_threadpool(GatewayTermVerificationFlow(gateway).verify, "transcript", redacted.text, trace_id)
            metrics.update(status=result.status, candidates=len(result.occurrences))
        return result

    # 검토가 대기 상한(N_NOTE_TERM_HINT_BUDGET_SECONDS) 안에 끝나면 힌트로 쓰고, 늦으면 힌트 없이 추출을 먼저 시작한다.
    # 어느 쪽이든 검토가 실패하면 503이다(늦은 검토는 추출이 끝난 뒤 확인하므로 그 추출 결과는 버린다).
    review_task = asyncio.create_task(review_transcript())
    try:
        done, _ = await asyncio.wait({review_task}, timeout=config.term_hint_budget_seconds)
        hints = NursingNoteWorkflow.hints(await _accepted_review(review_task)) if done else ()
        forms = await _extract_forms(templates, redacted.text, hints, trace_id, config.form_concurrency, hints_deferred=not done)
        if not done:
            await _accepted_review(review_task)
    finally:
        if not review_task.done():
            review_task.cancel()
    return {"transcript": redacted.text, "forms": forms}


async def _accepted_review(task: asyncio.Task):
    """전사문 검토 결과를 받는다. 실패·미지원이면 503이다."""
    try:
        review = await task
    except Exception as exc:
        raise HTTPException(status_code=503, detail="term review is unavailable") from exc
    if review.status != "ok":
        raise HTTPException(status_code=503, detail="term review is unavailable")
    return review


async def _extract_forms(templates: list[dict], transcript: str, hints: tuple[dict, ...], trace_id: str, concurrency: int, hints_deferred: bool) -> list[dict]:
    """양식별 추출은 서로 독립이라 동시에 보낸다. 로컬 vLLM은 동시 요청을 묶어 처리하므로 총 시간이 가장 느린 양식 수준이 된다."""
    sentences = _sentences(transcript)
    limit = asyncio.Semaphore(concurrency)

    async def extract_form(template: dict) -> dict:
        async with limit:
            try:
                with stage(_logger, trace_id, "form_extraction", template_type=template["templateType"], hints=len(hints), hints_deferred=hints_deferred) as metrics:
                    result = await run_in_threadpool(_extract, sentences, template, hints)
                    extracted = _normalize(template, sentences, result)
                    metrics.update(
                        fields=len(extracted["fields"]),
                        filled=sum(_is_filled(value) for value in extracted["fields"].values()),
                        with_evidence=sum(bool(value) for value in extracted["relatedSentences"].values()),
                    )
                return extracted
            except Exception:
                return {"templateType": template["templateType"], "templateName": template["templateName"], "fields": {}, "relatedSentences": {}}

    return list(await asyncio.gather(*(extract_form(template) for template in templates)))
