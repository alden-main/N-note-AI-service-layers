"""백엔드가 보낸 서식(``schemaDefinition``)에 맞춰 전사문에서 필드 값과 근거 문장을 추출한다(동기 STT API).

``api/routes/internal_stt.py``는 HTTP 입력·응답과 동시 실행만 맡고, 서식 검증·LLM 출력 스키마·결과 정리·
녹음 시각 채우기·수치 대조는 여기에 있다. 응답 계약은 ``docs/STT_API.md``다.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.llm_completion import StructuredOutput, json_text
from n_note_ai_service_layers.contracts.numeric_verification import NumericFinding
from n_note_ai_service_layers.observability import get_logger, stage
from n_note_ai_service_layers.services.draft_document import is_filled
from n_note_ai_service_layers.services.draft_token_scrub import has_content, scrub_value
from n_note_ai_service_layers.services.numeric_verification import NumericVerifier
from n_note_ai_service_layers.services.recording_time import format_for_field

# 필드 하나에 돌려줄 근거 문장 수 상한이다. LLM은 근거를 문장 번호로만 답해 출력 토큰을 줄인다.
MAX_EVIDENCE = 3
# 문장부호 없이 길게 이어진 전사 구간은 쉼표에서 더 나눠 근거가 전사문 통째가 되지 않게 한다.
MAX_SENTENCE_CHARS = 200
_SENTENCE_BREAK = re.compile(r"(?<=[.?!。])\s+|\n+")
_CLAUSE_BREAK = re.compile(r"(?<=,)\s+")
_FIELD_TYPES = frozenset({"text", "radio", "checkbox"})
_logger = get_logger("api.internal_stt")


def instructions() -> str:
    """서식 추출 지침이다(``prompts.toml``의 ``[form_extraction]``)."""
    return prompts.render(prompts.get().form_extraction.instructions, max_evidence=MAX_EVIDENCE)


def parse_templates(raw: str) -> list[dict]:
    """서식 목록 JSON을 읽고 구조를 검사한다. 형식이 틀리면 ``ValueError``다."""
    try:
        templates = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("templates is not JSON") from exc
    if not isinstance(templates, list):
        raise ValueError("templates must be a list")
    for template in templates:
        _check_template(template)
    return templates


def _check_template(template: Any) -> None:
    if not isinstance(template, dict) or not all(isinstance(template.get(key), str) and template[key] for key in ("templateType", "templateName")):
        raise ValueError("template needs templateType and templateName")
    schema = template.get("schemaDefinition")
    if not isinstance(schema, dict) or not isinstance(schema.get("sections"), list):
        raise ValueError("template needs schemaDefinition.sections")
    ids: set[str] = set()
    for section in schema["sections"]:
        if not isinstance(section, dict) or not isinstance(section.get("fields"), list):
            raise ValueError("section needs fields")
        for field in section["fields"]:
            if not isinstance(field, dict) or not isinstance(field.get("field_id"), str) or not field["field_id"] or field["field_id"] in ids:
                raise ValueError("field_id must be a unique non-empty string")
            ids.add(field["field_id"])
            if field.get("field_type") not in _FIELD_TYPES:
                raise ValueError("unsupported field_type")
            # 구술에 값이 없을 때 채울 기본값(선택)이다. 녹음 시각은 text 필드에만 쓴다.
            if "default_from" in field and (field["default_from"] != "recording_time" or field["field_type"] != "text"):
                raise ValueError("default_from is only recording_time on text fields")
            if field["field_type"] != "text" and (not isinstance(field.get("options"), list) or not all(isinstance(x, str) for x in field["options"])):
                raise ValueError("radio and checkbox fields need string options")


def template_fields(template: dict) -> list[dict]:
    return [field for section in template["schemaDefinition"]["sections"] for field in section["fields"]]


def split_sentences(text: str) -> list[str]:
    """전사문을 근거 단위 문장으로 나눈다. 각 문장은 전사문의 부분 문자열 그대로다."""
    sentences = []
    for part in _SENTENCE_BREAK.split(text):
        pieces = _CLAUSE_BREAK.split(part) if len(part) > MAX_SENTENCE_CHARS else [part]
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


def output_schema(template: dict, sentence_count: int) -> dict:
    """서식 스키마를 LLM 출력 JSON 스키마로 바꾼다. field_id를 ``fields`` 바로 아래에 평면으로 둔다.

    ``relatedSentences``는 문장 번호(0 ~ ``sentence_count - 1``) 배열이다. 응답에서는 문장 원문으로 바꾼다.
    """
    fields = template_fields(template)
    ids = [field["field_id"] for field in fields]
    values = {}
    for field in fields:
        label = " - ".join(str(field[key]) for key in ("label", "sub_label") if field.get(key))
        values[field["field_id"]] = {**_value_schema(field), "description": label} if label else _value_schema(field)
    sentence_id = {"type": "integer", "enum": list(range(sentence_count))} if sentence_count else {"type": "integer"}
    related = {key: {"type": "array", "items": sentence_id, "maxItems": MAX_EVIDENCE if sentence_count else 0} for key in ids}
    return {
        "type": "object",
        "properties": {
            "fields": {"type": "object", "properties": values, "required": ids, "additionalProperties": False},
            "relatedSentences": {"type": "object", "properties": related, "required": ids, "additionalProperties": False},
        },
        "required": ["fields", "relatedSentences"],
        "additionalProperties": False,
    }


def extract(sentences: list[str], template: dict, hints: tuple[dict, ...], reference_time: datetime | None = None) -> dict:
    """LLM 한 번으로 서식을 추출한다. 구조가 스키마와 다르면 ``ValueError``다."""
    schema = output_schema(template, len(sentences))
    transcript = [{"id": index, "text": sentence} for index, sentence in enumerate(sentences)]
    request: dict[str, Any] = {"transcript_sentences": transcript, "schemaDefinition": template["schemaDefinition"], "term_review_hints": hints}
    if reference_time is not None:
        request["reference_time"] = reference_time.isoformat(timespec="minutes")
    payload = json.dumps(request, ensure_ascii=False)
    # 초안 생성과 같은 요청기다(API 형식·store=False·temperature 등 옵션·Chat의 스키마 프롬프트를 공유한다).
    content = model_providers.structured_completion().complete(instructions(), payload, StructuredOutput("form_extraction", schema))
    result = json.loads(json_text(content))
    # 구조가 다르면 빈 서식으로 오인하지 않고 실패로 처리한다. options·문장 번호 범위 밖의 값과 근거 초과분은 normalize가 정리한다.
    if any(error.validator not in {"enum", "maxItems"} for error in Draft202012Validator(schema).iter_errors(result)):
        raise ValueError("extraction did not follow the form schema")
    return result


def normalize(template: dict, sentences: list[str], result: dict) -> dict:
    """LLM 결과를 응답 계약 모양으로 정리한다(options 밖의 값·범위 밖 문장 번호·보호 토큰 제거)."""
    values = result.get("fields") if isinstance(result.get("fields"), dict) else {}
    evidence = result.get("relatedSentences") if isinstance(result.get("relatedSentences"), dict) else {}
    fields, related = {}, {}
    for field in template_fields(template):
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
            related[key] = [sentences[i] for i in list(dict.fromkeys(valid))[:MAX_EVIDENCE]]
    # 프롬프트로 막아도 남은 보호 토큰은 초안 생성과 같은 규칙으로 지운다. 근거 문장은 전사문 원문 그대로 둔다.
    fields = {key: _blank_without_content(scrub_value(value)) for key, value in fields.items()}
    return {"templateType": template["templateType"], "templateName": template["templateName"], "fields": fields, "relatedSentences": related}


def _blank_without_content(value: Any) -> Any:
    """토큰을 지운 뒤 "환자명:"·"."만 남은 문자열(상세 텍스트 포함)은 빈 값으로 둔다."""
    if isinstance(value, str):
        return value if has_content(value) else ""
    if isinstance(value, dict) and isinstance(value.get("detail"), str) and not has_content(value["detail"]):
        return {**value, "detail": ""}
    return value


def fill_recording_time(template: dict, extracted: dict, recorded_at: datetime) -> set[str]:
    """``default_from: "recording_time"``인 빈 text 필드를 녹음 시각(``YYYY-MM-DD HH:MM``)으로 채우고 채운 field_id를 돌려준다.

    비동기 초안 작업의 ``nursing_note_workflow.fill_recording_time``과 같은 규칙(빈 값 판정·표시 형식)이다.
    """
    filled = set()
    for field in template_fields(template):
        key = field["field_id"]
        if field.get("default_from") == "recording_time" and not is_filled(extracted["fields"].get(key)):
            extracted["fields"][key] = format_for_field(recorded_at, "datetime")
            filled.add(key)
    return filled


def _finding(item: NumericFinding) -> dict[str, Any]:
    return {"findingId": item.finding_id, "fieldId": item.field_id, "startOffset": item.start_offset,
            "endOffset": item.end_offset, "text": item.text, "reason": item.reason}


def numeric_verification(template: dict, fields: dict, transcript: str, recorded_at: datetime, defaulted: set[str]) -> dict:
    """서식 값의 수치 대조다: 출처에 없는 수치(text 값·상세 텍스트)와 전사문에 있는데 서식에서 빠진 수치.

    서비스가 녹음 시각으로 채운 필드는 출처 대조에서 뺀다. 추출에 실패한 서식(빈 ``fields``)은 ``skipped``다.
    """
    if not fields:
        return {"status": "skipped", "findings": []}
    values = []
    for field in template_fields(template):
        key = field["field_id"]
        if key in defaulted:
            continue
        value = fields.get(key)
        text = value.get("detail") if isinstance(value, dict) else value if field["field_type"] == "text" else None
        values.append((key, "text", text))
    verifier = NumericVerifier()
    findings = verifier.values_not_in_source(values, (transcript,), recorded_at) + verifier.omissions(transcript, fields)
    return {"status": "needs_review" if findings else "ok", "findings": [_finding(item) for item in findings]}


def _has_answer(value: object) -> bool:
    """로그용 작성 여부다. radio·checkbox 사전은 값이 하나라도 있으면 작성으로 센다(``draft_document.is_filled``와 다르다)."""
    return any(value.values()) if isinstance(value, dict) else bool(value)


def empty_form(template: dict) -> dict:
    """추출에 실패한 서식의 응답이다. 전사문은 그대로 돌려주고 서식만 비운다."""
    return {"templateType": template["templateType"], "templateName": template["templateName"], "fields": {}, "relatedSentences": {},
            "numericVerification": {"status": "skipped", "findings": []}}


def extract_form(
    template: dict, sentences: list[str], transcript: str, hints: tuple[dict, ...], trace_id: str, recorded_at: datetime, hints_deferred: bool,
) -> dict:
    """서식 하나의 전 단계다: 추출 → 정리 → 녹음 시각 채우기 → 수치 대조. 실패하면 빈 서식이다(작업 스레드에서 부른다)."""
    try:
        with stage(_logger, trace_id, "form_extraction", template_type=template["templateType"], hints=len(hints), hints_deferred=hints_deferred) as metrics:
            extracted = normalize(template, sentences, extract(sentences, template, hints, recorded_at))
            defaulted = fill_recording_time(template, extracted, recorded_at)
            extracted["numericVerification"] = numeric_verification(template, extracted["fields"], transcript, recorded_at, defaulted)
            metrics.update(
                fields=len(extracted["fields"]),
                filled=sum(_has_answer(value) for value in extracted["fields"].values()),
                with_evidence=sum(bool(value) for value in extracted["relatedSentences"].values()),
            )
        return extracted
    except Exception:
        return empty_form(template)
