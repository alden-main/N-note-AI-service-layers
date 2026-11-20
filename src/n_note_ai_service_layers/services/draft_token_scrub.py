"""LLM 초안에 남은 보호 토큰([[TYPE_n]])을 지운다.

환자 신원은 EMR에서 확정되므로 초안 문장에 "[[PERSON_1]]님은" 같은 주어가 필요 없다. 프롬프트로 막지만
모델이 어길 수 있으므로 화면·PDF로 가기 전에 한 번 더 지운다.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.privacy import PRIVACY_TOKEN

# 토큰 + 선택 호칭(님·씨) + 바로 붙은 조사를 한 덩어리로 지운다. 조사 뒤에 한글이 이어지면(이틀 등) 조사로 보지 않는다.
_TOKEN_PHRASE = re.compile(
    PRIVACY_TOKEN.pattern +
    r"(?:\s?(?:님|씨))?"
    r"(?:(?:께서는|께서|에게는|에게|한테|께|이고|이며|은|는|이|가|을|를|의|와|과|도)(?![가-힣]))?"
    r"\s*"
)
# 비식별화 라벨 규칙(transcript_redaction)이 이름 앞에 남기는 환자 신원 라벨이다. "환자명: [[...]]"에서 토큰만 지우면
# "환자명:"만 남으므로 라벨까지 한 덩어리로 지운다. 토큰이 바로 뒤에 올 때만 지우므로 일반 문장의 "이름"은 남는다.
_PATIENT_LABEL = r"(?:(?:환자\s?)?(?:성함|이름|성명)|환자\s?명)(?:\s?(?:은|는|이요))?\s*[:：]?\s*"
_LABELED_TOKEN_PHRASE = re.compile(_PATIENT_LABEL + _TOKEN_PHRASE.pattern)
# 보호자 이름은 역할로 바꾼다("보호자명: [[...]]에게 설명함" → "보호자에게 설명함"). 조사는 그대로 둔다.
_GUARDIAN_TOKEN = re.compile(r"보호자\s?(?:명|성함|이름)?\s*[:：]?\s*" + PRIVACY_TOKEN.pattern + r"(?:\s?(?:님|씨))?")
_SPACES = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,!?])")
# 지운 뒤 글자·숫자가 하나도 없으면("환자명:", ".") 내용이 없는 값이다.
_CONTENT = re.compile(r"[가-힣A-Za-z0-9]")


def scrub_text(text: str) -> str:
    """문자열에서 보호 토큰 구절(앞의 신원 라벨 포함)을 지우고 남은 공백을 정리한다."""
    scrubbed = _TOKEN_PHRASE.sub("", _LABELED_TOKEN_PHRASE.sub("", _GUARDIAN_TOKEN.sub("보호자", text)))
    if scrubbed == text:
        return text
    scrubbed = _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", scrubbed))
    return "\n".join(line.strip() for line in scrubbed.split("\n")).strip()


def has_content(text: str) -> bool:
    """글자·숫자가 하나라도 있는지다. 토큰을 지운 뒤 "환자명:"·"."만 남은 값은 내용이 없다."""
    return bool(_CONTENT.search(text))


def scrub_value(value: Any) -> Any:
    """문자열·목록·사전 안의 문자열에서 보호 토큰 구절을 지운다(구조는 유지)."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [scrub_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_value(item) for item in value)
    if isinstance(value, dict):
        return {key: scrub_value(item) for key, item in value.items()}
    return value


def scrub_privacy_tokens(draft: DraftResult) -> DraftResult:
    """초안 필드 값에서 보호 토큰을 지운다.

    입력: LLM이 만든 ``DraftResult``이다.
    출력: 토큰이 없으면 같은 객체를, 있으면 지운 사본을 돌려준다. 지우고 비거나 문장부호만 남은 필드는 뺀다.
    토큰을 지운 경우 문장이 어색해질 수 있으므로 ``needs_confirmation``을 켠다.
    """
    fields: dict[str, Any] = {}
    changed = False
    for field_id, value in draft.proposed_fields.items():
        scrubbed = scrub_value(value)
        if scrubbed != value:
            changed = True
            if scrubbed == "" or (isinstance(scrubbed, str) and not has_content(scrubbed)):
                continue
        fields[field_id] = scrubbed
    if not changed:
        return draft
    evidence = {key: refs for key, refs in draft.evidence_refs.items() if key in fields}
    return replace(draft, proposed_fields=fields, evidence_refs=evidence, needs_confirmation=True)
