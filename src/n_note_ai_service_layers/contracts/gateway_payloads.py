"""Gateway(upstream MCP)가 돌려주는 구조화 결과의 입력 계약이다.

표준어·의학용어 검토와 보고서 검증 결과를 Pydantic 모델로 검사한다. 모르는 필드는 무시하고, 타입이 틀린 값은
``pydantic.ValidationError``가 된다. 호출자는 이 오류를 "도구 결과 계약 불일치(unsupported)"로 다룬다.
정수·실수는 ``bool``과 문자열 숫자를 받지 않는다(strict).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, StrictFloat, StrictInt, StrictStr, TypeAdapter


def _text_or_none(value: Any) -> Any:
    """문자열이 아닌 설명(reason)은 없는 것으로 본다(결과 전체를 거부하지 않는다)."""
    return value if isinstance(value, str) else None


OptionalText = Annotated[str | None, BeforeValidator(_text_or_none)]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


# --- 표준국어 검토(korean__review_standard_korean_text) ---

class KoreanCandidate(_Payload):
    replacement: StrictStr
    confidence: StrictInt | StrictFloat | None = None
    reason: OptionalText = None


class KoreanFinding(_Payload):
    start: StrictInt
    end: StrictInt
    source_text: StrictStr
    analysis_form: StrictStr
    candidates: list[KoreanCandidate]
    status: str | None = None


# --- 의학용어 문서 분석(medic__analyze_document_terms) ---

class MedicalTermName(_Payload):
    kor_name: StrictStr
    eng_name: StrictStr


class MedicalCandidate(_Payload):
    """``automatic_correction``이 ``false``가 아닌 후보는 계약 위반이다(자동 교정 금지)."""

    start_offset: StrictInt
    end_offset: StrictInt
    source_text: StrictStr
    automatic_correction: Literal[False]
    candidate_type: str | None = None
    terms: list[MedicalTermName]


class MedicalSuggestion(_Payload):
    start_offset: StrictInt
    end_offset: StrictInt
    source_text: StrictStr
    review_status: str | None = None
    reason: OptionalText = None
    recommended_term: MedicalTermName | None = None


# --- 보고서 검증(report__evaluate_report_against_standard) ---

ReportStatus = Literal["satisfied", "missing", "insufficient", "uncertain"]


class ReportElement(_Payload):
    id: StrictStr
    status: ReportStatus
    warnings: list[Any] = []


class ReportFinding(_Payload):
    status: ReportStatus
    warnings: list[Any] = []


class ReportCriterion(_Payload):
    criterion_id: Any = None
    description: Any = None
    finding: ReportFinding
    elements: list[ReportElement]


class ReportEvaluation(_Payload):
    criteria: list[ReportCriterion]


KOREAN_FINDINGS = TypeAdapter(list[KoreanFinding])
MEDICAL_CANDIDATES = TypeAdapter(list[MedicalCandidate])
MEDICAL_SUGGESTIONS = TypeAdapter(list[MedicalSuggestion])
