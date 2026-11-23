"""초안 생성 요청의 접근 검사 입력과, 검사를 통과한 요청 문맥이다.

EMR 데이터는 양식 필드에 매칭하지 않는다. bridge가 요청에 함께 보낸 EMR JSON 전체를 참고 문맥으로 쓴다
(``services.emr_context``). 여기에는 어떤 양식으로 누구의 기록을 만들지에 대한 문맥만 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ClinicalContextQuery:
    """초안 생성 요청의 접근 검사 입력이다.

    입력: 환자·내원 식별자, 사용 목적, 기준 시각, 스키마와 추적 정보이다.
    출력: 접근 정책과 양식 선택에 전달되는 불변 요청 객체이다.
    """
    patient_id: str
    encounter_id: str
    purpose: str
    as_of: datetime
    schema_id: str
    trace_id: str
    schema_version: str | None = None


@dataclass(frozen=True)
class AuthorizedContext:
    """접근 검사를 통과한 요청 문맥이다. LLM 입력은 이 문맥과 같은 활성 양식으로만 만든다(``LlmPayloadProjector``)."""

    schema_id: str
    schema_version: str
    trace_id: str
