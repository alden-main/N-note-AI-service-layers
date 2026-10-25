"""임상 사실 조회·정규화·스냅샷 생성을 위한 서비스 간 계약이다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol


class Availability(StrEnum):
    """임상 사실을 초안 생성에 사용해도 되는지 나타내는 상태값이다.

    입력: 사실의 조회·정규화·신선도 판정 결과이다.
    출력: 사용 가능, 불가, 오래됨 또는 충돌 상태 문자열이다.
    """
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ClinicalContextQuery:
    """환자 임상 문맥 조회 요청을 표현한다.

    입력: 환자·내원 식별자, 사용 목적, 기준 시각, 스키마와 추적 정보이다.
    출력: 접근 제어와 저장소 조회에 전달되는 불변 요청 객체이다.
    """
    patient_id: str
    encounter_id: str
    purpose: str
    as_of: datetime
    schema_id: str
    trace_id: str
    schema_version: str | None = None


@dataclass(frozen=True)
class SourceRecord:
    """외부 원천 시스템에서 읽은 정규화 전 임상 관찰값이다.

    입력: 사실 ID, 원천 값, 관찰 시각, 출처와 선택 참조값이다.
    출력: 정규화기가 ``ClinicalFact``로 변환할 불변 레코드이다.
    """
    fact_id: str
    value: Any
    observed_at: datetime
    source: str
    source_reference: str | None = None


@dataclass(frozen=True)
class ClinicalFact:
    """정규화와 검증이 끝난 하나의 임상 사실을 표현한다.

    입력: 사실 ID, 값·표시값, 관찰 시각, 출처, 가용 상태 및 사유이다.
    출력: 임상 문맥 스냅샷과 LLM 페이로드에 포함되는 불변 사실이다.
    """
    fact_id: str
    value: Any
    display: str | None
    observed_at: datetime | None
    source: str | None
    availability: Availability
    reason: str | None = None


@dataclass(frozen=True)
class ClinicalContextSnapshot:
    """특정 활성 스키마 기준으로 확정한 임상 사실 집합이다.

    입력: 스키마 ID·버전, 임상 사실 튜플과 추적 ID이다.
    출력: 초안 생성과 결정 규칙에 전달되는 불변 스냅샷이다.
    """
    schema_id: str
    schema_version: str
    facts: tuple[ClinicalFact, ...]
    trace_id: str


class ClinicalDataRepository(Protocol):
    """환자 요청과 사실 ID로 원천 임상 레코드를 조회하는 인터페이스이다.

    입력: ``ClinicalContextQuery``와 필요한 사실 ID 집합이다.
    출력: 정규화 전 ``SourceRecord`` 목록이다.
    """
    def fetch(self, query: ClinicalContextQuery, requested_fact_ids: set[str]) -> list[SourceRecord]: ...


class ClinicalFactNormalizer(Protocol):
    """원천 임상 레코드를 표준 임상 사실로 정규화하는 인터페이스이다.

    입력: ``SourceRecord`` 목록이다.
    출력: 상태가 지정된 ``ClinicalFact`` 목록이다.
    """
    def normalize(self, records: list[SourceRecord]) -> list[ClinicalFact]: ...
