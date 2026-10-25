"""양식 원본에서 검토용 스키마 후보를 추출하기 위한 계약과 테스트 구현이다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from n_note_ai_service_layers.contracts.form_schema import SchemaCandidate


@dataclass(frozen=True)
class SchemaSourceDocument:
    """스키마 후보 추출의 입력이 되는 원본 문서 참조를 표현한다.

    입력: 문서 식별자 ``reference``와 선택 본문 ``text``이다.
    출력: 추출 어댑터에 전달할 불변 문서 객체이다.
    """
    reference: str
    text: str | None = None


class SchemaExtractionAdapter(Protocol):
    """원본 문서를 실행 불가 상태의 스키마 후보로 추출하는 인터페이스이다.

    입력: ``SchemaSourceDocument``이다.
    출력: 검토 전 ``SchemaCandidate``이다.
    """
    def extract(self, document: SchemaSourceDocument) -> SchemaCandidate: ...


class TestSchemaExtractor:
    """결정적인 스키마 추출 결과를 제공하는 테스트용 어댑터이다.

    입력: 미리 지정한 후보 또는 발생시킬 예외이다.
    출력: 후보를 반환하거나 지정된 예외를 발생시키며, 실행 가능한 스키마는 만들지 않는다.
    """
    def __init__(self, candidate: SchemaCandidate | None = None, error: Exception | None = None) -> None:
        self.candidate, self.error = candidate, error

    def extract(self, document: SchemaSourceDocument) -> SchemaCandidate:
        if self.error:
            raise self.error
        if self.candidate:
            return self.candidate
        return SchemaCandidate(document.reference, datetime.now().astimezone(), "test-extractor", notes=("incomplete extraction",))
