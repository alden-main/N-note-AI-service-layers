"""간호 양식 스키마의 구조, 검토 이력 및 브리지 전달 계약을 정의한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from n_note_ai_service_layers.contracts.codec import IsoDatetime


class SchemaStatus(StrEnum):
    """양식 스키마가 검토와 운영 과정에서 가지는 상태값이다.

    입력: 스키마 작성·검토·승인·활성화 과정의 전이 결과이다.
    출력: 스키마 사용 가능 여부를 나타내는 상태 문자열이다.
    """
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class PdfPlacement:
    """PDF 양식 위에 필드 값을 그릴 위치를 정의한다.

    입력: AcroForm 필드명 또는 쪽 번호(1부터)·좌하단 기준 좌표·크기(pt)와 글자 크기이다.
    출력: PDF renderer가 값을 그릴 영역을 계산하는 불변 배치 정보이다.
    """
    field_name: str | None = None
    page: int | None = None
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    font_size: float = 10.0

    @property
    def has_box(self) -> bool:
        return None not in (self.page, self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class FormField:
    """양식의 개별 입력 필드와 AI 사용 범위를 정의한다.

    입력: 필드 ID·자료형, 필수 여부, AI 허용 여부, 연결 사실 ID와 작성 지침·선택지·PDF 배치이다.
    출력: 섹션과 스키마를 구성하는 불변 필드 정의이다.
    """
    field_id: str
    type: str
    required: bool = False
    llm_allowed: bool = True
    fact_id: str | None = None
    conditional_required_when: str | None = None
    label: str | None = None
    display_order: int | None = None
    description: str | None = None
    options: tuple[str, ...] = ()
    pdf: PdfPlacement | None = None


@dataclass(frozen=True)
class FormSection:
    """관련 양식 필드를 하나로 묶는 섹션을 정의한다.

    입력: 섹션 ID와 ``FormField`` 튜플이다.
    출력: ``FormSchema``에 포함되는 불변 섹션 정의이다.
    """
    section_id: str
    fields: tuple[FormField, ...]
    label: str | None = None
    display_order: int | None = None


@dataclass(frozen=True)
class BridgeMapping:
    """양식 필드를 외부 브리지 전송 키로 연결한다.

    입력: 스키마 필드 ID와 외부 출력 키이다.
    출력: 핸드오프 투영에 사용하는 불변 매핑 정의이다.
    """
    field_id: str
    output_key: str


@dataclass(frozen=True)
class FormSchema:
    """특정 업무 목적의 승인 가능한 간호 양식 전체를 정의한다.

    입력: 식별자·버전·목적, 섹션, 검증 규칙, 브리지 매핑 및 승인 메타데이터이다.
    출력: 사실 조회, LLM 투영, 검증과 핸드오프의 기준이 되는 불변 스키마이다.
    """
    schema_id: str
    version: str
    purpose: str
    sections: tuple[FormSection, ...]
    validation_rule_ids: tuple[str, ...] = ()
    bridge_mapping: tuple[BridgeMapping, ...] = ()
    status: SchemaStatus = SchemaStatus.DRAFT
    approved_by: str | None = None
    approved_at: IsoDatetime | None = None
    freshness_windows_minutes: dict[str, int] = field(default_factory=dict)
    title: str | None = None
    instructions: str | None = None
    pdf_template: str | None = None
    # 양식 PDF 위에 환자 ID·확정자 등 확정 메타데이터를 그릴 위치다.
    pdf_meta: tuple[tuple[str, PdfPlacement], ...] = ()
    # report-validation MCP의 report-validation-standard/v1 criteria에 추가할 양식별 기준이다.
    validation_criteria: tuple[dict[str, Any], ...] = ()

    @property
    def display_title(self) -> str:
        return self.title or self.schema_id

    @property
    def fields(self) -> tuple[FormField, ...]:
        return tuple(field for section in self.sections for field in section.fields)

    @property
    def requested_fact_ids(self) -> set[str]:
        return {field.fact_id for field in self.fields if field.fact_id}


@dataclass(frozen=True)
class SchemaCandidate:
    """원본 문서에서 추출되어 검토를 기다리는 스키마 후보를 표현한다.

    입력: 원본 문서 참조, 추출 시각·도구, 추출 섹션과 메모이다.
    출력: 초안 스키마 생성에 사용할 수 있는 불변 후보이며 승인·활성 상태는 허용하지 않는다.
    """
    source_document_ref: str
    extracted_at: datetime
    extractor_id: str
    sections: tuple[FormSection, ...] = ()
    notes: tuple[str, ...] = ()
    status: SchemaStatus = SchemaStatus.DRAFT

    def __post_init__(self) -> None:
        if self.status not in (SchemaStatus.DRAFT, SchemaStatus.REVIEW):
            raise ValueError("schema candidates cannot be approved or active")


@dataclass(frozen=True)
class ChangeLogEntry:
    """스키마 필드를 수정한 변경 이력 한 건을 표현한다.

    입력: 필드 ID, 변경 전후 값, 사유, 변경자와 변경 시각이다.
    출력: 검토 이력에 보관되는 불변 변경 기록이다.
    """
    field_id: str
    before: Any
    after: Any
    reason: str
    changed_by: str
    changed_at: datetime


@dataclass(frozen=True)
class SchemaReview:
    """스키마 버전에 대한 섹션 검토와 변경 이력을 관리한다.

    입력: 스키마 ID·버전, 검토 완료 섹션, 변경 이력 및 검토자 정보이다.
    출력: 검토 서비스가 갱신하는 불변 검토 상태이다.
    """
    schema_id: str
    version: str
    reviewed_sections: tuple[str, ...] = ()
    change_log: tuple[ChangeLogEntry, ...] = ()
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


@dataclass(frozen=True)
class DraftHandoff:
    """검증된 초안을 외부 브리지에 전달할 데이터로 표현한다.

    입력: 스키마 ID·버전, 전송 필드, 검증 결과와 간호사 검토 상태이다.
    출력: 브리지 API 요청으로 직렬화할 불변 핸드오프 객체이다.
    """
    schema_id: str
    schema_version: str
    fields: dict[str, Any]
    validation: dict[str, Any]
    nurse_review_status: str
