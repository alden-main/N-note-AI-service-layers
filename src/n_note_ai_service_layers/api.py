"""검토 완료 초안을 브리지 시스템으로 전달하기 위한 FastAPI 엔드포인트이다."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.form_definition_loader import FormDefinitionLoader
from n_note_ai_service_layers.adapters.note_artifact_store import FileNursingNoteStore
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.adapters.openai_stt import OpenAiSpeechToTextAdapter
from n_note_ai_service_layers.adapters.pdf_renderer import NursingNotePdfRenderer, PdfConfigurationError
from n_note_ai_service_layers.adapters.transcript_only_repository import TranscriptOnlyRepository
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft_review import DraftCandidate, FinalizedNursingNote
from n_note_ai_service_layers.contracts.report_validation import ReportValidationResult
from n_note_ai_service_layers.settings import load_dotenv
from n_note_ai_service_layers.adapters.synthetic_repository import SyntheticScenarioRepository
from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery
from n_note_ai_service_layers.policies.pressure_injury_schema import pressure_injury_poc_schema
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.services.access_policy import PurposeAccessPolicy
from n_note_ai_service_layers.services.context_assembler import ClinicalContextAssembler
from n_note_ai_service_layers.services.context_flow import ClinicalContextFlow
from n_note_ai_service_layers.services.draft_document import ordered_fields, ordered_sections
from n_note_ai_service_layers.services.draft_generation_flow import DraftGenerationFlow
from n_note_ai_service_layers.services.gateway_term_verification import GatewayTermVerificationFlow
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from n_note_ai_service_layers.services.report_validation import FormReportValidationFlow
from n_note_ai_service_layers.services.schema_review import FormSchemaRegistry, SchemaReviewService
from n_note_ai_service_layers.services.draft_review import DraftAlreadyFinalized, InMemoryDraftReviewStore, NumericReviewRequired
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor


class BridgeHandoffRequest(BaseModel):
    """브리지 전송을 요청하는 검토 완료 초안의 API 입력 모델이다.

    입력: 스키마 식별자·버전, 전송 필드, 검증 결과 및 간호사 검토 상태이다.
    출력: 전송 대기 항목을 만들 때 ``HandoffEnvelope``의 기반 데이터가 된다.
    """
    schema_id: str
    schema_version: str
    fields: dict[str, Any]
    validation: dict[str, Any]
    nurse_review_status: str


class BridgeAcknowledgement(BaseModel):
    """브리지 수신 시스템이 전달 상태를 회신하는 API 입력 모델이다.

    입력: 처리 상태와 선택적인 브리지 참조·메시지이다.
    출력: 해당 핸드오프의 상태를 갱신하는 확인 객체이다.
    """
    status: str = Field(description="received, accepted, rejected, or failed")
    bridge_reference: str | None = None
    message: str | None = None


class HandoffEnvelope(BridgeHandoffRequest):
    """저장된 핸드오프의 식별자와 전송 상태를 포함하는 API 응답 모델이다.

    입력: 요청 모델의 필드와 서버가 부여한 식별자·생성 시각·확인 정보이다.
    출력: 생성, 조회, 확인 API가 반환하는 핸드오프 상태이다.
    """
    handoff_id: str
    delivery_status: str
    created_at: datetime
    acknowledgement: BridgeAcknowledgement | None = None


class TranscriptionResponse(BaseModel):
    """음성 파일을 전사한 결과다. 원본 음성 바이트는 응답에 포함하지 않는다."""

    text: str
    provider: str
    model: str
    language: str | None = None
    trace_id: str
    redacted: bool
    redaction_count: int


class TemplateFieldRequest(BaseModel):
    """운영자가 등록할 template 필드의 API 입력 모델이다."""

    field_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    label: str | None = None
    required: bool = False
    llm_allowed: bool = True
    fact_id: str | None = None
    conditional_required_when: str | None = None
    display_order: int | None = Field(default=None, ge=0)


class TemplateSectionRequest(BaseModel):
    """운영자가 등록할 template 섹션의 API 입력 모델이다."""

    section_id: str = Field(min_length=1)
    label: str | None = None
    display_order: int | None = Field(default=None, ge=0)
    fields: list[TemplateFieldRequest] = Field(min_length=1)


class TemplateRegistrationRequest(BaseModel):
    """새 양식 template 버전을 초안 상태로 등록하는 API 입력 모델이다."""

    schema_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    sections: list[TemplateSectionRequest] = Field(min_length=1)


class TemplateActionRequest(BaseModel):
    """template 검토·승인 API에 필요한 작업자와 시각 입력 모델이다."""

    operator_id: str = Field(min_length=1)
    occurred_at: datetime | None = None


class DraftRevisionRequest(BaseModel):
    """후보 결정과 직접 편집을 다음 초안 revision으로 남기는 API 입력 모델이다.

    ``selected_suggestions``는 승인한 초안 후보에 적용할 제안어이며, 생략하면 첫 제안어를 적용한다.
    """

    edited_by: str = Field(min_length=1)
    edited_fields: dict[str, Any] = Field(default_factory=dict)
    candidate_decisions: dict[str, Literal["pending", "accepted", "rejected"]] = Field(default_factory=dict)
    selected_suggestions: dict[str, str] = Field(default_factory=dict)


class DraftFinalizeRequest(BaseModel):
    """최신 초안 revision을 확정하고 PDF를 생성하는 API 입력 모델이다."""

    finalized_by: str = Field(min_length=1)
    # 전사문에 없는 초안 수치(numeric_verification.findings)를 확인했다는 표시다. 남아 있으면 필수다.
    numeric_review_confirmed: bool = False


class InMemoryHandoffStore:
    """개발 환경에서 핸드오프를 메모리에 보관하는 임시 전송 저장소이다.

    입력: ``BridgeHandoffRequest``이다.
    출력: UUID와 대기 상태가 추가된 ``HandoffEnvelope``이며, 운영에서는 영속 아웃박스로 교체해야 한다.
    """
    def __init__(self) -> None:
        self.items: dict[str, HandoffEnvelope] = {}

    def create(self, handoff: BridgeHandoffRequest) -> HandoffEnvelope:
        """전송 요청에 식별자와 생성 시각을 부여하여 대기 상태로 저장한다.

        입력: ``handoff``는 검토 완료된 브리지 전송 요청이다.
        출력: ``pending`` 상태로 저장된 ``HandoffEnvelope``이다.
        """
        envelope = HandoffEnvelope(
            **handoff.model_dump(),
            handoff_id=str(uuid4()),
            delivery_status="pending",
            created_at=datetime.now(timezone.utc),
        )
        self.items[envelope.handoff_id] = envelope
        return envelope


load_dotenv()

store = InMemoryHandoffStore()
template_registry = FormSchemaRegistry()
template_registry.add(pressure_injury_poc_schema(status=SchemaStatus.ACTIVE))
# 운영자가 배포한 양식 정의(N_NOTE_FORMS_DIR)를 활성 양식으로 등록한다. 잘못된 정의는 기동을 중단한다.
for _form in FormDefinitionLoader(settings.forms_dir()).load_all(settings.enabled_form_ids()):
    template_registry.add(_form)
template_review_service = SchemaReviewService(template_registry)
draft_review_store = InMemoryDraftReviewStore()
app = FastAPI(title="N-note Bridge API", version="0.1.0")


def get_speech_to_text_provider() -> OpenAiSpeechToTextAdapter:
    """설정한 STT 서버(OpenAI 또는 로컬 Whisper)로 STT provider를 만든다."""
    try:
        return model_providers.speech_to_text_provider()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="speech transcription is not configured") from exc


def get_note_store() -> FileNursingNoteStore:
    """확정 간호기록 JSON·PDF 저장소를 만든다."""
    return FileNursingNoteStore(settings.artifact_dir())


def get_pdf_renderer() -> NursingNotePdfRenderer:
    """설정한 한글 글꼴로 확정본 PDF renderer를 만든다."""
    return NursingNotePdfRenderer(settings.pdf_font_path())


def get_transcript_redactor() -> TranscriptRedactor:
    """외부 처리 경계에 전달할 전사문을 비식별화하는 서비스를 만든다."""
    return TranscriptRedactor()


def get_workflow_context_flow() -> ClinicalContextFlow:
    """임상 문맥 흐름을 만든다.

    합성 POC fixture(N_NOTE_SYNTHETIC_FIXTURE)가 설정되면 그 사실을 사용하고, 없으면 전사문만으로
    초안을 만든다. 운영 EMR repository는 별도 adapter로 주입한다.
    """
    fixture = os.getenv("N_NOTE_SYNTHETIC_FIXTURE")
    if fixture and not Path(fixture).is_file():
        raise HTTPException(status_code=503, detail="nursing note workflow context is not configured")
    repository = SyntheticScenarioRepository(Path(fixture)) if fixture else TranscriptOnlyRepository()
    purposes = {schema.purpose for schema in template_registry.list_versions() if schema.status is SchemaStatus.ACTIVE}
    return ClinicalContextFlow(PurposeAccessPolicy(purposes), template_registry, repository, ClinicalContextAssembler())


def get_nursing_note_workflow(
    stt: OpenAiSpeechToTextAdapter = Depends(get_speech_to_text_provider),
) -> NursingNoteWorkflow:
    """Gateway를 유일한 외부 MCP 경계로 사용해 초안과 검토 artifact를 조립한다."""
    gateway = McpGatewayClient()
    try:
        generator = model_providers.draft_generator(gateway)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="draft generation is not configured") from exc
    draft_flow = DraftGenerationFlow(stt, generator)
    return NursingNoteWorkflow(draft_flow, GatewayTermVerificationFlow(gateway), report_validation=FormReportValidationFlow(gateway))


@app.get("/healthz", tags=["system"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _template_response(schema: FormSchema) -> dict[str, Any]:
    """내부 불변 template 계약을 API 응답 형식으로 변환한다."""
    return {
        "schema_id": schema.schema_id,
        "version": schema.version,
        "purpose": schema.purpose,
        "title": schema.display_title,
        "status": schema.status.value,
        "approved_by": schema.approved_by,
        "approved_at": schema.approved_at,
        "pdf_template": Path(schema.pdf_template).name if schema.pdf_template else None,
        "is_default": schema.schema_id == settings.default_form_id(),
        "sections": [
            {
                "section_id": section.section_id,
                "label": section.label,
                "display_order": section.display_order,
                "fields": [
                    {
                        "field_id": field.field_id,
                        "type": field.type,
                        "label": field.label,
                        "required": field.required,
                        "llm_allowed": field.llm_allowed,
                        "fact_id": field.fact_id,
                        "conditional_required_when": field.conditional_required_when,
                        "display_order": field.display_order,
                        "description": field.description,
                        "options": list(field.options),
                    }
                    for field in section.fields
                ],
            }
            for section in schema.sections
        ],
    }


def _request_schema(request: TemplateRegistrationRequest) -> FormSchema:
    """template 등록 요청을 항상 draft 상태의 내부 계약으로 변환한다."""
    return FormSchema(
        request.schema_id,
        request.version,
        request.purpose,
        tuple(
            FormSection(
                section.section_id,
                tuple(
                    FormField(
                        field.field_id,
                        field.type,
                        field.required,
                        field.llm_allowed,
                        field.fact_id,
                        field.conditional_required_when,
                        field.label,
                        field.display_order,
                    )
                    for field in section.fields
                ),
                section.label,
                section.display_order,
            )
            for section in request.sections
        ),
        status=SchemaStatus.DRAFT,
    )


@app.post("/v1/form-templates", status_code=status.HTTP_201_CREATED, tags=["form-template"])
def register_form_template(request: TemplateRegistrationRequest) -> dict[str, Any]:
    """운영자가 승인 전 양식 template 버전을 등록한다."""
    try:
        schema = template_registry.add(_request_schema(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _template_response(schema)


@app.get("/v1/form-templates", tags=["form-template"])
def list_form_templates(schema_id: str | None = None) -> list[dict[str, Any]]:
    """등록된 양식 template 버전을 반환한다."""
    return [_template_response(schema) for schema in template_registry.list_versions(schema_id)]


@app.get("/v1/form-templates/{schema_id}/{version:path}", tags=["form-template"])
def get_form_template(schema_id: str, version: str) -> dict[str, Any]:
    """특정 양식 template 버전을 반환한다."""
    try:
        return _template_response(template_registry.get(schema_id, version))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="form template not found") from exc


@app.post("/v1/form-templates/{schema_id}/{version:path}/approve", tags=["form-template"])
def approve_form_template(schema_id: str, version: str, request: TemplateActionRequest) -> dict[str, Any]:
    """구조 검증을 통과한 template 버전을 운영자 승인 상태로 전환한다."""
    try:
        schema = template_registry.get(schema_id, version)
        approved = template_review_service.approve_all(schema, request.operator_id, request.occurred_at or datetime.now(timezone.utc))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _template_response(approved)


@app.post("/v1/form-templates/{schema_id}/{version:path}/activate", tags=["form-template"])
def activate_form_template(schema_id: str, version: str, request: TemplateActionRequest) -> dict[str, Any]:
    """승인된 template 버전을 양식별 유일한 active 상태로 전환한다."""
    try:
        active = template_registry.activate(schema_id, version)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _template_response(active)


@app.post("/v1/transcriptions", response_model=TranscriptionResponse, tags=["speech-to-text"])
async def transcribe_audio(
    audio: UploadFile = File(..., description="전사할 음성 파일"),
    language_hint: str | None = Form(default=None, max_length=16),
    x_trace_id: str | None = Header(default=None),
    provider: OpenAiSpeechToTextAdapter = Depends(get_speech_to_text_provider),
    redactor: TranscriptRedactor = Depends(get_transcript_redactor),
) -> TranscriptionResponse:
    """음성 파일을 전사하고 응답에는 명시적 식별자를 치환한 텍스트만 포함한다."""
    allowed_mime_types = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg", "audio/flac", "video/mp4"}
    if audio.content_type not in allowed_mime_types:
        raise HTTPException(status_code=415, detail="unsupported audio content type")
    content = await audio.read(25 * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=422, detail="audio file is empty")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio file exceeds 25 MiB limit")
    trace_id = x_trace_id or str(uuid4())
    try:
        transcript: Transcript = provider.transcribe(AudioInput(content, audio.filename or "audio", audio.content_type, trace_id, language_hint))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="speech transcription is unavailable") from exc
    redacted = redactor.redact(transcript.text)
    return TranscriptionResponse(text=redacted.text, provider=transcript.provider, model=transcript.model, language=transcript.language, trace_id=trace_id, redacted=True, redaction_count=redacted.replacement_count)


def _candidate_id(phase: str, tool_name: str, index: int) -> str:
    return f"{phase}:{tool_name}:{index}"


def _gateway_verification_response(result) -> dict[str, Any]:
    return {
        "phase": result.phase,
        "status": result.status,
        "tool_names": list(result.tool_names),
        "auto_apply": result.auto_apply,
        "occurrences": [
            {
                "candidate_id": _candidate_id(result.phase, item.tool_name, index),
                "tool_name": item.tool_name, "start_offset": item.start_offset, "end_offset": item.end_offset,
                "input_text": item.input_text, "normalized_text": item.normalized_text,
                "field_id": item.field_id, "category": item.category,
                "review_status": item.review_status,
                "suggestions": [{"text": suggestion.text, "score": suggestion.score, "reason": suggestion.reason} for suggestion in item.suggestions],
            }
            for index, item in enumerate(result.occurrences)
        ],
    }


def _candidate_ids(*results) -> tuple[str, ...]:
    """workflow 응답의 후보 ID와 revision 저장소의 후보 ID를 같은 규칙으로 만든다."""
    return tuple(
        _candidate_id(result.phase, item.tool_name, index)
        for result in results
        for index, item in enumerate(result.occurrences)
    )


def _draft_candidates(*results) -> tuple[DraftCandidate, ...]:
    """Gateway 후보를 사용자가 승인·적용할 수 있는 초안 검토 후보로 변환한다."""
    return tuple(
        DraftCandidate(
            _candidate_id(result.phase, item.tool_name, index), result.phase, item.tool_name,
            item.start_offset, item.end_offset, item.input_text,
            tuple(suggestion.text for suggestion in item.suggestions), item.field_id, item.category,
        )
        for result in results
        for index, item in enumerate(result.occurrences)
    )


def _numeric_verification_response(findings) -> dict[str, Any]:
    """출처에 없는 초안 수치 목록이다. 비어 있지 않으면 확정 시 수치 확인이 필요하다."""
    return {
        "status": "needs_review" if findings else "ok",
        "findings": [
            {"finding_id": item.finding_id, "field_id": item.field_id, "start_offset": item.start_offset,
             "end_offset": item.end_offset, "text": item.text, "reason": item.reason}
            for item in findings
        ],
    }


def _report_validation_response(result: ReportValidationResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "tool_name": result.tool_name,
        "review_required": result.review_required,
        "summary": result.summary,
        "criteria": [
            {
                "criterion_id": criterion.criterion_id,
                "description": criterion.description,
                "status": criterion.status,
                "warnings": list(criterion.warnings),
                "elements": [
                    {"element_id": element.element_id, "field_id": element.field_id, "status": element.status, "warnings": list(element.warnings)}
                    for element in criterion.elements
                ],
            }
            for criterion in result.criteria
        ],
    }


def _sections_response(schema: FormSchema, fields: dict[str, Any], evidence_refs: dict[str, Any] | None = None, candidates: tuple[DraftCandidate, ...] = ()) -> list[dict[str, Any]]:
    """평면 필드를 양식 섹션·표시 순서대로 묶고, 필드별 적용 가능한 후보 ID를 함께 반환한다."""
    return [
        {
            "section_id": section.section_id,
            "label": section.label,
            "display_order": section.display_order,
            "fields": [
                {
                    "field_id": field.field_id,
                    "label": field.label,
                    "type": field.type,
                    "required": field.required,
                    "display_order": field.display_order,
                    "value": fields.get(field.field_id),
                    "evidence_refs": list((evidence_refs or {}).get(field.field_id, ())),
                    "candidate_ids": [item.candidate_id for item in candidates if item.field_id == field.field_id and not item.stale],
                }
                for field in ordered_fields(section)
            ],
        }
        for section in ordered_sections(schema)
    ]


def _draft_review_response(record) -> dict[str, Any]:
    """초안 검토 레코드를 사용자 수정 API의 응답 형식으로 변환한다."""
    latest = record.revisions[-1]
    schema = template_registry.get(record.schema_id, record.schema_version)
    return {
        "draft_id": record.draft_id,
        "schema_id": record.schema_id,
        "schema_version": record.schema_version,
        "form_title": schema.display_title,
        "transcript": record.transcript,
        "candidate_ids": list(record.candidate_ids),
        "revision": latest.revision,
        "fields": latest.fields,
        "sections": _sections_response(schema, latest.fields, candidates=record.candidates),
        "candidates": [
            {
                "candidate_id": item.candidate_id, "phase": item.phase, "tool_name": item.tool_name,
                "field_id": item.field_id, "category": item.category,
                "start_offset": item.start_offset, "end_offset": item.end_offset,
                "input_text": item.input_text, "suggestions": list(item.suggestions),
                "decision": latest.candidate_decisions.get(item.candidate_id, "pending"),
                "applicable": bool(item.field_id) and not item.stale,
                "stale": item.stale,
            }
            for item in record.candidates
        ],
        "candidate_decisions": latest.candidate_decisions,
        "applied_suggestions": {key: value for revision in record.revisions for key, value in revision.applied_suggestions.items()},
        "edited_by": latest.edited_by,
        "created_at": latest.created_at,
        "finalized_note_id": record.finalized_note_id,
        "numeric_verification": _numeric_verification_response(draft_review_store.numeric_findings(record, schema)),
    }


def _resolve_workflow_form(schema_id: str | None, purpose: str | None) -> tuple[str, str]:
    """요청 양식이 없으면 기본 양식(N_NOTE_DEFAULT_FORM)을, 목적이 없으면 활성 양식의 목적을 사용한다."""
    schema_id = schema_id or settings.default_form_id()
    if not schema_id:
        raise HTTPException(status_code=422, detail="schema_id is required when N_NOTE_DEFAULT_FORM is not set")
    if purpose:
        return schema_id, purpose
    active = [item for item in template_registry.list_versions(schema_id) if item.status is SchemaStatus.ACTIVE]
    if len(active) != 1:
        raise HTTPException(status_code=422, detail="workflow schema or context is not available")
    return schema_id, active[0].purpose


@app.post("/v1/nursing-note-workflows", tags=["nursing-note"])
async def create_nursing_note_workflow(
    audio: UploadFile = File(..., description="간호기록 초안에 사용할 음성 파일"),
    patient_id: str = Form(..., min_length=1),
    encounter_id: str = Form(..., min_length=1),
    purpose: str | None = Form(default=None, min_length=1),
    schema_id: str | None = Form(default=None, min_length=1, description="양식 ID. 생략하면 N_NOTE_DEFAULT_FORM"),
    schema_version: str | None = Form(default=None, min_length=1),
    language_hint: str | None = Form(default=None, max_length=16),
    x_trace_id: str | None = Header(default=None),
    context_flow: ClinicalContextFlow = Depends(get_workflow_context_flow),
    workflow: NursingNoteWorkflow = Depends(get_nursing_note_workflow),
) -> dict[str, Any]:
    """음성을 양식 초안으로 만들고, 원문은 바꾸지 않은 채 Gateway 용어 후보와 양식 검증 결과를 함께 반환한다."""
    allowed_mime_types = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg", "audio/flac", "video/mp4"}
    if audio.content_type not in allowed_mime_types:
        raise HTTPException(status_code=415, detail="unsupported audio content type")
    content = await audio.read(25 * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=422, detail="audio file is empty")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio file exceeds 25 MiB limit")
    schema_id, purpose = _resolve_workflow_form(schema_id, purpose)
    trace_id = x_trace_id or str(uuid4())
    query = ClinicalContextQuery(patient_id, encounter_id, purpose, datetime.now(timezone.utc), schema_id, trace_id, schema_version)
    try:
        snapshot = context_flow.build(query)
    except (PermissionError, LookupError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="workflow schema or context is not available") from exc
    try:
        selected_schema = context_flow.registry.resolve_active(schema_id, purpose, schema_version)
        result = workflow.generate(snapshot, AudioInput(content, audio.filename or "audio", audio.content_type, trace_id, language_hint), selected_schema)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="nursing note workflow is unavailable") from exc
    review = draft_review_store.create(
        selected_schema,
        result.transcript.text,
        result.draft.proposed_fields,
        _candidate_ids(result.transcript_gateway_verification, result.draft_gateway_verification),
        trace_id,
        _draft_candidates(result.transcript_gateway_verification, result.draft_gateway_verification),
        patient_id,
        encounter_id,
        result.numeric_sources,
        result.reference_time,
    )
    return {
        "trace_id": trace_id, "auto_apply": False, "redacted": True, "redaction_count": result.redaction_count,
        "draft_id": review.draft_id,
        "revision": review.revisions[-1].revision,
        "form": {"schema_id": selected_schema.schema_id, "version": selected_schema.version, "title": selected_schema.display_title},
        "transcript": {"text": result.transcript.text, "provider": result.transcript.provider, "model": result.transcript.model, "language": result.transcript.language},
        "draft": {
            "proposed_fields": result.draft.proposed_fields,
            "evidence_refs": result.draft.evidence_refs,
            "needs_confirmation": result.draft.needs_confirmation,
            "model_version": result.draft.model_version,
            "sections": _sections_response(selected_schema, result.draft.proposed_fields, result.draft.evidence_refs, review.candidates),
        },
        "transcript_verification": _gateway_verification_response(result.transcript_gateway_verification),
        "draft_verification": _gateway_verification_response(result.draft_gateway_verification),
        "report_validation": _report_validation_response(result.report_validation),
        "numeric_verification": _numeric_verification_response(result.numeric_findings),
    }


@app.get("/v1/nursing-note-drafts/{draft_id}", tags=["nursing-note"])
def get_nursing_note_draft(draft_id: str) -> dict[str, Any]:
    """확정 전 초안의 최신 revision과 후보 결정을 반환한다."""
    try:
        return _draft_review_response(draft_review_store.get(draft_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note draft not found") from exc


@app.post("/v1/nursing-note-drafts/{draft_id}/revisions", tags=["nursing-note"])
def revise_nursing_note_draft(draft_id: str, request: DraftRevisionRequest, x_trace_id: str | None = Header(default=None)) -> dict[str, Any]:
    """사용자 후보 결정·제안어 적용과 직접 편집을 새 revision으로 기록한다."""
    try:
        current = draft_review_store.get(draft_id)
        schema = template_registry.get(current.schema_id, current.schema_version)
        record = draft_review_store.revise(
            draft_id,
            schema,
            request.edited_fields,
            request.candidate_decisions,
            request.edited_by,
            x_trace_id or str(uuid4()),
            request.selected_suggestions,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note draft not found") from exc
    except DraftAlreadyFinalized as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _draft_review_response(record)


def _note_response(note: FinalizedNursingNote, pdf_status: str) -> dict[str, Any]:
    return {**note.as_dict(), "pdf_status": pdf_status, "pdf_url": f"/v1/nursing-notes/{note.note_id}/pdf" if pdf_status == "ready" else None}


def _render_pdf(note: FinalizedNursingNote, note_store: FileNursingNoteStore, renderer: NursingNotePdfRenderer) -> Path:
    schema = template_registry.get(note.schema_id, note.schema_version)
    return note_store.save_pdf(note.note_id, renderer.render(schema, note))


@app.post("/v1/nursing-note-drafts/{draft_id}/finalize", status_code=status.HTTP_201_CREATED, tags=["nursing-note"])
def finalize_nursing_note_draft(
    draft_id: str,
    request: DraftFinalizeRequest,
    x_trace_id: str | None = Header(default=None),
    note_store: FileNursingNoteStore = Depends(get_note_store),
    renderer: NursingNotePdfRenderer = Depends(get_pdf_renderer),
) -> dict[str, Any]:
    """최신 revision을 불변 확정본으로 저장하고, 선택한 양식의 PDF를 생성한다."""
    try:
        current = draft_review_store.get(draft_id)
        schema = template_registry.get(current.schema_id, current.schema_version)
        note = draft_review_store.finalize(draft_id, schema, request.finalized_by, x_trace_id or str(uuid4()), request.numeric_review_confirmed)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note draft not found") from exc
    except DraftAlreadyFinalized as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NumericReviewRequired as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), **_numeric_verification_response(exc.findings)}) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    note_store.save(note)
    try:
        _render_pdf(note, note_store, renderer)
        pdf_status = "ready"
    except PdfConfigurationError:
        # 확정본은 보존하고, 글꼴·양식 설정을 고친 뒤 PDF 조회 시 다시 렌더링한다.
        pdf_status = "unavailable"
    return _note_response(note, pdf_status)


@app.get("/v1/nursing-notes/{note_id}", tags=["nursing-note"])
def get_nursing_note(note_id: str, note_store: FileNursingNoteStore = Depends(get_note_store)) -> dict[str, Any]:
    """확정 간호기록과 PDF 상태를 반환한다."""
    try:
        note = note_store.get(note_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note not found") from exc
    return _note_response(note, "ready" if note_store.pdf_path(note_id) else "unavailable")


@app.get("/v1/nursing-notes/{note_id}/pdf", tags=["nursing-note"], response_class=Response)
def get_nursing_note_pdf(
    note_id: str,
    note_store: FileNursingNoteStore = Depends(get_note_store),
    renderer: NursingNotePdfRenderer = Depends(get_pdf_renderer),
) -> Response:
    """확정본 PDF를 반환한다. 저장된 PDF가 없으면 지금 렌더링해 저장한다."""
    try:
        note = note_store.get(note_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="nursing note not found") from exc
    path = note_store.pdf_path(note_id)
    if path is None:
        try:
            path = _render_pdf(note, note_store, renderer)
        except (PdfConfigurationError, KeyError) as exc:
            raise HTTPException(status_code=503, detail="PDF rendering is not configured") from exc
    filename = f"{note.schema_id}-{note.note_id}.pdf"
    return Response(path.read_bytes(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/v1/bridge/handoffs", response_model=HandoffEnvelope, status_code=status.HTTP_202_ACCEPTED, tags=["bridge"])
def enqueue_handoff(handoff: BridgeHandoffRequest) -> HandoffEnvelope:
    """간호사 검토가 끝난 핸드오프를 브리지 조회 및 확인 대상으로 접수한다."""
    return store.create(handoff)


@app.get("/v1/bridge/handoffs", response_model=list[HandoffEnvelope], tags=["bridge"])
def list_handoffs(delivery_status: str = "pending") -> list[HandoffEnvelope]:
    return [item for item in store.items.values() if item.delivery_status == delivery_status]


@app.get("/v1/bridge/handoffs/{handoff_id}", response_model=HandoffEnvelope, tags=["bridge"])
def get_handoff(handoff_id: str) -> HandoffEnvelope:
    item = store.items.get(handoff_id)
    if item is None:
        raise HTTPException(status_code=404, detail="handoff not found")
    return item


@app.post("/v1/bridge/handoffs/{handoff_id}/acknowledgements", response_model=HandoffEnvelope, tags=["bridge"])
def acknowledge_handoff(handoff_id: str, acknowledgement: BridgeAcknowledgement) -> HandoffEnvelope:
    item = store.items.get(handoff_id)
    if item is None:
        raise HTTPException(status_code=404, detail="handoff not found")
    updated = item.model_copy(update={"delivery_status": acknowledgement.status, "acknowledgement": acknowledgement})
    store.items[handoff_id] = updated
    return updated


@app.delete("/v1/bridge/handoffs/{handoff_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["bridge"])
def delete_handoff(handoff_id: str) -> Response:
    if handoff_id not in store.items:
        raise HTTPException(status_code=404, detail="handoff not found")
    del store.items[handoff_id]
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def main() -> None:
    import uvicorn
    uvicorn.run("n_note_ai_service_layers.api:app", host="0.0.0.0", port=8000)
