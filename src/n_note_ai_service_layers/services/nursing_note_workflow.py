"""음성, 초안, Gateway 용어 후보와 양식 검증을 한 응답으로 조립한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.contracts.numeric_verification import NumericFinding
from n_note_ai_service_layers.contracts.report_validation import ReportValidationResult
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.draft_document import DraftDocument
from n_note_ai_service_layers.services.draft_generation_flow import DraftGenerationFlow
from n_note_ai_service_layers.services.gateway_term_verification import GatewayTermVerificationFlow
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayError
from n_note_ai_service_layers.services.numeric_verification import NumericVerifier
from n_note_ai_service_layers.services.report_validation import FormReportValidationFlow
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor

MAX_TERM_HINTS = 50


@dataclass(frozen=True)
class NursingNoteWorkflowResult:
    transcript: Transcript
    draft: DraftResult
    transcript_gateway_verification: GatewayTermVerificationResult
    draft_gateway_verification: GatewayTermVerificationResult
    redaction_count: int = 0
    auto_apply: bool = False
    report_validation: ReportValidationResult = field(default_factory=lambda: ReportValidationResult("skipped"))
    # 출처(전사문·사실)에 없는 초안 수치다. 간호사가 확인하기 전에는 확정하지 않는다.
    numeric_findings: tuple[NumericFinding, ...] = ()
    numeric_sources: tuple[str, ...] = ()
    reference_time: datetime | None = None


class NursingNoteWorkflow:
    """문맥 준비가 완료된 요청만 외부 STT/LLM/Gateway로 전달하는 조립 경계다.

    순서: STT → 비식별화 → 전사문 표준어·의학용어 검토 → 검토 힌트를 포함한 양식 초안 생성 →
    초안 수치 대조 → 초안 필드별 용어 검토 → 보고서 검증 MCP의 양식 완결성 검증이다. 어떤 후보도 자동 적용하지 않는다.
    """

    def __init__(
        self,
        draft_flow: DraftGenerationFlow,
        gateway_verification: GatewayTermVerificationFlow,
        redactor: TranscriptRedactor | None = None,
        report_validation: FormReportValidationFlow | None = None,
    ) -> None:
        self.draft_flow = draft_flow
        self.gateway_verification = gateway_verification
        self.redactor = redactor or TranscriptRedactor()
        self.report_validation = report_validation

    def generate(self, snapshot, audio: AudioInput, schema: FormSchema) -> NursingNoteWorkflowResult:
        transcript = self.draft_flow.stt.transcribe(audio)
        redacted = self.redactor.redact(transcript.text)
        external_transcript = Transcript(redacted.text, transcript.provider, transcript.model, transcript.language)
        catalog = self._discover(audio.trace_id)
        if catalog is None:
            transcript_gateway = GatewayTermVerificationResult("transcript", "unavailable")
        else:
            transcript_gateway = self.gateway_verification.verify("transcript", external_transcript.text, audio.trace_id, catalog)
        reference_time = datetime.now().astimezone()
        payload = self.draft_flow.projector.project(snapshot, external_transcript.text, schema, term_review_hints=self.hints(transcript_gateway), reference_time=reference_time)
        draft = self.draft_flow.generator.generate(payload, schema)
        sources = self.numeric_sources(external_transcript.text, payload)
        numeric_findings = NumericVerifier().check(schema, draft.proposed_fields, sources, reference_time)
        document = DraftDocument.build(schema, draft.proposed_fields)
        if catalog is None:
            draft_gateway = GatewayTermVerificationResult("draft", "unavailable")
            validation = ReportValidationResult("unavailable") if self.report_validation else ReportValidationResult("skipped")
        else:
            draft_gateway = self.gateway_verification.verify_document("draft", document, audio.trace_id, catalog)
            validation = self.report_validation.validate(schema, document, audio.trace_id, catalog) if self.report_validation else ReportValidationResult("skipped")
        return NursingNoteWorkflowResult(
            external_transcript, draft, transcript_gateway, draft_gateway, redacted.replacement_count,
            report_validation=validation, numeric_findings=numeric_findings, numeric_sources=sources, reference_time=reference_time,
        )

    @staticmethod
    def numeric_sources(transcript: str, payload: dict[str, Any]) -> tuple[str, ...]:
        """수치 대조 출처다: LLM에 준 전사문과 사용 가능한 사실 값이다."""
        facts = tuple(str(item["value"]) for item in payload.get("facts", []) if item.get("value") is not None)
        return (transcript, *facts)

    def _discover(self, request_id: str) -> tuple[dict[str, Any], ...] | None:
        """workflow 한 번에 Gateway catalog를 한 번만 조회한다. 실패하면 ``None``이다."""
        try:
            return self.gateway_verification.gateway.discover(request_id)
        except McpGatewayError:
            return None

    @staticmethod
    def hints(result: GatewayTermVerificationResult) -> tuple[dict[str, Any], ...]:
        """전사문 후보를 LLM이 문맥에 맞을 때만 쓰는 힌트로 축약한다."""
        return tuple(
            {
                "source_text": item.input_text,
                "category": item.category,
                "suggestions": [suggestion.text for suggestion in item.suggestions[:3]],
            }
            for item in result.occurrences[:MAX_TERM_HINTS]
        )
