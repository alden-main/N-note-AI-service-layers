"""음성, 초안, Gateway 용어 후보와 양식 검증을 한 응답으로 조립한다."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from n_note_ai_service_layers.contracts.ai_provider import DraftGenerationProvider, SpeechToTextProvider
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.contracts.numeric_verification import NumericFinding
from n_note_ai_service_layers.contracts.report_validation import ReportValidationResult
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.observability import get_logger, stage
from n_note_ai_service_layers.services.clock import now_local
from n_note_ai_service_layers.services.draft_document import DraftDocument
from n_note_ai_service_layers.services.gateway_term_verification import GatewayTermVerificationFlow
from n_note_ai_service_layers.services.llm_payload import LlmPayloadProjector
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayError
from n_note_ai_service_layers.services.numeric_verification import NumericVerifier
from n_note_ai_service_layers.services.report_validation import FormReportValidationFlow
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor

MAX_TERM_HINTS = 50
_logger = get_logger("workflow")


class WorkflowStageError(RuntimeError):
    """필수 단계(STT·초안 생성 등)가 실패했다. ``stage``로 실패 위치를 구분한다(본문은 담지 않는다)."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"nursing note workflow failed at stage: {stage}")
        self.stage = stage


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
    # LLM이 용어 힌트를 받아 초안에 바로 반영한 교정이다. 후보 승인을 거치지 않았으므로 따로 보여준다.
    llm_applied_term_hints: tuple[dict[str, str], ...] = ()


class NursingNoteWorkflow:
    """문맥 준비가 완료된 요청만 외부 STT/LLM/Gateway로 전달하는 조립 경계다.

    순서: STT → 비식별화 → 전사문 표준어·의학용어 검토 → 검토 힌트를 포함한 양식 초안 생성 →
    초안 수치 대조 → (초안 필드별 용어 검토 ∥ 보고서 검증 MCP의 양식 완결성 검증)이다.
    어떤 후보도 자동 적용하지 않는다. STT·초안 생성 실패는 ``WorkflowStageError``이고,
    Gateway 실패는 ``unavailable`` 상태로 응답에 남긴다.
    """

    def __init__(
        self,
        stt: SpeechToTextProvider,
        generator: DraftGenerationProvider,
        gateway_verification: GatewayTermVerificationFlow,
        redactor: TranscriptRedactor | None = None,
        report_validation: FormReportValidationFlow | None = None,
        projector: LlmPayloadProjector | None = None,
    ) -> None:
        self.stt = stt
        self.generator = generator
        self.gateway_verification = gateway_verification
        self.redactor = redactor or TranscriptRedactor()
        self.report_validation = report_validation
        self.projector = projector or LlmPayloadProjector()

    def generate(
        self, snapshot, audio: AudioInput, schema: FormSchema, known_identifiers: Iterable[str] = (),
        progress: Callable[[str], None] | None = None,
    ) -> NursingNoteWorkflowResult:
        """``progress``는 각 단계(stt, redaction, transcript_term_review, draft_generation, draft_review)를 시작할 때 호출된다."""
        trace_id = audio.trace_id
        notify = progress or (lambda _: None)
        notify("stt")
        with stage(_logger, trace_id, "stt", audio_bytes=len(audio.content)):
            try:
                transcript = self.stt.transcribe(audio)
            except Exception as exc:
                raise WorkflowStageError("stt") from exc
        notify("redaction")
        with stage(_logger, trace_id, "redaction") as metrics:
            try:
                redacted = self.redactor.redact(transcript.text, known_identifiers)
            except Exception as exc:
                # NER 실패 시 덜 가린 전사문을 외부로 보내지 않고 중단한다.
                raise WorkflowStageError("redaction") from exc
            metrics["replacements"] = redacted.replacement_count
        external_transcript = Transcript(redacted.text, transcript.provider, transcript.model, transcript.language)
        notify("transcript_term_review")
        catalog = self._discover(trace_id)
        with stage(_logger, trace_id, "transcript_term_review") as metrics:
            if catalog is None:
                transcript_gateway = GatewayTermVerificationResult("transcript", "unavailable")
            else:
                transcript_gateway = self.gateway_verification.verify("transcript", external_transcript.text, trace_id, catalog)
            metrics.update(status=transcript_gateway.status, candidates=len(transcript_gateway.occurrences))
        reference_time = now_local()
        hints = self.hints(transcript_gateway)
        notify("draft_generation")
        with stage(_logger, trace_id, "draft_generation", schema_id=schema.schema_id) as metrics:
            try:
                payload = self.projector.project(snapshot, external_transcript.text, schema, term_review_hints=hints, reference_time=reference_time)
                draft = self.generator.generate(payload, schema)
            except Exception as exc:
                raise WorkflowStageError("draft_generation") from exc
            metrics["fields"] = len(draft.proposed_fields)
        sources = self.numeric_sources(external_transcript.text, payload)
        numeric_findings = NumericVerifier().check(schema, draft.proposed_fields, sources, reference_time)
        document = DraftDocument.build(schema, draft.proposed_fields)
        notify("draft_review")
        with stage(_logger, trace_id, "draft_review") as metrics:
            draft_gateway, validation = self._review_draft(schema, document, trace_id, catalog)
            metrics.update(term_status=draft_gateway.status, report_status=validation.status, numeric_findings=len(numeric_findings))
        return NursingNoteWorkflowResult(
            external_transcript, draft, transcript_gateway, draft_gateway, redacted.replacement_count,
            report_validation=validation, numeric_findings=numeric_findings, numeric_sources=sources, reference_time=reference_time,
            llm_applied_term_hints=self.applied_hints(hints, external_transcript.text, document.text),
        )

    def _review_draft(
        self, schema: FormSchema, document: DraftDocument, trace_id: str, catalog: tuple[dict[str, Any], ...] | None,
    ) -> tuple[GatewayTermVerificationResult, ReportValidationResult]:
        """초안 용어 검토와 양식 완결성 검증은 서로 독립이라 동시에 호출한다."""
        if catalog is None:
            validation = ReportValidationResult("unavailable") if self.report_validation else ReportValidationResult("skipped")
            return GatewayTermVerificationResult("draft", "unavailable"), validation
        if self.report_validation is None:
            return self.gateway_verification.verify_document("draft", document, trace_id, catalog), ReportValidationResult("skipped")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="draft-review") as pool:
            terms = pool.submit(self.gateway_verification.verify_document, "draft", document, trace_id, catalog)
            report = pool.submit(self.report_validation.validate, schema, document, trace_id, catalog)
            return terms.result(), report.result()

    @staticmethod
    def numeric_sources(transcript: str, payload: dict[str, Any]) -> tuple[str, ...]:
        """수치 대조 출처다: LLM에 준 전사문과 사용 가능한 사실 값이다."""
        facts = tuple(str(item["value"]) for item in payload.get("facts", []) if item.get("value") is not None)
        return (transcript, *facts)

    def _discover(self, request_id: str) -> tuple[dict[str, Any], ...] | None:
        """workflow 한 번에 Gateway catalog를 한 번만 조회한다(클라이언트 TTL 캐시). 실패하면 ``None``이다."""
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

    @staticmethod
    def applied_hints(hints: tuple[dict[str, Any], ...], transcript: str, draft_text: str) -> tuple[dict[str, str], ...]:
        """LLM이 힌트의 제안어로 바꿔 쓴 것으로 보이는 교정을 찾는다.

        원래 표기가 초안에 없고, 전사문에 없던 제안어가 초안에 나타나면 반영한 것으로 본다.
        LLM의 자기 보고에 의존하지 않는 결정적 비교라 일부 재서술은 놓칠 수 있다.
        """
        applied: list[dict[str, str]] = []
        seen: set[str] = set()
        for hint in hints:
            source = hint.get("source_text") or ""
            if not source or source in seen or source in draft_text:
                continue
            seen.add(source)
            used = next((text for text in hint.get("suggestions", []) if text and text in draft_text and text not in transcript), None)
            if used:
                applied.append({"source_text": source, "applied_text": used, "category": str(hint.get("category") or "")})
        return tuple(applied)
