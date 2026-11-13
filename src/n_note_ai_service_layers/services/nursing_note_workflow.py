"""음성, 초안, Gateway 용어 후보와 양식 검증을 한 응답으로 조립한다."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from n_note_ai_service_layers.contracts.ai_provider import DraftGenerationProvider, SpeechToTextProvider
from n_note_ai_service_layers.contracts.audio import AudioDecodeError, AudioInput, PreTranscribedAudio, Transcript
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


@dataclass(frozen=True)
class PreparedTranscript:
    """여러 양식이 함께 쓰는 공통 단계(STT·비식별화·전사문 용어 검토)의 결과다."""

    trace_id: str
    # 비식별화한 전사문이다. 외부 LLM·Gateway에는 이것만 보낸다.
    transcript: Transcript
    redaction_count: int
    catalog: tuple[dict[str, Any], ...] | None
    # 힌트 대기 상한 안에 끝난 전사문 검토로 만든 LLM 힌트다. 늦으면 비어 있다.
    hints: tuple[dict[str, Any], ...]
    reference_time: datetime
    review: Future[GatewayTermVerificationResult]
    review_pool: ThreadPoolExecutor

    def close(self) -> None:
        """검토 스레드 풀을 정리한다. 진행 중인 검토는 끝까지 실행되므로 ``review.result()``는 계속 쓸 수 있다."""
        self.review_pool.shutdown(wait=False)


class NursingNoteWorkflow:
    """문맥 준비가 완료된 요청만 외부 STT/LLM/Gateway로 전달하는 조립 경계다.

    순서: STT → 비식별화 → 전사문 표준어·의학용어 검토 → 검토 힌트를 포함한 양식 초안 생성 →
    초안 수치 대조 → (초안 필드별 용어 검토 ∥ 보고서 검증 MCP의 양식 완결성 검증)이다.
    앞의 세 단계(``prepare``)는 양식과 무관하므로, 여러 양식을 만들 때(``generate_many``)는 한 번만 하고
    양식별 단계(``draft``)만 병렬로 실행한다.
    전사문 검토가 ``hint_budget_seconds`` 안에 끝나지 않으면 힌트 없이 초안 생성을 시작하고, 검토 결과는
    초안 생성 뒤에 받아 응답(검토 후보)에 담는다. Gateway 지연이 LLM 시작을 붙잡지 않게 하기 위해서다.
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
        hint_budget_seconds: float | None = None,
    ) -> None:
        self.stt = stt
        self.generator = generator
        self.gateway_verification = gateway_verification
        self.redactor = redactor or TranscriptRedactor()
        self.report_validation = report_validation
        self.projector = projector or LlmPayloadProjector()
        # None이면 전사문 검토가 끝날 때까지 기다린다.
        self.hint_budget_seconds = hint_budget_seconds

    def generate(
        self, snapshot, audio: AudioInput | PreTranscribedAudio, schema: FormSchema, known_identifiers: Iterable[str] = (),
        progress: Callable[[str], None] | None = None,
    ) -> NursingNoteWorkflowResult:
        """``progress``는 각 단계(stt, redaction, transcript_term_review, draft_generation, draft_review)를 시작할 때 호출된다."""
        notify = progress or (lambda _: None)
        prepared = self.prepare(audio, known_identifiers, notify)
        try:
            return self.draft(prepared, snapshot, schema, notify)
        finally:
            prepared.close()

    def generate_many(
        self,
        audio: AudioInput | PreTranscribedAudio,
        forms: Sequence[tuple[FormSchema, Any]],
        known_identifiers: Iterable[str] = (),
        progress: Callable[[str], None] | None = None,
        on_form_stage: Callable[[int, str], None] | None = None,
        on_form_done: Callable[[int, NursingNoteWorkflowResult | Exception], None] | None = None,
        max_parallel: int = 1,
    ) -> list[NursingNoteWorkflowResult | Exception]:
        """음성 하나로 여러 양식의 초안을 만든다.

        STT·비식별화·전사문 용어 검토는 한 번만 하고, 양식별 초안 생성·수치 대조·초안 검토는
        ``max_parallel``개까지 동시에 실행한다. ``forms``는 (양식, 그 양식의 임상 문맥) 목록이다.

        공통 단계가 실패하면 ``WorkflowStageError``를 던진다. 양식 하나의 실패는 다른 양식에 영향을 주지 않고
        결과 목록의 그 자리에 예외로 남는다(초안 생성 실패는 ``WorkflowStageError``).
        ``on_form_stage(index, stage)``는 양식별 단계가 바뀔 때, ``on_form_done(index, outcome)``은 양식 하나가
        끝나는 즉시 작업 스레드에서 호출된다. 먼저 끝난 초안을 바로 저장·표시하기 위해서다.
        """
        if not forms:
            raise ValueError("at least one form is required")
        notify = progress or (lambda _: None)
        form_stage = on_form_stage or (lambda _index, _stage: None)
        form_done = on_form_done or (lambda _index, _outcome: None)
        prepared = self.prepare(audio, known_identifiers, notify)
        notify("draft_generation")

        def run(index: int, schema: FormSchema, snapshot: Any) -> NursingNoteWorkflowResult | Exception:
            outcome: NursingNoteWorkflowResult | Exception
            try:
                outcome = self.draft(prepared, snapshot, schema, lambda name: form_stage(index, name))
            except Exception as exc:  # 한 양식의 실패가 다른 양식의 결과를 버리지 않게 결과로 돌려준다.
                outcome = exc
            form_done(index, outcome)
            return outcome

        try:
            workers = max(1, min(len(forms), max_parallel))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="form-draft") as pool:
                futures = [pool.submit(run, index, schema, snapshot) for index, (schema, snapshot) in enumerate(forms)]
                return [future.result() for future in futures]
        finally:
            prepared.close()

    def prepare(
        self, audio: AudioInput | PreTranscribedAudio, known_identifiers: Iterable[str], notify: Callable[[str], None],
    ) -> PreparedTranscript:
        """양식과 무관한 공통 단계다: STT → 비식별화 → 전사문 용어 검토 시작(힌트 대기 상한까지 기다림).

        ``PreTranscribedAudio``(녹음 세션에서 이미 전사한 음성)면 STT를 다시 하지 않고 비식별화부터 시작한다.
        반환값의 ``close()``를 호출해 검토 스레드를 정리한다. 검토가 아직 진행 중이면 결과는 ``review``로 받는다.
        """
        trace_id = audio.trace_id
        if isinstance(audio, PreTranscribedAudio):
            transcript = audio.transcript
        else:
            notify("stt")
            with stage(_logger, trace_id, "stt", audio_bytes=len(audio.content)):
                try:
                    transcript = self.stt.transcribe(audio)
                except AudioDecodeError as exc:
                    raise WorkflowStageError("audio_decode") from exc
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
        review_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcript-review")
        try:
            review = self._start_transcript_review(review_pool, external_transcript.text, trace_id, catalog)
            with stage(_logger, trace_id, "transcript_term_review") as metrics:
                done, _ = wait([review], timeout=self.hint_budget_seconds)
                hint_source = review.result() if done else None
                metrics.update(
                    status=hint_source.status if hint_source else "pending",
                    candidates=len(hint_source.occurrences) if hint_source else 0,
                    hints_deferred=not done,
                )
        except BaseException:
            review_pool.shutdown(wait=False)
            raise
        return PreparedTranscript(
            trace_id, external_transcript, redacted.replacement_count, catalog,
            self.hints(hint_source) if hint_source else (), now_local(), review, review_pool,
        )

    def draft(
        self, prepared: PreparedTranscript, snapshot, schema: FormSchema, notify: Callable[[str], None] | None = None,
    ) -> NursingNoteWorkflowResult:
        """양식 하나의 단계다: 검토 힌트를 포함한 초안 생성 → 수치 대조 → (초안 용어 검토 ∥ 양식 완결성 검증)."""
        notify = notify or (lambda _: None)
        trace_id, transcript, hints = prepared.trace_id, prepared.transcript, prepared.hints
        notify("draft_generation")
        with stage(_logger, trace_id, "draft_generation", schema_id=schema.schema_id) as metrics:
            try:
                payload = self.projector.project(snapshot, transcript.text, schema, term_review_hints=hints, reference_time=prepared.reference_time)
                draft = self.generator.generate(payload, schema)
            except Exception as exc:
                raise WorkflowStageError("draft_generation") from exc
            metrics.update(fields=len(draft.proposed_fields), hints=len(hints))
        # 힌트 대기 상한을 넘긴 검토도 화면의 검토 후보로는 필요하다. Gateway timeout 안에 끝난다.
        transcript_gateway = prepared.review.result()
        sources = self.numeric_sources(transcript.text, payload)
        numeric_findings = NumericVerifier().check(schema, draft.proposed_fields, sources, prepared.reference_time)
        document = DraftDocument.build(schema, draft.proposed_fields)
        notify("draft_review")
        with stage(_logger, trace_id, "draft_review", schema_id=schema.schema_id) as metrics:
            draft_gateway, validation = self._review_draft(schema, document, trace_id, prepared.catalog)
            metrics.update(term_status=draft_gateway.status, report_status=validation.status, numeric_findings=len(numeric_findings))
        return NursingNoteWorkflowResult(
            transcript, draft, transcript_gateway, draft_gateway, prepared.redaction_count,
            report_validation=validation, numeric_findings=numeric_findings, numeric_sources=sources, reference_time=prepared.reference_time,
            llm_applied_term_hints=self.applied_hints(hints, transcript.text, document.text),
        )

    def _start_transcript_review(
        self, pool: ThreadPoolExecutor, text: str, trace_id: str, catalog: tuple[dict[str, Any], ...] | None,
    ) -> Future[GatewayTermVerificationResult]:
        if catalog is None:
            done: Future[GatewayTermVerificationResult] = Future()
            done.set_result(GatewayTermVerificationResult("transcript", "unavailable"))
            return done
        return pool.submit(self.gateway_verification.verify, "transcript", text, trace_id, catalog)

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
