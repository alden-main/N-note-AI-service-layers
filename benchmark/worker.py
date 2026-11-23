"""One isolated model invocation; stdin contains only model inputs, never reference answers."""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime


class NoopGateway:
    def discover(self, request_id):
        return ()

class NoopReview:
    gateway = NoopGateway()
    def verify(self, phase, text, request_id=None, catalog=None):
        from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
        return GatewayTermVerificationResult(phase, 'ok')
    def verify_document(self, phase, document, request_id=None, catalog=None):
        return self.verify(phase, document.text, request_id, catalog)

class TimedGenerator:
    def __init__(self, inner): self.inner, self.seconds = inner, 0.0
    def generate(self, payload, schema):
        t = time.perf_counter()
        try: return self.inner.generate(payload, schema)
        finally: self.seconds += time.perf_counter()-t

def run(data):
    from n_note_ai_service_layers import settings
    from n_note_ai_service_layers.adapters.model_providers import draft_generator
    from n_note_ai_service_layers.api.deps import get_transcript_redactor, get_workflow_context_flow
    from n_note_ai_service_layers.api.routes.workflow_jobs import resolve_workflow_form
    from n_note_ai_service_layers.contracts.audio import PreTranscribedAudio, Transcript
    from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery
    from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
    from n_note_ai_service_layers.services.report_validation import FormReportValidationFlow
    if settings.get().llm_tool_calling_enabled:
        raise ValueError('Set N_NOTE_LLM_TOOL_CALLING=false for the fixed no-tools benchmark')
    req = data['request']; ref = datetime.fromisoformat(data['reference_time'])
    if ref.tzinfo is None: raise ValueError('reference time must include timezone')
    form, purpose = resolve_workflow_form(req['schema_id'], None)
    context = get_workflow_context_flow()
    schema = context.registry.resolve_active(form, purpose, None)
    trace = str(uuid.uuid4())
    snapshot = context.build(ClinicalContextQuery(req['patient_id'],req['encounter_id'],purpose,ref,form,trace,None))
    gen = TimedGenerator(draft_generator(None))
    flow = NursingNoteWorkflow(None, gen, NoopReview(), redactor=get_transcript_redactor(), report_validation=FormReportValidationFlow(None, None), hint_budget_seconds=None)
    source = PreTranscribedAudio(Transcript(data['transcript'], 'benchmark', 'fixture', req['language_hint']), trace)
    started = time.perf_counter()
    prepared = flow.preparation.prepare(source, (req['patient_id'], req['encounter_id']), lambda _: None)
    result = flow.draft(replace(prepared, reference_time=ref), snapshot, schema)
    return {'fields':result.draft.proposed_fields, 'model_version':result.draft.model_version,
            'prompt_version':result.draft.prompt_version, 'total_seconds':time.perf_counter()-started,
            'llm_seconds':gen.seconds, 'numeric_findings':len(result.numeric_findings)}

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv('.env')
        data = json.load(sys.stdin)
        if set(data) != {'request','transcript','reference_time'}: raise ValueError('unexpected worker input')
        print(json.dumps({'status':'ok', **run(data)}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'status':'error', 'error_type':type(exc).__name__}))
        return 1
    return 0
if __name__ == '__main__': sys.exit(main())
