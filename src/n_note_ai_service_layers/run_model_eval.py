"""같은 평가 사례로 STT·LLM 모델 프로필을 비교하는 실행기이다.

사례 디렉터리(``eval/cases/<case_id>/``)마다 ``case.json``과 ``transcript.txt`` 또는 음성 파일
(``audio.wav`` 등)을 둔다. 전사문만 있으면 STT를 건너뛰어 LLM만 비교하고, 음성과
``reference_transcript.txt``가 있으면 STT 문자 오류율(CER)도 잰다. 결과는 사례별 JSONL과 요약 JSON으로
``eval/results/<profile>/<시각>/``에 저장하며(전사문에 없는 수치 검출 포함), ``compare``로 프로필별 요약을 표로 비교한다.

실제 환자 음성·기록은 평가 사례에 넣지 않는다(합성 또는 비식별 자료만 사용).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.contracts.audio import AudioInput, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormSchema
from n_note_ai_service_layers.services.draft_document import format_value, is_filled

AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".webm": "audio/webm", ".ogg": "audio/ogg", ".flac": "audio/flac"}


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    schema_id: str | None
    language_hint: str | None
    transcript: str | None
    audio_path: Path | None
    reference_transcript: str | None
    expected: dict[str, Any]


class FixedTranscriptProvider:
    """전사문 사례에서 STT를 건너뛰고 주어진 전사문을 돌려준다."""

    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: AudioInput) -> Transcript:
        return Transcript(self._text, "fixture", "reference-transcript", "ko")


class _Timed:
    """감싼 provider 호출 시간을 잰다(``transcribe``·``generate``)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.seconds = 0.0

    def transcribe(self, audio: AudioInput) -> Transcript:
        started = time.perf_counter()
        try:
            return self.inner.transcribe(audio)
        finally:
            self.seconds += time.perf_counter() - started

    def generate(self, payload: dict, schema: FormSchema) -> DraftResult:
        started = time.perf_counter()
        try:
            return self.inner.generate(payload, schema)
        finally:
            self.seconds += time.perf_counter() - started


def load_cases(root: Path) -> list[EvalCase]:
    cases = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_path = directory / "case.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        transcript_path = directory / "transcript.txt"
        reference_path = directory / "reference_transcript.txt"
        audio = next((path for path in sorted(directory.iterdir()) if path.stem == "audio" and path.suffix.lower() in AUDIO_TYPES), None)
        if audio is None and not transcript_path.is_file():
            continue
        cases.append(EvalCase(
            case_id=directory.name,
            schema_id=meta.get("schema_id"),
            language_hint=meta.get("language_hint", "ko"),
            transcript=transcript_path.read_text().strip() if transcript_path.is_file() else None,
            audio_path=audio,
            reference_transcript=reference_path.read_text().strip() if reference_path.is_file() else None,
            expected=meta.get("expected", {}),
        ))
    return cases


def character_error_rate(reference: str, hypothesis: str) -> float:
    """공백을 뺀 글자 단위 편집 거리 / 참조 글자 수다."""
    ref = "".join(reference.split())
    hyp = "".join(hypothesis.split())
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, 1):
        current = [i]
        for j, hyp_char in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1] / len(ref)


def score_draft(schema: FormSchema, fields: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """필수 항목 작성률, 기대 핵심어 포함률, 금지어(지어낸 사실) 검출을 계산한다.

    ``expected.keywords``는 ``{field_id: [핵심어, ...]}``, ``expected.forbidden``은 초안 어디에도
    나오면 안 되는 문자열 목록이다(예: 구술에 없는 날짜).
    """
    required = [field.field_id for field in schema.fields if field.required and field.llm_allowed]
    filled_required = [field_id for field_id in required if is_filled(fields.get(field_id))]
    keywords = expected.get("keywords", {})
    total = sum(len(items) for items in keywords.values())
    missing = {
        field_id: [word for word in words if word not in format_value(fields.get(field_id))]
        for field_id, words in keywords.items()
    }
    missing = {key: value for key, value in missing.items() if value}
    found = total - sum(len(items) for items in missing.values())
    whole = "\n".join(format_value(value) for value in fields.values())
    forbidden = [word for word in expected.get("forbidden", []) if word in whole]
    return {
        "required_filled": len(filled_required),
        "required_total": len(required),
        "keyword_recall": round(found / total, 3) if total else None,
        "missing_keywords": missing,
        "forbidden_found": forbidden,
    }


def run_case(case: EvalCase, stt_provider: Any | None = None) -> dict[str, Any]:
    from n_note_ai_service_layers import api
    from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery

    uses_audio = case.audio_path is not None
    if uses_audio:
        stt = _Timed(stt_provider or api.get_speech_to_text_provider())
    else:
        stt = _Timed(FixedTranscriptProvider(case.transcript or ""))
    workflow = api.get_nursing_note_workflow(stt)
    generator = _Timed(workflow.generator)
    workflow.generator = generator
    schema_id, purpose = api._resolve_workflow_form(case.schema_id, None)
    trace_id = str(uuid4())
    context_flow = api.get_workflow_context_flow()
    snapshot = context_flow.build(ClinicalContextQuery("EVAL-PATIENT", "EVAL-ENCOUNTER", purpose, datetime.now(timezone.utc), schema_id, trace_id, None))
    schema = context_flow.registry.resolve_active(schema_id, purpose, None)
    if uses_audio:
        audio = AudioInput(case.audio_path.read_bytes(), case.audio_path.name, AUDIO_TYPES[case.audio_path.suffix.lower()], trace_id, case.language_hint)
    else:
        audio = AudioInput(b"transcript-fixture", "transcript.txt", "audio/wav", trace_id, case.language_hint)
    started = time.perf_counter()
    record: dict[str, Any] = {"case_id": case.case_id, "schema_id": schema_id, "input": "audio" if uses_audio else "transcript"}
    try:
        result = workflow.generate(snapshot, audio, schema)
    except Exception as exc:  # 모델별 실패 유형(구조화 출력 불가, 시간 초과 등)을 결과에 남긴다.
        record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "total_seconds": round(time.perf_counter() - started, 2)})
        return record
    record.update({
        "status": "ok",
        "total_seconds": round(time.perf_counter() - started, 2),
        "stt_seconds": round(stt.seconds, 2) if uses_audio else None,
        "llm_seconds": round(generator.seconds, 2),
        "transcript": result.transcript.text,
        "cer": round(character_error_rate(case.reference_transcript, result.transcript.text), 4) if uses_audio and case.reference_transcript else None,
        "model_version": result.draft.model_version,
        "needs_confirmation": result.draft.needs_confirmation,
        "fields": result.draft.proposed_fields,
        "transcript_candidates": len(result.transcript_gateway_verification.occurrences),
        "draft_candidates": len(result.draft_gateway_verification.occurrences),
        "gateway_status": result.draft_gateway_verification.status,
        "report_validation": result.report_validation.status,
        "report_findings": {item.criterion_id: item.status for item in result.report_validation.criteria},
        # 전사문에 없는 초안 수치(바뀐 활력징후, 지어낸 시각 등)다.
        "numeric_findings": [f"{item.field_id}: {item.text}" for item in result.numeric_findings],
        **score_draft(schema, result.draft.proposed_fields, case.expected),
    })
    return record


def summarize(profile: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    config = settings.get()
    ok = [item for item in records if item["status"] == "ok"]

    def mean(key: str) -> float | None:
        values = [item[key] for item in ok if item.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    required_total = sum(item["required_total"] for item in ok)
    return {
        "profile": profile,
        "stt_model": config.stt_endpoint.model,
        "llm_model": next((item["model_version"] for item in ok), config.llm_endpoint.model),
        "llm_api": config.llm_api,
        "tool_calling": config.llm_tool_calling_enabled,
        "cases": len(records),
        "ok": len(ok),
        "errors": [f"{item['case_id']}: {item['error']}" for item in records if item["status"] != "ok"],
        "required_fill_rate": round(sum(item["required_filled"] for item in ok) / required_total, 3) if required_total else None,
        "keyword_recall": mean("keyword_recall"),
        "forbidden_cases": sum(1 for item in ok if item["forbidden_found"]),
        "numeric_flag_cases": sum(1 for item in ok if item["numeric_findings"]),
        "mean_cer": mean("cer"),
        "mean_llm_seconds": mean("llm_seconds"),
        "mean_stt_seconds": mean("stt_seconds"),
        "mean_total_seconds": mean("total_seconds"),
    }


COLUMNS = ("profile", "llm_model", "ok", "cases", "required_fill_rate", "keyword_recall", "forbidden_cases", "numeric_flag_cases", "mean_cer", "mean_llm_seconds", "mean_total_seconds")


def compare(results_root: Path) -> list[dict[str, Any]]:
    """프로필마다 가장 최근 실행의 요약을 모은다."""
    rows: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return rows
    for profile_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        runs = sorted(path for path in profile_dir.iterdir() if (path / "summary.json").is_file())
        if runs:
            rows.append(json.loads((runs[-1] / "summary.json").read_text()))
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\t".join(COLUMNS))
    for row in rows:
        print("\t".join("" if row.get(column) is None else str(row.get(column)) for column in COLUMNS))


def main(argv: list[str] | None = None) -> int:
    load_dotenv(".env")
    config = settings.get()
    parser = argparse.ArgumentParser(description="N-note STT·LLM 모델 프로필 비교 평가")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="현재 설정(모델 프로필)으로 평가 사례를 실행한다")
    run.add_argument("--cases", type=Path, default=Path("eval/cases"))
    run.add_argument("--out", type=Path, default=Path("eval/results"))
    run.add_argument("--profile", default=None, help="결과에 남길 프로필 이름(기본: N_NOTE_MODEL_PROFILE 또는 LLM 모델명)")
    run.add_argument("--case", action="append", default=None, help="실행할 사례 ID(여러 번 지정 가능)")
    cmp_parser = sub.add_parser("compare", help="프로필별 최근 요약을 표로 출력한다")
    cmp_parser.add_argument("--out", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)

    if args.command == "compare":
        _print_table(compare(args.out))
        return 0

    profile = args.profile or config.model_profile or config.llm_endpoint.model.replace("/", "_")
    cases = [case for case in load_cases(args.cases) if not args.case or case.case_id in args.case]
    if not cases:
        print(f"no evaluation cases in {args.cases}", file=sys.stderr)
        return 1
    run_dir = args.out / profile / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (run_dir / "results.jsonl").open("w") as output:
        for case in cases:
            record = run_case(case)
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{profile}] {case.case_id}: {record['status']} {record.get('total_seconds')}s", file=sys.stderr)
    summary = summarize(profile, records)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    _print_table([summary])
    print(f"results: {run_dir}", file=sys.stderr)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
