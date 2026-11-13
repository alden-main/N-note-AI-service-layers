"""실행 중인 N-note API에 실제 음성을 보내 모델 프로필이 끝까지 동작하는지 확인하는 스모크 테스트다.

``scripts/smoke-test.sh <profile>``이 프로필을 띄운 뒤 ai-service 컨테이너 안에서 실행한다. 확인 항목:

1. 모델 서버: 로컬 STT·LLM 서버(``*_BASE_URL``)의 ``/models`` 응답과 서빙 모델 이름
2. 단독 전사: ``POST /v1/transcriptions``의 지연 시간과 참조 전사문 대비 CER
3. 녹음 세션: 음성을 청크(WAV)로 나눠 ``/v1/recording-sessions``로 보낸 청크별 지연 시간과 누적 전사문 CER
4. 초안 생성: 녹음 세션을 마쳐 시작한 작업의 단계·소요 시간·채운 필드 수(``--stt-only``면 생략)

결과는 ``eval/smoke/<profile>/<시각>.json``에 남긴다. 필수 단계가 하나라도 실패하면 종료 코드 1이다.
API 키·원본 음성은 결과에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters.audio_preprocessing import SAMPLE_RATE, decode_to_pcm, encode_wav
from n_note_ai_service_layers.run_model_eval import character_error_rate

DEFAULT_CASE = Path("eval/cases/stt-admission-tts")
TERMINAL = ("succeeded", "failed")


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


def _request(method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 120) -> tuple[Any, float]:
    """JSON 응답과 소요 시간(초)을 돌려준다."""
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise ApiError(exc.code, exc.read().decode(errors="replace")) from None
    elapsed = time.perf_counter() - started
    return (json.loads(raw) if raw else None), elapsed


def _json(method: str, url: str, payload: dict[str, Any], timeout: float = 120) -> tuple[Any, float]:
    return _request(method, url, json.dumps(payload).encode(), {"Content-Type": "application/json"}, timeout)


def _multipart(url: str, fields: dict[str, str], filename: str, content: bytes, mime: str, timeout: float) -> tuple[Any, float]:
    boundary = uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
        + content + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return _request("POST", url, b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout)


def _run_step(report: dict[str, Any], name: str, func, required: bool = True) -> dict[str, Any]:
    """단계 하나를 실행해 결과·오류·소요 시간을 보고서에 남긴다. 실패해도 다음 단계는 계속 실행한다."""
    started = time.perf_counter()
    entry: dict[str, Any] = {"required": required}
    try:
        entry.update(func() or {})
        entry["status"] = "ok"
    except Exception as exc:
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    entry["seconds"] = round(time.perf_counter() - started, 3)
    report["steps"][name] = entry
    _print_step(name, entry)
    return entry


def _print_step(name: str, entry: dict[str, Any]) -> None:
    keys = [key for key in entry if key not in ("required", "status", "seconds", "chunks", "stages", "error", "transcript_preview")]
    details = ", ".join(f"{key}={entry[key]}" for key in keys)
    mark = {"ok": "OK  ", "failed": "FAIL", "skipped": "SKIP"}[entry["status"]]
    print(f"[{mark}] {name:<16} {entry['seconds']:>8.2f}s  {details}", file=sys.stderr)
    if entry.get("error"):
        print(f"       error: {entry['error']}", file=sys.stderr)


def _model_server(base_url: str | None, api_key: str | None) -> dict[str, Any]:
    if not base_url:
        return {"server": "openai-api", "checked": False}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body, elapsed = _request("GET", base_url.rstrip("/") + "/models", headers=headers, timeout=10)
    return {"models": [item.get("id") for item in (body or {}).get("data", [])], "latency_seconds": round(elapsed, 3)}


def _filled(value: Any) -> bool:
    return value not in (None, "", [], {}) and not (isinstance(value, dict) and not any(_filled(item) for item in value.values()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = settings.get()
    api = args.api_url.rstrip("/")
    case_dir: Path = args.case
    audio_path = args.audio or case_dir / "audio.wav"
    reference_path = case_dir / "reference_transcript.txt"
    reference = reference_path.read_text().strip() if reference_path.is_file() else None
    case_meta = json.loads((case_dir / "case.json").read_text()) if (case_dir / "case.json").is_file() else {}
    schema_id = args.schema_id or case_meta.get("schema_id")
    content = audio_path.read_bytes()
    profile = config.model_profile or "unknown"
    report: dict[str, Any] = {
        "profile": profile,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "api_url": api,
        "stt": {"model": config.stt_endpoint.model, "base_url": config.stt_base_url, "normalize_audio": config.stt_normalizes_audio, "vad": config.stt_vad_enabled},
        "llm": {"model": config.llm_endpoint.model, "base_url": config.llm_base_url, "api": config.llm_api, "tool_calling": config.llm_tool_calling_enabled},
        "audio": {"file": audio_path.name, "bytes": len(content)},
        "steps": {},
    }
    print(f"N-note smoke test: profile={profile} api={api} audio={audio_path}", file=sys.stderr)

    def api_health() -> dict[str, Any]:
        body, _elapsed = _request("GET", f"{api}/healthz", timeout=10)
        return {"response": body}
    _run_step(report, "api_health", api_health)

    def stt_server() -> dict[str, Any]:
        return _model_server(config.stt_base_url, config.stt_api_key)
    _run_step(report, "stt_server", stt_server)

    if not args.stt_only:
        def llm_server() -> dict[str, Any]:
            return _model_server(config.llm_base_url, config.llm_api_key)
        _run_step(report, "llm_server", llm_server)

    def transcription() -> dict[str, Any]:
        body, elapsed = _multipart(f"{api}/v1/transcriptions", {"language_hint": "ko"}, audio_path.name, content, "audio/wav", args.timeout)
        result = {"latency_seconds": round(elapsed, 3), "chars": len(body["text"]), "model": body["model"], "transcript_preview": body["text"][:120]}
        if reference:
            result["cer"] = round(character_error_rate(reference, body["text"]), 4)
        return result
    _run_step(report, "transcription", transcription)

    session: dict[str, Any] = {}

    def recording_chunks() -> dict[str, Any]:
        samples = decode_to_pcm(content, 3600)
        size = int(args.chunk_seconds * SAMPLE_RATE)
        created, _ = _json("POST", f"{api}/v1/recording-sessions", {"patient_id": "SMOKE-P0001", "encounter_id": "SMOKE-E0001", "language_hint": "ko"})
        session["id"] = created["session_id"]
        chunks = []
        transcript = ""
        for sequence, start in enumerate(range(0, samples.size, size)):
            body, elapsed = _multipart(
                f"{api}/v1/recording-sessions/{session['id']}/chunks", {"sequence": str(sequence)},
                f"chunk-{sequence}.wav", encode_wav(samples[start:start + size]), "audio/wav", args.timeout,
            )
            chunks.append({"sequence": sequence, "seconds": round(elapsed, 3), "chars": len(body["text"])})
            transcript = body["transcript"]
        latencies = [chunk["seconds"] for chunk in chunks]
        result = {
            "chunk_seconds": args.chunk_seconds, "count": len(chunks),
            "mean_latency_seconds": round(sum(latencies) / len(latencies), 3), "max_latency_seconds": max(latencies),
            "silent_chunks": sum(1 for chunk in chunks if chunk["chars"] == 0),
            "chunks": chunks, "transcript_preview": transcript[:120],
        }
        if reference:
            result["cer"] = round(character_error_rate(reference, transcript), 4)
        return result
    _run_step(report, "recording_chunks", recording_chunks)

    if args.stt_only:
        report["steps"]["draft_workflow"] = {"required": False, "status": "skipped", "seconds": 0.0, "reason": "--stt-only"}
        _print_step("draft_workflow", report["steps"]["draft_workflow"])
        if session.get("id"):
            try:
                _request("DELETE", f"{api}/v1/recording-sessions/{session['id']}")
            except Exception:
                pass
    else:
        def draft_workflow() -> dict[str, Any]:
            if not session.get("id"):
                raise RuntimeError("recording session was not created")
            payload = {key: value for key, value in (("schema_id", schema_id), ("purpose", args.purpose)) if value}
            job, _ = _json("POST", f"{api}/v1/recording-sessions/{session['id']}/workflows", payload)
            deadline = time.monotonic() + args.job_timeout
            while job["status"] not in TERMINAL and time.monotonic() < deadline:
                time.sleep(1)
                job, _ = _request("GET", f"{api}/v1/nursing-note-jobs/{job['job_id']}")
            if job["status"] != "succeeded":
                raise RuntimeError(f"job {job['status']}: {job.get('error') or 'timed out'}")
            fields = job["result"]["draft"]["proposed_fields"]
            return {
                "schema_id": job["result"]["form"]["schema_id"], "model": job["result"]["draft"]["model_version"],
                "stages": job.get("completed_stages"), "fields": len(fields), "filled": sum(_filled(value) for value in fields.values()),
                "term_review": job["result"]["transcript_verification"]["status"],
                "report_validation": job["result"]["report_validation"]["status"],
                "numeric_findings": len(job["result"]["numeric_verification"]["findings"]),
            }
        _run_step(report, "draft_workflow", draft_workflow)

    if args.audio is not None and not args.include_text:
        # 실제 녹음일 수 있으므로 비식별화했더라도 전사문 내용은 보고서에 남기지 않는다.
        for step in report["steps"].values():
            step.pop("transcript_preview", None)
    report["ok"] = all(step["status"] != "failed" for step in report["steps"].values() if step["required"])
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실행 중인 N-note API로 모델 프로필 스모크 테스트를 한다")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="N-note API 주소(기본: ai-service 컨테이너 안에서 자기 자신)")
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE, help="audio.wav·reference_transcript.txt·case.json이 있는 평가 사례")
    parser.add_argument("--audio", type=Path, default=None, help="사례 대신 쓸 음성 파일")
    parser.add_argument("--schema-id", default=None, help="초안을 만들 양식(기본: 사례의 schema_id 또는 서버 기본 양식)")
    parser.add_argument("--purpose", default=None, help="양식 목적(기본: 활성 양식의 목적)")
    parser.add_argument("--chunk-seconds", type=float, default=20.0, help="녹음 세션 청크 길이(초)")
    parser.add_argument("--stt-only", action="store_true", help="LLM 서버 확인과 초안 생성을 건너뛴다")
    parser.add_argument("--timeout", type=float, default=180.0, help="요청 하나의 제한 시간(초)")
    parser.add_argument("--job-timeout", type=float, default=900.0, help="초안 생성 작업을 기다릴 최대 시간(초)")
    parser.add_argument("--out", type=Path, default=Path("eval/smoke"))
    parser.add_argument("--include-text", action="store_true", help="--audio를 쓸 때도 비식별 전사문 앞부분을 보고서에 남긴다(기본은 동봉 사례만)")
    args = parser.parse_args(argv)
    report = run(args)
    out_dir = args.out / report["profile"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"{'PASSED' if report['ok'] else 'FAILED'}  report: {path}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
