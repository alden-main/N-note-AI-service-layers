"""음성 → 초안 생성처럼 오래 걸리는 작업을 비동기로 실행하고 상태를 저장소에 남긴다.

작업은 요청을 받은 worker 프로세스의 스레드 풀에서 실행되고, 상태는 문서 저장소(SQLite)에
기록되므로 어느 worker로 조회하든 같은 상태를 본다. 음성 바이트는 저장하지 않는다.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from n_note_ai_service_layers.adapters.document_store import DocumentStore, VersionConflict
from n_note_ai_service_layers.observability import get_logger, log_event

JOB_COLLECTION = "workflow_jobs"
IDEMPOTENCY_COLLECTION = "workflow_job_keys"
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
_logger = get_logger("jobs")

Progress = Callable[[str], None]


class JobQueueFull(RuntimeError):
    """worker의 대기열이 가득 찼다. 클라이언트는 ``Retry-After`` 뒤에 다시 요청한다."""


class JobFailure(RuntimeError):
    """작업이 알려진 이유로 실패했다. ``code``와 ``message``가 그대로 작업 상태에 남는다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """작업 상태 문서를 만들고 갱신한다.

    입력: 문서 저장소, 상태가 멈춘 작업을 중단으로 볼 시간, 보관 시간이다.
    출력: ``queued → running → succeeded | failed`` 상태와 진행 단계, 결과·오류를 담은 사전이다.
    """

    def __init__(self, store: DocumentStore, timeout_seconds: float, retention_hours: float | None = None) -> None:
        self.store = store
        self.timeout_seconds = timeout_seconds
        self.retention_hours = retention_hours

    def create(self, kind: str, trace_id: str, request: dict[str, Any], idempotency_key: str | None = None) -> tuple[dict[str, Any], bool]:
        """대기 상태 작업을 만든다. 같은 ``idempotency_key``의 작업이 있으면 그 작업을 반환한다(두 번째 값 ``False``)."""
        if self.retention_hours:
            cutoff = time.time() - self.retention_hours * 3600
            self.store.purge_older_than(JOB_COLLECTION, cutoff)
            self.store.purge_older_than(IDEMPOTENCY_COLLECTION, cutoff)
        job_id = str(uuid4())
        if idempotency_key:
            try:
                self.store.put(IDEMPOTENCY_COLLECTION, idempotency_key, {"job_id": job_id}, expected_version=0)
            except VersionConflict:
                existing = self.store.get(IDEMPOTENCY_COLLECTION, idempotency_key)
                job = self.get(existing.payload["job_id"]) if existing else None
                if job is not None:
                    return job, False
                self.store.put(IDEMPOTENCY_COLLECTION, idempotency_key, {"job_id": job_id})
        now = _now()
        job = {
            "job_id": job_id, "kind": kind, "status": "queued", "stage": None, "completed_stages": [],
            "trace_id": trace_id, "request": request, "result": None, "error": None,
            "created_at": now, "started_at": None, "finished_at": None, "updated_at": now,
        }
        self.store.put(JOB_COLLECTION, job_id, job, expected_version=0)
        return job, True

    def get(self, job_id: str) -> dict[str, Any] | None:
        """작업 상태다. 진행 중인데 오래 갱신되지 않았으면 서버 중단으로 보고 실패로 표시한다."""
        stored = self.store.get(JOB_COLLECTION, job_id)
        if stored is None:
            return None
        job = stored.payload
        if job["status"] not in TERMINAL_STATUSES and time.time() - stored.updated_at > self.timeout_seconds:
            job = {**job, "status": "failed", "error": {"code": "interrupted", "message": "job stopped updating; the server may have restarted"}}
        return job

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        """실행 스레드만 자기 작업을 갱신하므로 마지막 쓰기로 저장한다."""
        stored = self.store.get(JOB_COLLECTION, job_id)
        job = {**(stored.payload if stored else {}), **changes, "updated_at": _now()}
        self.store.put(JOB_COLLECTION, job_id, job)
        return job

    def stage(self, job_id: str, stage: str) -> None:
        stored = self.store.get(JOB_COLLECTION, job_id)
        current = stored.payload if stored else {}
        completed = list(current.get("completed_stages", []))
        if current.get("stage") and current["stage"] not in completed:
            completed.append(current["stage"])
        self.update(job_id, stage=stage, completed_stages=completed)


class JobRunner:
    """작업을 스레드 풀에서 실행하고 결과·오류를 ``JobStore``에 기록한다.

    입력: 작업 저장소, 동시 실행 수, 대기열 상한이다.
    출력: ``submit``은 즉시 반환하고, 작업 함수는 ``progress(stage)``로 진행 단계를 알린다.
    """

    def __init__(self, jobs: JobStore, concurrency: int, max_pending: int) -> None:
        self.jobs = jobs
        self.concurrency = concurrency
        self.max_pending = max_pending
        self._executor = self._new_executor()
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="workflow-job")

    @property
    def pending(self) -> int:
        with self._lock:
            return sum(1 for future in self._futures.values() if not future.done())

    def ensure_capacity(self) -> None:
        if self.pending >= self.max_pending:
            raise JobQueueFull("workflow job queue is full")

    def submit(self, job_id: str, work: Callable[[Progress], dict[str, Any]]) -> None:
        self.ensure_capacity()
        future = self._executor.submit(self._run, job_id, work)
        with self._lock:
            self._futures = {key: value for key, value in self._futures.items() if not value.done()}
            self._futures[job_id] = future

    def _run(self, job_id: str, work: Callable[[Progress], dict[str, Any]]) -> None:
        started = time.perf_counter()
        self.jobs.update(job_id, status="running", started_at=_now())
        try:
            result = work(lambda stage: self.jobs.stage(job_id, stage))
        except JobFailure as exc:
            self.jobs.update(job_id, status="failed", finished_at=_now(), error={"code": exc.code, "message": exc.message})
            log_event(_logger, "job_failed", job_id=job_id, code=exc.code, seconds=round(time.perf_counter() - started, 3))
            return
        except Exception as exc:
            self.jobs.update(job_id, status="failed", finished_at=_now(), error={"code": "internal_error", "message": "nursing note workflow failed"})
            log_event(_logger, "job_failed", job_id=job_id, code="internal_error", error=type(exc).__name__)
            return
        current = self.jobs.get(job_id) or {}
        completed = list(current.get("completed_stages", []))
        if current.get("stage") and current["stage"] not in completed:
            completed.append(current["stage"])
        self.jobs.update(job_id, status="succeeded", stage=None, completed_stages=completed, finished_at=_now(), result=result)
        log_event(_logger, "job_succeeded", job_id=job_id, seconds=round(time.perf_counter() - started, 3))

    def shutdown(self) -> None:
        """대기 중인 작업은 취소해 실패로 남기고, 실행 중인 작업은 끝날 때까지 기다린다."""
        with self._lock:
            futures = dict(self._futures)
        for job_id, future in futures.items():
            if future.cancel():
                self.jobs.update(job_id, status="failed", finished_at=_now(), error={"code": "server_shutdown", "message": "server stopped before the job started"})
        self._executor.shutdown(wait=True)
        # 같은 프로세스에서 앱을 다시 띄우는 경우(테스트 등)를 위해 새 풀로 교체한다.
        self._executor = self._new_executor()
