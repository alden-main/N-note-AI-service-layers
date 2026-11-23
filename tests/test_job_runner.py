"""작업 실행기(JobRunner)가 작업 루프에서 동시 실행 수·종료 규칙을 지키는지 검증한다."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
import unittest.mock

from n_note_ai_service_layers.adapters.document_store import InMemoryDocumentStore
from n_note_ai_service_layers.services.jobs import JobFailure, JobRunner, JobStore


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class JobRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = JobStore(InMemoryDocumentStore(), timeout_seconds=60)

    def _job(self) -> str:
        return self.jobs.create("test", "trace", {})[0]["job_id"]

    def test_async_and_sync_work_both_succeed_with_progress(self) -> None:
        runner = JobRunner(self.jobs, concurrency=2, max_pending=4)

        async def async_work(progress):
            await asyncio.to_thread(progress, "a")
            return {"kind": "async"}

        def sync_work(progress):
            progress("b")
            return {"kind": "sync"}

        first, second = self._job(), self._job()
        runner.submit(first, async_work)
        runner.submit(second, sync_work)
        runner.shutdown()
        self.assertEqual((self.jobs.get(first)["status"], self.jobs.get(first)["completed_stages"]), ("succeeded", ["a"]))
        self.assertEqual(self.jobs.get(second)["result"], {"kind": "sync"})

    def test_known_failures_keep_their_code(self) -> None:
        runner = JobRunner(self.jobs, concurrency=1, max_pending=2)

        async def work(progress):
            raise JobFailure("stt_unavailable", "speech transcription is unavailable")

        job = self._job()
        runner.submit(job, work)
        runner.shutdown()
        self.assertEqual(self.jobs.get(job)["error"]["code"], "stt_unavailable")

    def test_concurrency_limit_and_shutdown_fails_only_jobs_that_never_started(self) -> None:
        runner = JobRunner(self.jobs, concurrency=1, max_pending=4)
        release = threading.Event()
        running = threading.Event()

        async def slow(progress):
            running.set()
            await asyncio.to_thread(release.wait, 3)
            return {"done": True}

        first, second = self._job(), self._job()
        runner.submit(first, slow)
        runner.submit(second, slow)
        _wait_until(running.is_set)
        self.assertEqual(self.jobs.get(second)["status"], "queued")  # 동시 실행 수 1이라 기다린다
        threading.Timer(0.1, release.set).start()
        runner.shutdown()
        self.assertEqual(self.jobs.get(first)["status"], "succeeded")  # 실행 중이던 작업은 끝까지 기다린다
        self.assertEqual(self.jobs.get(second)["error"]["code"], "server_shutdown")


if __name__ == "__main__":
    unittest.main()


class JobSubscriptionTests(unittest.TestCase):
    def test_subscribers_are_notified_on_update_until_unsubscribed(self) -> None:
        jobs = JobStore(InMemoryDocumentStore(), timeout_seconds=60)
        job_id = jobs.create("test", "trace", {})[0]["job_id"]
        calls: list[int] = []
        unsubscribe = jobs.subscribe(job_id, lambda: calls.append(1))
        jobs.update(job_id, status="running")
        jobs.stage(job_id, "stt")
        unsubscribe()
        jobs.update(job_id, status="succeeded")
        self.assertEqual(len(calls), 2)
        self.assertEqual(jobs._listeners, {})

    def test_sse_delivers_a_same_worker_change_without_waiting_for_the_recheck_interval(self) -> None:
        from fastapi.testclient import TestClient

        from n_note_ai_service_layers.api.app import create_app
        from n_note_ai_service_layers.api.routes import workflow_jobs
        from n_note_ai_service_layers.api.state import get_services

        jobs = get_services().jobs
        job_id = jobs.create("test", "trace", {})[0]["job_id"]
        with unittest.mock.patch.object(workflow_jobs, "_SSE_RECHECK_SECONDS", 30.0), TestClient(create_app()) as client:
            threading.Timer(0.3, lambda: jobs.update(job_id, status="succeeded", result={})).start()
            started = time.monotonic()
            with client.stream("GET", f"/v1/nursing-note-jobs/{job_id}/events") as stream:
                body = "".join(stream.iter_text())
            elapsed = time.monotonic() - started
        self.assertIn("event: succeeded", body)
        self.assertLess(elapsed, 5.0)  # 재확인 간격(30초)이 아니라 변경 알림으로 깨어난다
