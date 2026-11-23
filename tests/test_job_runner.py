"""작업 실행기(JobRunner)가 작업 루프에서 동시 실행 수·종료 규칙을 지키는지 검증한다."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest

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
