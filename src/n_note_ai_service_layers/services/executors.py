"""요청마다 새로 만들던 단기 스레드 풀을 대신하는, 프로세스가 공유하는 bounded 스레드 풀이다.

호출할 때마다 ``ThreadPoolExecutor``를 만들면 초안 하나마다 스레드가 생기고 사라지며, worker 전체의 스레드 수가
``작업 수 × 양식 수 × 호출별 풀 크기``로 곱해져 늘어난다. 이름별로 풀을 한 번만 만들어 스레드를 재사용하고,
크기를 worker의 최대 동시 실행 수에 맞춰 상한을 둔다.

교착을 막는 규칙: 한 풀의 작업은 **같은 풀**의 다른 작업을 기다리지 않는다. 계층은 다음과 같다.

    workflow-job(JobRunner) → form-draft(작업별) → draft-review · llm-tool → Gateway 클라이언트 풀

아래 계층의 작업은 위 계층을 기다리지 않으므로 풀이 가득 차도 순환 대기가 생기지 않는다.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from n_note_ai_service_layers.observability import ContextThreadPoolExecutor

T = TypeVar("T")
R = TypeVar("R")

_executors: dict[str, ThreadPoolExecutor] = {}
_lock = threading.Lock()


def shared(name: str, max_workers: int) -> ThreadPoolExecutor:
    """``name`` 풀을 돌려준다. 처음 요청한 크기로 한 번만 만든다(스레드는 필요할 때 생기고 재사용된다)."""
    with _lock:
        executor = _executors.get(name)
        if executor is None:
            # 제출한 쪽의 추적 문맥을 이어받아 양식별 초안·도구 호출 span이 작업 span 아래에 놓인다.
            executor = ContextThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix=name)
            _executors[name] = executor
        return executor


def map_ordered(executor: ThreadPoolExecutor, function: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """``items``를 ``executor``에서 동시에 처리하고 입력 순서대로 결과를 돌려준다(예외는 그대로 올린다)."""
    futures = [executor.submit(function, item) for item in items]
    return [future.result() for future in futures]


def worker_parallelism() -> int:
    """worker 하나에서 동시에 진행될 수 있는 양식 초안 수다(``JOB_CONCURRENCY × FORM_CONCURRENCY``)."""
    from n_note_ai_service_layers import settings

    config = settings.get()
    return config.job_concurrency * config.form_concurrency


def reset_for_tests() -> None:
    """테스트가 설정을 바꿔 풀 크기를 다시 정하게 할 때만 쓴다."""
    with _lock:
        executors = list(_executors.values())
        _executors.clear()
    for executor in executors:
        executor.shutdown(wait=False)
