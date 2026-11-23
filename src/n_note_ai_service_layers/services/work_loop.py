"""서비스 작업이 도는 프로세스 공유 이벤트 루프다.

초안 생성 작업(``JobRunner``)·양식별 초안·동기 서식 추출·LLM 요청이 모두 이 루프 하나에서 asyncio 작업으로 돈다.
API 이벤트 루프(uvicorn)와 분리되어 있어 HTTP 처리가 작업에 막히지 않고, 요청이 끝나도 작업이 계속된다.

블로킹 경계(STT·비식별화·Gateway·SQLite)는 ``asyncio.to_thread``로 이 루프의 기본 스레드 풀에서 실행한다.
``to_thread``는 호출한 쪽의 contextvars(현재 추적 span)를 넘기므로 span이 작업·양식 추적 아래에 놓인다.

- ``submit(coroutine)``: 어느 스레드에서든 작업을 넘기고 ``concurrent.futures.Future``를 받는다.
- ``run_sync(coroutine)``: 동기 코드(스크립트·테스트·작업 스레드)가 결과를 기다린다. 이 루프 안에서는 부르지 않는다.
- ``await run(coroutine)``: 다른 이벤트 루프(API 라우트)에서 결과를 기다린다.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")

# 블로킹 경계 스레드 수의 최솟값이다. 실제 값은 동시 작업·양식 수로 정한다(``_io_threads``).
MIN_IO_THREADS = 16
# 양식 초안 하나가 동시에 쓰는 블로킹 호출 수의 대략값이다: 초안 검토 2(용어·완결성) + 사전 도구 호출 4.
_IO_CALLS_PER_FORM = 6


def _io_threads() -> int:
    from n_note_ai_service_layers import settings

    config = settings.get()
    # 작업마다 공통 단계(STT·비식별화·전사문 검토) 2개 + 양식별 블로킹 호출이다.
    return max(MIN_IO_THREADS, config.job_concurrency * (2 + config.form_concurrency * _IO_CALLS_PER_FORM))


class _WorkLoop:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def get(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                loop.set_default_executor(ThreadPoolExecutor(max_workers=_io_threads(), thread_name_prefix="work-io"))
                threading.Thread(target=loop.run_forever, name="work-loop", daemon=True).start()
                self._loop = loop
            return self._loop


_work_loop = _WorkLoop()


def loop() -> asyncio.AbstractEventLoop:
    return _work_loop.get()


def on_work_loop() -> bool:
    try:
        return asyncio.get_running_loop() is loop()
    except RuntimeError:
        return False


def submit(coroutine: Coroutine[Any, Any, T]) -> Future[T]:
    """작업 루프에서 코루틴을 시작한다. 호출한 스레드의 contextvars(추적 span)를 이어받는다."""
    return asyncio.run_coroutine_threadsafe(coroutine, loop())


def run_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """작업 루프에서 코루틴을 끝까지 실행하고 결과를 돌려준다(동기 호출자용)."""
    if on_work_loop():
        coroutine.close()
        raise RuntimeError("run_sync must not be called from the work loop; await the coroutine instead")
    return submit(coroutine).result()


async def run(coroutine: Coroutine[Any, Any, T]) -> T:
    """다른 이벤트 루프(API)에서 작업 루프의 결과를 기다린다. 이미 작업 루프면 그대로 기다린다."""
    if on_work_loop():
        return await coroutine
    return await asyncio.wrap_future(submit(coroutine))


async def resolve(value: T | Awaitable[T]) -> T:
    """동기 대역(테스트)과 async 구현을 함께 받는다. 값이 awaitable이면 기다린다."""
    return await value if inspect.isawaitable(value) else value  # type: ignore[misc]
