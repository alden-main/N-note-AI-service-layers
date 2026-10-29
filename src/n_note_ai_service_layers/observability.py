"""PHI 없이 추적 ID·단계·소요 시간·오류 유형만 남기는 구조화 로그 도우미이다."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LOGGER_NAME = "n_note"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {"level": record.levelname, "logger": record.name, "event": record.getMessage()}
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """``n_note`` 로거에 JSON 한 줄 형식 handler를 한 번만 붙인다."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(handler, "_n_note", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        handler._n_note = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """전사문·필드 값 같은 본문은 넣지 않는다. 식별자·상태·개수·시간만 넘긴다."""
    logger.log(level, event, extra={"fields": fields})


@contextmanager
def stage(logger: logging.Logger, trace_id: str, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """단계 소요 시간과 성공·실패(예외 유형)를 기록한다. 반환한 dict에 결과 지표를 추가할 수 있다."""
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    try:
        yield extra
    except Exception as exc:
        log_event(logger, "stage_failed", logging.WARNING, trace_id=trace_id, stage=name,
                  seconds=round(time.perf_counter() - started, 3), error=type(exc.__cause__ or exc).__name__, **fields, **extra)
        raise
    log_event(logger, "stage_completed", trace_id=trace_id, stage=name,
              seconds=round(time.perf_counter() - started, 3), **fields, **extra)
