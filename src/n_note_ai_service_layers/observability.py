"""PHI 없이 추적 ID·단계·소요 시간·오류 유형만 남기는 구조화 로그와 OpenTelemetry 추적 도우미이다.

로그: ``n_note`` 로거가 JSON 한 줄을 남긴다(본문 없음).

추적: ``stage()``·``span()``이 OpenTelemetry span을 만든다. ``configure_tracing()``을 부르지 않았거나 꺼져 있으면
OpenTelemetry API의 no-op tracer라 비용이 거의 없다. 켜면(``N_NOTE_OTEL_ENABLED=true``) 표준 환경 변수
``OTEL_EXPORTER_OTLP_ENDPOINT``(예: ``http://phoenix:6006``)로 OTLP/HTTP 전송한다.

본문(전사문·프롬프트·LLM 응답·도구 인자)은 ``N_NOTE_OTEL_CAPTURE_CONTENT=true``일 때만 span에 남긴다.
합성 데이터를 쓰는 개발·평가 환경 전용이다. 꺼져 있으면 span에도 로그와 같은 정보(식별자·개수·시간·오류 유형)만 있다.
예외 메시지는 본문을 담을 수 있으므로 span에도 예외 **유형**만 기록한다.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

LOGGER_NAME = "n_note"
TRACER_NAME = "n_note"
DEFAULT_SERVICE_NAME = "n-note-ai-service"
# span 속성 이름 접두사다. 표준 ``gen_ai.*`` 속성은 LLM 계층이 따로 붙인다.
ATTRIBUTE_PREFIX = "n_note."
# 접두사 없이 그대로 쓰는 표준 속성 이름공간이다(OpenTelemetry GenAI·오류 규약).
_STANDARD_NAMESPACES = ("gen_ai.", "error.", ATTRIBUTE_PREFIX)
# 본문 속성 하나의 최대 길이다. OTLP 수신기의 속성 크기 제한에 걸리지 않게 자른다.
MAX_CONTENT_CHARS = 32_000

_T = TypeVar("_T")
_capture_content = False
_tracing_configured = False


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


def configure_tracing(enabled: bool, capture_content: bool = False) -> bool:
    """OTLP 전송 tracer provider를 한 번만 설치한다. 실제로 켜졌으면 ``True``다.

    ``opentelemetry-sdk``를 불러오지 못하면 경고만 남기고 no-op으로 둔다. 추적 때문에 서비스가
    기동하지 못하는 일은 없게 한다. 본문 기록은 추적이 켜졌을 때만 적용한다.
    """
    global _capture_content, _tracing_configured
    logger = get_logger("observability")
    if not enabled:
        _capture_content = False
        return False
    if _tracing_configured:
        _capture_content = capture_content
        return True
    # compose의 ${VAR:-}는 빈 문자열을 넘긴다. 빈 값은 미설정으로 보고(빈 주소면 exporter가 잘못된 URL을 만든다),
    # 주소가 없으면 켜지 않는다(컨테이너 안의 기본값 localhost:4318로 조용히 보내지 않게 한다).
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"):
        if not os.environ.get(name, "x").strip():
            os.environ.pop(name)
    if not (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")):
        log_event(logger, "tracing_unavailable", logging.WARNING, reason="set OTEL_EXPORTER_OTLP_ENDPOINT")
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log_event(logger, "tracing_unavailable", logging.WARNING, reason="opentelemetry-sdk is not installed")
        return False
    # OTEL_SERVICE_NAME·OTEL_RESOURCE_ATTRIBUTES가 있으면 그 값을 쓴다(명시 속성이 환경 변수보다 우선하므로 없을 때만 준다).
    attributes = {} if os.environ.get("OTEL_SERVICE_NAME") else {"service.name": DEFAULT_SERVICE_NAME}
    provider = TracerProvider(resource=Resource.create(attributes))
    # 내보내기 전에 쿼리 문자열을 가린다(BatchSpanProcessor보다 먼저 붙여야 한다).
    provider.add_span_processor(_redact_url_query())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracing_configured = True
    _capture_content = capture_content
    log_event(
        logger, "tracing_enabled", logging.WARNING if capture_content else logging.INFO,
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "default",
        capture_content=capture_content,
    )
    return True


# 추적하지 않는 요청 경로다. 헬스체크는 10초마다 호출되어 추적 화면을 채우고, 문서 화면은 서비스 동작이 아니다.
UNTRACED_PATHS = frozenset({"/healthz", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


def fastapi_telemetry() -> dict[str, Any]:
    """FastAPI 내장 OpenTelemetry(0.142+) 설정이다. ``FastAPI(telemetry=...)``에 넘긴다.

    FastAPI는 전역 tracer provider가 설치되면(``configure_tracing``) 요청 span을 만든다. 요청 span은 남기고
    (작업·단계·LLM span이 그 아래에 놓인다) 이 서비스의 원칙에 맞지 않는 기본값은 끈다.
    """
    return {
        "tracing": True,
        # 의존성 해석·직렬화 단위 span은 요청마다 여러 개가 생겨 LLM 흐름을 가린다.
        "operation_spans": False,
        # 켜면 검증 실패·처리되지 않은 예외의 메시지와 스택을 남긴다. 본문이 섞일 수 있으므로 끈다(예외 유형은 span에 남는다).
        "logs": False,
        # Phoenix는 metrics를 받지 않는다. 수신기를 정한 뒤 필요하면 켠다.
        "metrics": False,
        # exporter는 configure_tracing만 붙인다. 켜 두면 FastAPI가 환경 변수로 exporter를 하나 더 붙여 같은 span을 두 번 보낸다.
        "auto_configure": False,
        "exclude": lambda scope: scope.get("path") in UNTRACED_PATHS,
    }


def capture_content() -> bool:
    """span에 본문(프롬프트·응답·도구 인자)을 남겨도 되는지다. 기본은 ``False``다."""
    return _capture_content


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """전사문·필드 값 같은 본문은 넣지 않는다. 식별자·상태·개수·시간만 넘긴다.

    추적 중이면 OpenTelemetry trace ID(``otel_trace_id``)를 함께 남겨 로그와 추적 화면을 오갈 수 있게 한다.
    """
    context = trace.get_current_span().get_span_context()
    if context.is_valid and "otel_trace_id" not in fields:
        fields["otel_trace_id"] = format(context.trace_id, "032x")
    logger.log(level, event, extra={"fields": fields})


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def current_span() -> Span:
    return trace.get_current_span()


def set_attributes(span: Span, fields: Mapping[str, Any], prefix: str = ATTRIBUTE_PREFIX) -> None:
    """식별자·개수·상태 같은 단순 값만 span 속성으로 남긴다(``None``·구조체는 건너뛴다)."""
    if not span.is_recording():
        return
    for key, value in fields.items():
        name = _attribute_name(key, prefix)
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(name, value)
        elif isinstance(value, (list, tuple)) and value and all(isinstance(item, (str, bool, int, float)) for item in value):
            span.set_attribute(name, list(value))


def set_content(span: Span, name: str, value: Any) -> None:
    """``capture_content()``일 때만 본문을 span 속성으로 남긴다(문자열이 아니면 JSON으로)."""
    if not _capture_content or not span.is_recording():
        return
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    span.set_attribute(_attribute_name(name), text[:MAX_CONTENT_CHARS])


def _attribute_name(key: str, prefix: str = ATTRIBUTE_PREFIX) -> str:
    return key if key.startswith(_STANDARD_NAMESPACES) else prefix + key


def mark_failed(span: Span, error: BaseException) -> None:
    """예외 유형만 기록한다. 예외 메시지·traceback은 본문을 담을 수 있어 남기지 않는다."""
    name = type(error.__cause__ or error).__name__
    span.set_attribute("error.type", name)
    span.set_status(Status(StatusCode.ERROR, name))


@contextmanager
def span(name: str, trace_id: str | None = None, **fields: Any) -> Iterator[Span]:
    """``n_note.<name>`` span을 연다. 예외는 유형만 기록하고 그대로 올린다."""
    with tracer().start_as_current_span(name if name.startswith(ATTRIBUTE_PREFIX) else ATTRIBUTE_PREFIX + name,
                                        record_exception=False, set_status_on_exception=False) as current:
        set_attributes(current, {"trace_id": trace_id, **fields})
        try:
            yield current
        except BaseException as exc:
            mark_failed(current, exc)
            raise


@contextmanager
def stage(logger: logging.Logger, trace_id: str, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """단계 소요 시간과 성공·실패(예외 유형)를 로그와 span(``n_note.stage.<name>``)으로 남긴다.

    반환한 dict에 결과 지표를 추가하면 로그와 span 속성 양쪽에 들어간다.
    """
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    with span(f"stage.{name}", trace_id, stage=name, **fields) as current:
        try:
            yield extra
        except Exception as exc:
            set_attributes(current, extra)
            log_event(logger, "stage_failed", logging.WARNING, trace_id=trace_id, stage=name,
                      seconds=round(time.perf_counter() - started, 3), error=type(exc.__cause__ or exc).__name__, **fields, **extra)
            raise
        set_attributes(current, extra)
        log_event(logger, "stage_completed", trace_id=trace_id, stage=name,
                  seconds=round(time.perf_counter() - started, 3), **fields, **extra)


def _redact_url_query() -> Any:
    """요청 span의 쿼리 문자열(``url.query``) 값을 가리는 span processor다.

    FastAPI는 토큰 같은 정해진 이름만 가리고 나머지 쿼리 값은 그대로 남긴다. 환자·내원 식별자가 쿼리로 들어오는
    경로가 생겨도 추적 저장소에 남지 않도록 이름만 두고 값을 모두 가린다.
    """
    from opentelemetry.sdk.trace import SpanProcessor

    class RedactUrlQuery(SpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            query = (span.attributes or {}).get("url.query")
            if isinstance(query, str) and query:
                names = [part.split("=", 1)[0] for part in query.split("&") if part]
                span.set_attribute("url.query", "&".join(f"{name}=REDACTED" for name in names))

    return RedactUrlQuery()


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """제출한 쪽의 contextvars(현재 span 포함)를 이어받아 작업을 실행하는 스레드 풀이다.

    표준 ``ThreadPoolExecutor``는 문맥을 넘기지 않아 작업마다 추적이 끊긴 새 root span이 된다.
    """

    def submit(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> Future[_T]:
        context = contextvars.copy_context()
        return super().submit(context.run, fn, *args, **kwargs)
