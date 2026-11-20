"""테스트용 OpenTelemetry 설정이다. 프로세스 전역 tracer provider는 한 번만 설치할 수 있으므로 여기서 공유한다."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from n_note_ai_service_layers import observability

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)


@contextmanager
def recorded_spans(capture_content: bool = False) -> Iterator[list[ReadableSpan]]:
    """블록 안에서 끝난 span을 모은다. ``capture_content``는 ``N_NOTE_OTEL_CAPTURE_CONTENT``와 같다."""
    EXPORTER.clear()
    previous = observability._capture_content
    observability._capture_content = capture_content
    spans: list[ReadableSpan] = []
    try:
        yield spans
    finally:
        observability._capture_content = previous
        spans.extend(EXPORTER.get_finished_spans())
        EXPORTER.clear()


def by_name(spans: list[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [item for item in spans if item.name == name]


def all_attribute_text(spans: list[ReadableSpan]) -> str:
    return "\n".join(str(value) for item in spans for value in (item.attributes or {}).values())
