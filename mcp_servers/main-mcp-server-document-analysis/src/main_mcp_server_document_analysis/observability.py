"""민감 원문을 보관하지 않는 최소 감사와 health 상태를 제공한다."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

SERVICE_ID = "report"
# 메모리 감사 이벤트 보관 상한이다. 넘으면 오래된 이벤트부터 버려 장기 실행 시 메모리가 늘지 않게 한다.
MAX_AUDIT_EVENTS = 1000
# 감사 이벤트는 이 logger로 한 줄 JSON으로 내보낸다. 필드는 `AuditEvent`와 같다(원문·citation 없음).
AUDIT_LOGGER = logging.getLogger("main_mcp_server_document_analysis.audit")


def _service_version() -> str:
    """설치된 패키지 메타데이터의 버전이다. 설치 없이 소스로 실행하면 `0+unknown`이다."""
    try:
        return version("main-mcp-server-document-analysis")
    except PackageNotFoundError:
        return "0+unknown"


SERVICE_VERSION = _service_version()


@dataclass(frozen=True)
class AuditEvent:
    """원문·citation·토큰을 제외한 안전한 감사 이벤트다."""

    request_id: str | None
    service_id: str
    service_version: str
    tool_name: str
    result_code: str
    standard_version: str | None
    duration_ms: int


class SafeAuditLog:
    """허용 목록 metadata만 logging으로 내보내고 최근 이벤트를 메모리에도 보관하는 감사 저장소다."""

    def __init__(self, max_events: int = MAX_AUDIT_EVENTS) -> None:
        """최근 ``max_events``개만 보관하는 이벤트 목록을 만든다. deque 추가는 스레드 안전하다."""
        self.events: deque[AuditEvent] = deque(maxlen=max_events)

    def record(self, *, request_id: str | None, tool_name: str, result_code: str, standard_version: str | None, started_at: float) -> None:
        """허용 목록의 metadata만 감사 이벤트로 추가하고 같은 필드를 logging으로 내보낸다."""
        event = AuditEvent(
            request_id, SERVICE_ID, SERVICE_VERSION, tool_name,
            result_code, standard_version, int((perf_counter() - started_at) * 1000),
        )
        self.events.append(event)
        AUDIT_LOGGER.info("%s", json.dumps(asdict(event), ensure_ascii=False))

    def health(self, ready: bool) -> dict[str, str]:
        """민감정보 없는 liveness/readiness 상태를 반환한다."""
        return {"status": "ok" if ready else "unavailable"}
