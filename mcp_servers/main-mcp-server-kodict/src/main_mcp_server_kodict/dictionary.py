"""읽기 전용 SQLite 색인을 통해 사전 근거를 조회하는 모듈이다."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from threading import local
from typing import Iterator

from .normalization import normalize_surface
from .models import Entry, Evidence, Source


class DictionaryRepository:
    """활성 SQLite DB만 읽는 사전 저장소다."""

    def __init__(self, database_path: Path, *, immutable: bool = False) -> None:
        self.database_path = database_path
        self.immutable = immutable
        # 읽기 전용 연결을 thread마다 하나씩 열어 재사용한다. 도구 호출은 anyio worker thread에서
        # 실행되므로 sqlite3의 같은 thread 제약을 지키면서 요청마다 연결을 여는 비용을 없앤다.
        self._thread_connections = local()

    def is_ready(self) -> bool:
        """DB를 빠르게 열고 필수 스키마를 읽을 수 있는지만 확인한다.

        ``PRAGMA integrity_check``는 데이터베이스 전체를 순회하므로 healthcheck마다
        수행하면 사전 파일 크기에 비례해 느려진다. 무결성 검증은 배포 전 DB 빌드
        단계에서 수행하고, 런타임 readiness는 실제 도구 호출에 필요한 읽기 가능
        여부와 필수 테이블 존재만 확인한다.
        """
        try:
            with self.read_session() as connection:
                row = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'dictionary_entry'"
                ).fetchone()
                return row is not None
        except sqlite3.Error:
            return False

    @contextmanager
    def read_session(self) -> Iterator[sqlite3.Connection]:
        """현재 thread의 읽기 전용 연결을 빌려준다.

        연결은 닫지 않고 같은 thread의 다음 요청이 재사용한다. 한 요청의 여러 조회가 같은
        연결을 공유하므로 연결 비용은 형태소 수가 아니라 thread 수에 비례한다. SQLite 오류가 나면
        손상됐을 수 있는 연결을 버려 다음 요청이 새로 연다.
        """
        connection = self._connect()
        try:
            yield connection
        except sqlite3.Error:
            self._discard(connection)
            raise

    def lookup(
        self, query: str, limit: int = 20, *, connection: sqlite3.Connection | None = None
    ) -> list[Entry]:
        """정규화한 표면형을 매개변수화한 정확 일치 색인 질의로 찾는다."""
        normalized = normalize_surface(query)
        if not normalized:
            return []
        statement = """
        SELECT sf.entry_id, sf.surface, de.payload_json, s.file_name, de.row_number, s.source_hash
        FROM surface_form sf JOIN dictionary_entry de ON de.entry_id = sf.entry_id
        JOIN source_file s ON s.id = de.source_file_id
        WHERE sf.normalized_form = ? ORDER BY sf.entry_id LIMIT ?
        """
        if connection is None:
            with self.read_session() as read_connection:
                rows = read_connection.execute(statement, (normalized, limit)).fetchall()
        else:
            rows = connection.execute(statement, (normalized, limit)).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        """현재 thread의 연결을 반환하고, 없으면 SQLite URI의 mode=ro로 열어 런타임 쓰기를 막는다.

        ``immutable``이면 이미지에 구운 불변 파일로 보고 잠금·변경 감지를 생략한다.
        """
        connection = getattr(self._thread_connections, "connection", None)
        if connection is None:
            options = "mode=ro&immutable=1" if self.immutable else "mode=ro"
            connection = sqlite3.connect(f"file:{self.database_path}?{options}", uri=True)
            self._thread_connections.connection = connection
        return connection

    def _discard(self, connection: sqlite3.Connection) -> None:
        """현재 thread에 보관한 연결을 닫고 지운다."""
        if getattr(self._thread_connections, "connection", None) is connection:
            del self._thread_connections.connection
        connection.close()

    @staticmethod
    def _entry_from_row(row: tuple[str, str, str, str, int, str]) -> Entry:
        """SQLite 조인 결과와 보존된 원본 payload를 외부 근거 모델로 변환한다."""
        entry_id, surface, payload_json, file_name, row_number, source_hash = row
        payload = json.loads(payload_json)
        return Entry(entry_id, surface, Evidence(payload["어휘"], payload["발음"], payload["품사"], payload["뜻풀이"], Source(file_name, row_number, source_hash)))
