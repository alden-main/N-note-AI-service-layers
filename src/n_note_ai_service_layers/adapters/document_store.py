"""초안·핸드오프·감사 이력·양식 상태를 JSON 문서로 보관하는 저장소 어댑터이다.

``SqliteDocumentStore``는 여러 uvicorn worker가 같은 파일을 공유해도 일관되며(WAL),
``put(expected_version=...)``으로 동시 수정 시 나중 쓰기를 거부한다(낙관적 잠금).
운영 규모가 커지면 같은 Protocol로 PostgreSQL 구현을 추가한다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class VersionConflict(RuntimeError):
    """``expected_version``과 저장된 문서 버전이 달라 쓰기를 거부했다."""


@dataclass(frozen=True)
class StoredDocument:
    key: str
    version: int
    payload: dict[str, Any]
    updated_at: float


class DocumentStore(Protocol):
    def get(self, collection: str, key: str) -> StoredDocument | None: ...

    def put(self, collection: str, key: str, payload: dict[str, Any], expected_version: int | None = None) -> int:
        """문서를 저장하고 새 버전을 반환한다. ``expected_version=0``은 새 문서만 허용한다."""
        ...

    def delete(self, collection: str, key: str) -> bool: ...

    def list(self, collection: str) -> list[StoredDocument]: ...

    def clear(self, collection: str) -> None: ...

    def purge_older_than(self, collection: str, cutoff: float) -> int: ...

    def fingerprint(self, collection: str) -> tuple[int, float]:
        """컬렉션 변경 여부를 싸게 확인하기 위한 (문서 수, 마지막 갱신 시각)이다."""
        ...


class InMemoryDocumentStore:
    """개발·테스트용 프로세스 메모리 저장소이다. 재시작하면 사라지고 worker 간에 공유되지 않는다."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, StoredDocument]] = {}
        self._lock = threading.Lock()

    def get(self, collection: str, key: str) -> StoredDocument | None:
        return self._items.get(collection, {}).get(key)

    def put(self, collection: str, key: str, payload: dict[str, Any], expected_version: int | None = None) -> int:
        # JSON 왕복으로 SQLite 구현과 같은 직렬화 제약을 갖게 한다.
        payload = json.loads(json.dumps(payload, ensure_ascii=False))
        with self._lock:
            items = self._items.setdefault(collection, {})
            current = items.get(key)
            if expected_version is not None and (current.version if current else 0) != expected_version:
                raise VersionConflict(f"{collection}/{key} was modified concurrently")
            version = (current.version if current else 0) + 1
            items[key] = StoredDocument(key, version, payload, time.time())
            return version

    def delete(self, collection: str, key: str) -> bool:
        with self._lock:
            return self._items.get(collection, {}).pop(key, None) is not None

    def list(self, collection: str) -> list[StoredDocument]:
        return list(self._items.get(collection, {}).values())

    def clear(self, collection: str) -> None:
        with self._lock:
            self._items.pop(collection, None)

    def purge_older_than(self, collection: str, cutoff: float) -> int:
        with self._lock:
            items = self._items.get(collection, {})
            expired = [key for key, item in items.items() if item.updated_at < cutoff]
            for key in expired:
                del items[key]
            return len(expired)

    def fingerprint(self, collection: str) -> tuple[int, float]:
        items = self._items.get(collection, {})
        return len(items), max((item.updated_at for item in items.values()), default=0.0)


class SqliteDocumentStore:
    """SQLite 파일 하나에 컬렉션별 JSON 문서를 보관한다."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS documents (
        collection TEXT NOT NULL,
        key TEXT NOT NULL,
        version INTEGER NOT NULL,
        payload TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (collection, key)
    );
    CREATE INDEX IF NOT EXISTS documents_updated ON documents (collection, updated_at);
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._initialize()

    def _initialize(self, attempts: int = 50) -> None:
        """WAL 전환·테이블 생성이다. 여러 worker가 새 파일을 동시에 열면 잠김(SQLITE_BUSY)이 나므로 재시도한다.

        WAL 전환은 busy timeout을 기다리지 않고 바로 실패할 수 있어 직접 재시도한다.
        """
        for attempt in range(attempts):
            try:
                if self._connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                    self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
                self._connection.executescript(self._SCHEMA)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or attempt == attempts - 1:
                    raise
                time.sleep(0.1 + 0.05 * attempt)

    def _rows(self, sql: str, params: tuple[Any, ...]) -> Iterator[tuple[Any, ...]]:
        with self._lock:
            return iter(self._connection.execute(sql, params).fetchall())

    def get(self, collection: str, key: str) -> StoredDocument | None:
        row = next(self._rows("SELECT key, version, payload, updated_at FROM documents WHERE collection = ? AND key = ?", (collection, key)), None)
        return StoredDocument(row[0], row[1], json.loads(row[2]), row[3]) if row else None

    def put(self, collection: str, key: str, payload: dict[str, Any], expected_version: int | None = None) -> int:
        body = json.dumps(payload, ensure_ascii=False)
        now = time.time()
        with self._lock:
            connection = self._connection
            if expected_version == 0:
                try:
                    connection.execute(
                        "INSERT INTO documents (collection, key, version, payload, updated_at) VALUES (?, ?, 1, ?, ?)",
                        (collection, key, body, now),
                    )
                except sqlite3.IntegrityError as exc:
                    raise VersionConflict(f"{collection}/{key} already exists") from exc
                return 1
            if expected_version is not None:
                # 조건부 UPDATE 한 문장이라 다른 worker의 쓰기와 원자적으로 경합한다.
                cursor = connection.execute(
                    "UPDATE documents SET version = version + 1, payload = ?, updated_at = ? WHERE collection = ? AND key = ? AND version = ?",
                    (body, now, collection, key, expected_version),
                )
                if cursor.rowcount != 1:
                    raise VersionConflict(f"{collection}/{key} was modified concurrently")
                return expected_version + 1
            connection.execute(
                """INSERT INTO documents (collection, key, version, payload, updated_at) VALUES (?, ?, 1, ?, ?)
                   ON CONFLICT (collection, key) DO UPDATE SET version = version + 1, payload = excluded.payload, updated_at = excluded.updated_at""",
                (collection, key, body, now),
            )
            row = connection.execute("SELECT version FROM documents WHERE collection = ? AND key = ?", (collection, key)).fetchone()
            return int(row[0])

    def delete(self, collection: str, key: str) -> bool:
        with self._lock:
            return self._connection.execute("DELETE FROM documents WHERE collection = ? AND key = ?", (collection, key)).rowcount > 0

    def list(self, collection: str) -> list[StoredDocument]:
        rows = self._rows("SELECT key, version, payload, updated_at FROM documents WHERE collection = ? ORDER BY updated_at, key", (collection,))
        return [StoredDocument(row[0], row[1], json.loads(row[2]), row[3]) for row in rows]

    def clear(self, collection: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM documents WHERE collection = ?", (collection,))

    def purge_older_than(self, collection: str, cutoff: float) -> int:
        with self._lock:
            return self._connection.execute("DELETE FROM documents WHERE collection = ? AND updated_at < ?", (collection, cutoff)).rowcount

    def fingerprint(self, collection: str) -> tuple[int, float]:
        row = next(self._rows("SELECT COUNT(*), COALESCE(MAX(updated_at), 0) FROM documents WHERE collection = ?", (collection,)))
        return int(row[0]), float(row[1])


def open_document_store(path: Path | None) -> DocumentStore:
    """경로가 있으면 SQLite, 없으면 메모리 저장소를 연다."""
    return SqliteDocumentStore(path) if path else InMemoryDocumentStore()
