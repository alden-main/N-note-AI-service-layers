"""SQLite 문서 저장소의 읽기 연결(스레드별)과 쓰기 연결(단일·lock)의 일관성을 검증한다."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from n_note_ai_service_layers.adapters.document_store import InMemoryDocumentStore, SqliteDocumentStore, VersionConflict


class SqliteDocumentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = SqliteDocumentStore(Path(self.directory.name) / "state.sqlite")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_writes_are_visible_to_readers_on_other_threads(self) -> None:
        seen: list[object] = []
        self.store.get("c", "k")  # 이 스레드의 읽기 연결을 먼저 연다
        self.store.put("c", "k", {"v": 1})
        reader = threading.Thread(target=lambda: seen.append(self.store.get("c", "k").payload))
        reader.start(); reader.join()
        self.store.put("c", "k", {"v": 2})
        self.assertEqual((seen, self.store.get("c", "k").payload, self.store.fingerprint("c")[0]), ([{"v": 1}], {"v": 2}, 1))

    def test_reads_do_not_wait_for_the_write_lock(self) -> None:
        self.store.put("c", "k", {"v": 1})
        with self.store._lock:  # 다른 쓰기가 진행 중인 상황
            result: list[object] = []
            reader = threading.Thread(target=lambda: result.append(self.store.get("c", "k")))
            reader.start(); reader.join(timeout=2)
            self.assertEqual(result[0].payload, {"v": 1})

    def test_optimistic_locking_still_rejects_stale_writes(self) -> None:
        version = self.store.put("c", "k", {"v": 1}, expected_version=0)
        self.store.put("c", "k", {"v": 2}, expected_version=version)
        with self.assertRaises(VersionConflict):
            self.store.put("c", "k", {"v": 3}, expected_version=version)


class ListFilterTests(unittest.TestCase):
    def test_both_stores_filter_by_top_level_field_in_update_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for store in (InMemoryDocumentStore(), SqliteDocumentStore(Path(directory) / "s.sqlite")):
                with self.subTest(store=type(store).__name__):
                    store.put("h", "a", {"delivery_status": "pending", "n": 1})
                    store.put("h", "b", {"delivery_status": "accepted", "n": 2})
                    store.put("h", "c", {"delivery_status": "pending", "n": 3})
                    self.assertEqual([item.key for item in store.list("h", {"delivery_status": "pending"})], ["a", "c"])
                    self.assertEqual([item.key for item in store.list("h")], ["a", "b", "c"])

    def test_filter_field_must_be_an_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SqliteDocumentStore(Path(directory) / "s.sqlite")
            with self.assertRaises(ValueError):
                store.list("h", {"x') OR 1=1 --": "y"})


if __name__ == "__main__":
    unittest.main()
