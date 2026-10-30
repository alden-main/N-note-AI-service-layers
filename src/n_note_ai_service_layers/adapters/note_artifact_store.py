"""확정 간호기록 JSON과 PDF를 파일 시스템에 보관하는 저장소 어댑터이다."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from n_note_ai_service_layers.contracts.codec import from_json_dict, to_json_dict
from n_note_ai_service_layers.contracts.draft_review import FinalizedNursingNote

NOTE_FILE = "note.json"
PDF_FILE = "note.pdf"


class FileNursingNoteStore:
    """확정본을 ``<root>/<note_id>/note.json``으로 한 번만 기록하고 PDF를 함께 보관한다.

    입력: 저장 루트 디렉터리이다.
    출력: 저장·조회된 ``FinalizedNursingNote``와 PDF 경로이다. 같은 확정본은 덮어쓰지 않는다.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, note: FinalizedNursingNote) -> None:
        directory = self._directory(note.note_id)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / NOTE_FILE).open("x", encoding="utf-8") as handle:
            json.dump(to_json_dict(note), handle, ensure_ascii=False, indent=2)

    def get(self, note_id: str) -> FinalizedNursingNote:
        path = self._directory(note_id) / NOTE_FILE
        try:
            return from_json_dict(FinalizedNursingNote, json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError as exc:
            raise LookupError("nursing note not found") from exc

    def save_pdf(self, note_id: str, content: bytes) -> Path:
        path = self._directory(note_id) / PDF_FILE
        path.write_bytes(content)
        return path

    def pdf_path(self, note_id: str) -> Path | None:
        path = self._directory(note_id) / PDF_FILE
        return path if path.is_file() else None

    def _directory(self, note_id: str) -> Path:
        # UUID만 허용해 경로 조작을 막는다.
        try:
            normalized = str(uuid.UUID(note_id))
        except ValueError as exc:
            raise LookupError("nursing note not found") from exc
        return self.root / normalized
