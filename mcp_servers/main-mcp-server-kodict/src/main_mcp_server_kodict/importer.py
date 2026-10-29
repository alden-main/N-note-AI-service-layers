"""분할 XLS 원본을 검증하고 SQLite 배포 산출물을 만드는 적재 모듈이다."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from typing import Iterable

from .normalization import normalize_surface


HEADERS = ("어휘", "구성 단위", "고유어 여부", "원어", "원어·어종", "어원", "주표제어", "부표제어", "발음", "활용", "검색용 이형태", "품사", "공통 문형", "의미 문형", "공통 문법", "의미 문법", "뜻풀이", "용례", "범주", "전문 분야", "속담", "관용구", "대역어", "생물 분류군 정보", "멀티미디어", "관련 어휘", "한 걸음 더")
SCHEMA_VERSION = "1"
SURFACE_COLUMNS = ("어휘", "주표제어", "부표제어", "활용", "검색용 이형태")


@dataclass(frozen=True)
class ImportIssue:
    """적재를 중단하게 하는 원본 파일 또는 행 오류다."""

    file_name: str
    row_number: int | None
    code: str
    detail: str


class ImportValidationError(Exception):
    """원본 계약이 어겨져 DB 생성이 중단된 경우의 예외다."""

    def __init__(self, issues: list[ImportIssue]) -> None:
        self.issues = issues
        super().__init__("원본 XLS 검증에 실패했습니다.")


def split_surfaces(value: str) -> Iterable[str]:
    """복수 표면형 필드를 줄바꿈·쉼표·세미콜론 기준으로 안전하게 나눈다."""
    for item in re.split(r"[\n,;]", value):
        cleaned = item.strip()
        if cleaned:
            yield cleaned


def _create_schema(connection: sqlite3.Connection) -> None:
    """원본 보존 테이블, 색인 테이블, 적재 오류 테이블을 생성한다."""
    connection.executescript("""
    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE source_file (id INTEGER PRIMARY KEY, file_name TEXT NOT NULL, source_hash TEXT NOT NULL UNIQUE, header_json TEXT NOT NULL);
    CREATE TABLE dictionary_entry (entry_id TEXT PRIMARY KEY, source_file_id INTEGER NOT NULL REFERENCES source_file(id), row_number INTEGER NOT NULL, payload_json TEXT NOT NULL, UNIQUE(source_file_id, row_number));
    CREATE TABLE surface_form (id INTEGER PRIMARY KEY, entry_id TEXT NOT NULL REFERENCES dictionary_entry(entry_id), source_column TEXT NOT NULL, surface TEXT NOT NULL, normalized_form TEXT NOT NULL);
    CREATE TABLE import_issue (id INTEGER PRIMARY KEY, file_name TEXT NOT NULL, row_number INTEGER, code TEXT NOT NULL, detail TEXT NOT NULL);
    CREATE INDEX idx_surface_form_normalized ON surface_form(normalized_form);
    CREATE INDEX idx_surface_form_entry ON surface_form(entry_id);
    """)


def build_database(source_dir: Path, output_dir: Path, dataset_version: str) -> Path:
    """모든 XLS를 임시 DB에 적재·검증한 후 버전 디렉터리로 원자적으로 공개한다.

    Args:
        source_dir: 분할 XLS 파일이 있는 디렉터리.
        output_dir: 버전별 DB와 manifest를 둘 상위 디렉터리.
        dataset_version: 산출물을 구분하는 데이터셋 버전.

    Returns:
        검증을 통과해 공개된 SQLite DB 경로.

    Raises:
        ImportValidationError: 헤더·행·파일 계약이 맞지 않을 때.
    """
    files = sorted(source_dir.glob("*.xls"))
    try:
        import xlrd
    except ImportError as error:
        raise RuntimeError("DB 빌드에는 xlrd optional dependency가 필요합니다.") from error
    issues: list[ImportIssue] = []
    if not files:
        raise ImportValidationError([ImportIssue(str(source_dir), None, "no_source_files", "XLS 파일이 없습니다.")])
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temporary:
        temp_dir, database = Path(temporary), Path(temporary) / "korean_standard_dictionary.sqlite3"
        connection = sqlite3.connect(database)
        _create_schema(connection)
        rows = 0
        hashes: dict[str, str] = {}
        for path in files:
            file_hash = sha256(path.read_bytes()).hexdigest()
            hashes[path.name] = file_hash
            try:
                sheet = xlrd.open_workbook(path).sheet_by_index(0)
            except Exception as error:
                issues.append(ImportIssue(path.name, None, "parse_error", type(error).__name__))
                continue
            header = tuple(str(value).strip() for value in sheet.row_values(0))
            if header != HEADERS:
                issues.append(ImportIssue(path.name, 1, "invalid_header", "27개 열 헤더가 일치하지 않습니다."))
                continue
            source_id = connection.execute("INSERT INTO source_file(file_name, source_hash, header_json) VALUES (?, ?, ?)", (path.name, file_hash, json.dumps(header, ensure_ascii=False))).lastrowid
            for row_index in range(1, sheet.nrows):
                values = sheet.row_values(row_index)
                if len(values) != len(HEADERS):
                    issues.append(ImportIssue(path.name, row_index + 1, "invalid_row", "열 수가 27개가 아닙니다."))
                    continue
                payload = dict(zip(HEADERS, (str(value) for value in values), strict=True))
                entry_id = f"{path.name}:{row_index + 1}"
                connection.execute("INSERT INTO dictionary_entry(entry_id, source_file_id, row_number, payload_json) VALUES (?, ?, ?, ?)", (entry_id, source_id, row_index + 1, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
                for column in SURFACE_COLUMNS:
                    for surface in split_surfaces(payload[column]):
                        normalized = normalize_surface(surface)
                        if normalized:
                            connection.execute("INSERT INTO surface_form(entry_id, source_column, surface, normalized_form) VALUES (?, ?, ?, ?)", (entry_id, column, surface, normalized))
                rows += 1
        for issue in issues:
            connection.execute("INSERT INTO import_issue(file_name, row_number, code, detail) VALUES (?, ?, ?, ?)", (issue.file_name, issue.row_number, issue.code, issue.detail))
        if issues:
            connection.rollback()
            connection.close()
            raise ImportValidationError(issues)
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", (("schema_version", SCHEMA_VERSION), ("dataset_version", dataset_version), ("row_count", str(rows))))
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            connection.close()
            raise ImportValidationError([ImportIssue("database", None, "integrity_error", "SQLite 무결성 검사 실패")])
        connection.commit()
        connection.close()
        release = output_dir / dataset_version
        release.mkdir(exist_ok=True)
        target = release / database.name
        database.replace(target)
        manifest = {"dataset_version": dataset_version, "schema_version": SCHEMA_VERSION, "row_count": rows, "source_hashes": hashes}
        (release / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
