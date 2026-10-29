"""배포 전용 사전 DB 빌드 명령의 진입점이다."""

import argparse
from pathlib import Path

from .importer import ImportValidationError, build_database


def main() -> None:
    """명령행 인자를 받아 XLS 원본을 버전된 SQLite 산출물로 변환한다."""
    parser = argparse.ArgumentParser(description="한국어 표준어 사전 SQLite DB를 빌드합니다.")
    parser.add_argument("--source", type=Path, default=Path("koreandict_xlsx"))
    parser.add_argument("--output", type=Path, default=Path("data/releases"))
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    try:
        print(build_database(arguments.source, arguments.output, arguments.version))
    except ImportValidationError as error:
        for issue in error.issues:
            print(f"{issue.file_name}:{issue.row_number or '-'} {issue.code} {issue.detail}")
        raise SystemExit(1) from error
