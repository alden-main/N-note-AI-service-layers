"""서버의 안전한 기본 설정을 모아 둔 모듈이다."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    """환경 변수에서 읽는 런타임 설정이다."""

    database_path: Path = Path(os.getenv("KODICT_DATABASE_PATH", "/app/data/korean_standard_dictionary.sqlite3"))
    max_request_body_size: int = int(os.getenv("KODICT_MAX_REQUEST_BODY_SIZE", "1048576"))
    max_text_length: int = int(os.getenv("KODICT_MAX_TEXT_LENGTH", "10000"))
    max_results: int = int(os.getenv("KODICT_MAX_RESULTS", "20"))
    max_candidates: int = int(os.getenv("KODICT_MAX_CANDIDATES", "10"))
    max_findings: int = int(os.getenv("KODICT_MAX_FINDINGS", "1000"))
    max_concurrent_requests: int = int(os.getenv("KODICT_MAX_CONCURRENT_REQUESTS", "4"))
    recommendation_data_path: Path = Path(os.getenv("KODICT_RECOMMENDATION_DATA_PATH", str(Path(__file__).parent / "data" / "standard_recommendations.v1.json")))
    service_version: str = "0.1.0"
    dictionary_version: str = os.getenv("KODICT_DICTIONARY_VERSION", "unknown")
    host: str = os.getenv("KODICT_HOST", "0.0.0.0")
    port: int = int(os.getenv("KODICT_PORT", "8000"))
