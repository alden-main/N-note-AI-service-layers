"""서버의 안전한 기본 설정을 모아 둔 모듈이다."""

from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@cache
def _service_version() -> str:
    """설치된 패키지 메타데이터의 버전이다. 소스만 있는 실행에서는 unknown이다."""
    try:
        return version("main-mcp-server-kodict")
    except PackageNotFoundError:
        return "unknown"


class Settings(BaseSettings):
    """``KODICT_`` 환경 변수에서 읽는 런타임 설정이다.

    환경 변수는 import 시점이 아니라 인스턴스를 만들 때 읽는다. 잘못된 값은 기동 시
    ``ValidationError``가 된다. 생성자 인자는 환경 변수보다 우선한다.
    """

    model_config = SettingsConfigDict(env_prefix="KODICT_", frozen=True)

    database_path: Path = Path("/app/data/korean_standard_dictionary.sqlite3")
    # 이미지에 구운 읽기 전용 사전처럼 프로세스 수명 동안 바뀌지 않는 파일일 때만 참으로 둔다.
    # 참이면 SQLite가 잠금·변경 감지를 생략한다. 실행 중 파일이 바뀌면 잘못된 결과를 읽을 수 있다.
    database_immutable: bool = False
    max_request_body_size: int = 1_048_576
    max_text_length: int = 10_000
    max_results: int = 20
    max_candidates: int = 10
    max_findings: int = 1_000
    max_concurrent_requests: int = 4
    recommendation_data_path: Path = Path(__file__).parent / "data" / "standard_recommendations.v1.json"
    dictionary_version: str = "unknown"
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def service_version(self) -> str:
        """감사 로그에 남길 서버 패키지 버전이다."""
        return _service_version()
