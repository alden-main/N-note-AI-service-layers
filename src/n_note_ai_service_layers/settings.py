"""로컬 개발용 환경 변수 파일을 안전하게 읽는 설정 도우미이다."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# LLM 초안 생성 중 호출을 허용하는 Gateway 읽기 전용 도구다. 제출·승인·이력 도구는
# upstream 상태를 바꾸므로 LLM tool 목록에서 제외한다.
DEFAULT_LLM_TOOLS = (
    "korean-standard-dictionary__lookup_standard_dictionary",
    "medical-term-review__search_medical_terms",
)
# 전사문·초안 원문 범위 기준 검토 후보를 만드는 Gateway 도구다.
DEFAULT_TERM_REVIEW_TOOLS = (
    "korean-standard-dictionary__review_standard_korean_text",
    "medical-term-review__analyze_document_terms",
)
DEFAULT_REPORT_VALIDATION_TOOL = "report-validation__evaluate_report_against_standard"
# 표준국어 MCP의 기본 text 상한(10,000자)보다 작게 분할한다.
DEFAULT_TERM_REVIEW_CHUNK_CHARS = 8000
# Docker(fonts-nanum)와 macOS에서 바로 쓸 수 있는 TrueType 한글 글꼴 후보다.
DEFAULT_PDF_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
)


@dataclass(frozen=True)
class ModelEndpoint:
    """STT·LLM 모델 서버 접속 정보다. ``base_url``이 없으면 OpenAI API를 사용한다."""

    model: str
    base_url: str | None = None
    api_key: str | None = None

    @property
    def provider(self) -> str:
        return "openai-compatible" if self.base_url else "openai"


def _env(name: str, default: str | None = None) -> str | None:
    """빈 문자열을 미설정으로 취급한다(compose의 ``${VAR:-}`` 전달 대응)."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def stt_endpoint() -> ModelEndpoint:
    """음성 전사 서버다. 로컬 Whisper(vLLM 등)는 ``N_NOTE_STT_BASE_URL``로 지정한다."""
    return ModelEndpoint(
        model=_env("N_NOTE_STT_MODEL") or _env("OPENAI_STT_MODEL", "gpt-4o-transcribe"),
        base_url=_env("N_NOTE_STT_BASE_URL"),
        api_key=_env("N_NOTE_STT_API_KEY"),
    )


def stt_prompt() -> str | None:
    """병동 약어·의학용어 인식률을 높이는 STT 문맥 힌트다."""
    return _env("N_NOTE_STT_PROMPT") or _env("OPENAI_STT_PROMPT")


def llm_endpoint() -> ModelEndpoint:
    """초안 생성 LLM 서버다. 로컬 모델(vLLM 등)은 ``N_NOTE_LLM_BASE_URL``로 지정한다."""
    return ModelEndpoint(
        model=_env("N_NOTE_LLM_MODEL") or _env("OPENAI_DRAFT_MODEL", "gpt-4o-mini"),
        base_url=_env("N_NOTE_LLM_BASE_URL"),
        api_key=_env("N_NOTE_LLM_API_KEY"),
    )


def llm_api() -> str:
    """초안 생성에 쓸 API 형식이다: ``responses``(OpenAI) 또는 ``chat``(OpenAI 호환 서버).

    로컬 서버의 Responses API 지원은 버전마다 달라, ``base_url``이 있으면 기본값을 ``chat``으로 둔다.
    """
    value = (_env("N_NOTE_LLM_API") or ("chat" if llm_endpoint().base_url else "responses")).lower()
    if value not in {"responses", "chat"}:
        raise ValueError("N_NOTE_LLM_API must be 'responses' or 'chat'")
    return value


def llm_tool_calling_enabled() -> bool:
    """초안 생성 중 LLM의 사전 도구 호출 허용 여부다. tool call을 지원하지 않는 모델은 끈다."""
    return (_env("N_NOTE_LLM_TOOL_CALLING", "true") or "true").lower() not in {"0", "false", "no", "off"}


def llm_extra_body() -> dict:
    """OpenAI 호환 서버에 그대로 전달할 추가 요청 필드(JSON)다. 예: thinking 끄기."""
    raw = _env("N_NOTE_LLM_EXTRA_BODY")
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("N_NOTE_LLM_EXTRA_BODY must be a JSON object")
    return value


def llm_max_tokens() -> int | None:
    """초안 생성 최대 출력 토큰 수다. 로컬 서버의 문맥 길이에 맞춰 제한한다."""
    raw = _env("N_NOTE_LLM_MAX_TOKENS")
    return int(raw) if raw else None


def llm_temperature() -> float | None:
    """초안 생성 temperature다. 미설정이면 서버 기본값을 쓴다."""
    raw = _env("N_NOTE_LLM_TEMPERATURE")
    return float(raw) if raw else None


def model_profile() -> str | None:
    """비교 평가 결과에 남길 모델 프로필 이름이다(``deploy/models/<profile>.env``)."""
    return _env("N_NOTE_MODEL_PROFILE")


def mcp_gateway_url() -> str:
    """Return the only permitted MCP endpoint for the AI service."""
    return os.getenv("MCP_GATEWAY_URL", "http://gateway:8080/mcp")


def mcp_gateway_timeout_seconds() -> float:
    """Read a bounded Gateway request timeout without exposing upstream settings."""
    try:
        timeout = float(os.getenv("MCP_GATEWAY_TIMEOUT_SECONDS", "15"))
    except ValueError:
        return 15.0
    return timeout if timeout > 0 else 15.0


def _names(variable: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """쉼표로 구분한 도구 이름 목록을 읽고, 미설정이면 기본값을 사용한다."""
    raw = os.getenv(variable)
    if raw is None:
        return default
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def llm_tool_names() -> tuple[str, ...]:
    """초안 생성 LLM에 노출할 Gateway 도구 이름 목록이다."""
    return _names("N_NOTE_LLM_TOOLS", DEFAULT_LLM_TOOLS)


def term_review_tool_names() -> tuple[str, ...]:
    """표준어·의학용어 검토 후보를 만들 Gateway 도구 이름 목록이다."""
    return _names("N_NOTE_TERM_REVIEW_TOOLS", DEFAULT_TERM_REVIEW_TOOLS)


def report_validation_tool_name() -> str | None:
    """양식 완결성 검증에 사용할 Gateway 도구 이름이다. 빈 값이면 검증을 생략한다."""
    return os.getenv("N_NOTE_REPORT_VALIDATION_TOOL", DEFAULT_REPORT_VALIDATION_TOOL).strip() or None


def term_review_chunk_chars() -> int:
    """Gateway 검토 도구 한 번에 보낼 최대 문자 수다."""
    try:
        value = int(os.getenv("N_NOTE_TERM_REVIEW_CHUNK_CHARS", str(DEFAULT_TERM_REVIEW_CHUNK_CHARS)))
    except ValueError:
        return DEFAULT_TERM_REVIEW_CHUNK_CHARS
    return value if value > 0 else DEFAULT_TERM_REVIEW_CHUNK_CHARS


def forms_dir() -> Path:
    """양식 정의(``<form>/form.json``)와 PDF 양식 파일을 읽을 디렉터리다."""
    return Path(os.getenv("N_NOTE_FORMS_DIR", "forms"))


def enabled_form_ids() -> tuple[str, ...]:
    """활성화할 양식 ID 목록이다. 비어 있으면 양식 디렉터리의 모든 양식을 사용한다."""
    return _names("N_NOTE_ENABLED_FORMS", ())


def default_form_id() -> str | None:
    """workflow 요청에 양식 ID가 없을 때 사용할 기본 양식이다."""
    return os.getenv("N_NOTE_DEFAULT_FORM") or None


def artifact_dir() -> Path:
    """확정 간호기록 JSON과 PDF를 저장할 디렉터리다."""
    return Path(os.getenv("N_NOTE_ARTIFACT_DIR", "data/artifacts"))


def pdf_font_path() -> Path | None:
    """PDF에 포함할 한글 TrueType 글꼴 경로다. 설정이 없으면 알려진 위치를 찾는다."""
    configured = os.getenv("N_NOTE_PDF_FONT_PATH")
    if configured:
        return Path(configured)
    return next((Path(path) for path in DEFAULT_PDF_FONT_CANDIDATES if Path(path).is_file()), None)


def load_dotenv(path: Path | str = ".env") -> None:
    """환경 변수가 아직 없을 때만 간단한 ``KEY=VALUE`` 항목을 불러온다.

    입력: ``path``는 읽을 환경 변수 파일 경로이며 기본값은 ``.env``이다.
    출력: 반환값은 없고, 기존 환경 변수는 덮어쓰지 않는다.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return
    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
