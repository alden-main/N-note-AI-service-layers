"""환경 변수(``N_NOTE_*`` 등)로 지정하는 서비스 설정이다.

``get()``은 호출할 때마다 환경 변수를 다시 읽는다. 잘못된 값(숫자가 아닌 timeout 등)은 기본값으로
대체하지 않고 ``pydantic.ValidationError``로 알린다. 로컬 ``.env``는 ``dotenv.load_dotenv()``로 불러온다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, AliasChoices, BeforeValidator, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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
DEFAULT_TIMEZONE = "Asia/Seoul"
# tzdata가 없는 최소 이미지에서도 병동 시각을 잃지 않도록 고정 오프셋을 둔다(서머타임 없는 지역만).
_FIXED_OFFSET_ZONES = {"Asia/Seoul": timezone(timedelta(hours=9), "KST")}
# 비우면 '사용하지 않음'을 뜻하는 환경 변수다. 나머지는 빈 값을 미설정(기본값)으로 본다.
_EMPTY_MEANS_NONE = frozenset({"N_NOTE_LLM_TOOLS", "N_NOTE_TERM_REVIEW_TOOLS", "N_NOTE_REPORT_VALIDATION_TOOL"})


def _split_names(value: Any) -> Any:
    """쉼표로 구분한 이름 목록을 튜플로 읽는다."""
    return tuple(name.strip() for name in value.split(",") if name.strip()) if isinstance(value, str) else value


Names = Annotated[tuple[str, ...], NoDecode, BeforeValidator(_split_names)]


def _env(*names: str) -> AliasChoices:
    """앞의 이름이 우선하는 환경 변수 별칭이다(``N_NOTE_*`` 다음 레거시 ``OPENAI_*``)."""
    return AliasChoices(*names)


@dataclass(frozen=True)
class ModelEndpoint:
    """STT·LLM 모델 서버 접속 정보다. ``base_url``이 없으면 OpenAI API를 사용한다."""

    model: str
    base_url: str | None = None
    api_key: str | None = None

    @property
    def provider(self) -> str:
        return "openai-compatible" if self.base_url else "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="N_NOTE_", extra="ignore")

    # 모델 서버. 로컬 Whisper·LLM(vLLM 등)은 ``*_BASE_URL``로 지정한다.
    stt_model: str = Field("gpt-4o-transcribe", validation_alias=_env("N_NOTE_STT_MODEL", "OPENAI_STT_MODEL"))
    stt_base_url: str | None = None
    stt_api_key: str | None = None
    # 병동 약어·의학용어 인식률을 높이는 STT 문맥 힌트다.
    stt_prompt: str | None = Field(None, validation_alias=_env("N_NOTE_STT_PROMPT", "OPENAI_STT_PROMPT"))
    llm_model: str = Field("gpt-4o-mini", validation_alias=_env("N_NOTE_LLM_MODEL", "OPENAI_DRAFT_MODEL"))
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    # 초안 생성 API 형식이다. 미설정이면 ``base_url``이 있을 때 ``chat``, 없으면 ``responses``다.
    llm_api_override: Literal["responses", "chat"] | None = Field(None, validation_alias="N_NOTE_LLM_API")
    # tool call을 지원하지 않는 모델은 끈다.
    llm_tool_calling_enabled: bool = Field(True, validation_alias="N_NOTE_LLM_TOOL_CALLING")
    # OpenAI 호환 서버에 그대로 전달할 추가 요청 필드(JSON 객체)다. 예: thinking 끄기.
    llm_extra_body: dict[str, Any] = {}
    # 로컬 서버의 문맥 길이에 맞춰 제한한다. 미설정이면 서버 기본값을 쓴다.
    llm_max_tokens: int | None = None
    llm_temperature: float | None = None
    # 비교 평가 결과에 남길 모델 프로필 이름이다(``deploy/models/<profile>.env``).
    model_profile: str | None = None

    # STT 요청 한 번의 제한 시간이다. SDK 기본값(600초)은 요청을 수십 분 붙잡는다.
    stt_timeout_seconds: PositiveFloat = 60.0
    # 초안 생성 LLM 요청 한 번의 제한 시간이다(도구 호출 왕복마다 적용).
    llm_timeout_seconds: PositiveFloat = 90.0
    # 전사문 용어 검토 결과를 LLM 힌트로 기다리는 최대 시간이다. 넘으면 힌트 없이 초안 생성을 시작한다.
    term_hint_budget_seconds: PositiveFloat = 3.0
    # STT·LLM 요청의 SDK 재시도 횟수다. 재시도마다 제한 시간이 다시 적용된다.
    model_max_retries: NonNegativeInt = 1
    # 원본 음성을 외부 SaaS로 보내지 않도록 로컬 STT만 허용한다. 비식별화는 전사 이후이므로 운영에서는 켠다.
    require_local_stt: bool = False

    # AI 서비스가 호출할 수 있는 유일한 MCP endpoint다.
    mcp_gateway_url: str = Field("http://gateway:8080/mcp", validation_alias="MCP_GATEWAY_URL")
    mcp_gateway_timeout_seconds: PositiveFloat = Field(15.0, validation_alias="MCP_GATEWAY_TIMEOUT_SECONDS")
    # Gateway 도구 catalog를 재사용할 시간이다. 요청마다 initialize·tools/list 왕복을 줄인다.
    mcp_gateway_catalog_ttl_seconds: NonNegativeFloat = Field(300.0, validation_alias="MCP_GATEWAY_CATALOG_TTL_SECONDS")
    # 용어 검토 도구·청크를 동시에 호출할 최대 개수다. 1이면 순차 호출한다.
    gateway_max_concurrency: PositiveInt = 4
    # 초안 생성 LLM에 노출할 Gateway 도구 이름 목록이다(쉼표 구분).
    llm_tool_names: Names = Field(DEFAULT_LLM_TOOLS, validation_alias="N_NOTE_LLM_TOOLS")
    # 표준어·의학용어 검토 후보를 만들 Gateway 도구 이름 목록이다.
    term_review_tool_names: Names = Field(DEFAULT_TERM_REVIEW_TOOLS, validation_alias="N_NOTE_TERM_REVIEW_TOOLS")
    # 양식 완결성 검증 Gateway 도구다. 빈 값이면 검증을 생략한다.
    report_validation_tool_name: Annotated[str | None, AfterValidator(lambda value: value or None)] = Field(
        DEFAULT_REPORT_VALIDATION_TOOL, validation_alias="N_NOTE_REPORT_VALIDATION_TOOL")
    # Gateway 검토 도구 한 번에 보낼 최대 문자 수다.
    term_review_chunk_chars: PositiveInt = DEFAULT_TERM_REVIEW_CHUNK_CHARS

    # 양식 정의(``<form>/form.json``)와 PDF 양식 파일을 읽을 디렉터리다.
    forms_dir: Path = Path("forms")
    # 활성화할 양식 ID 목록이다. 비어 있으면 양식 디렉터리의 모든 양식을 사용한다.
    enabled_form_ids: Names = Field((), validation_alias="N_NOTE_ENABLED_FORMS")
    # workflow 요청에 양식 ID가 없을 때 사용할 기본 양식이다.
    default_form_id: str | None = Field(None, validation_alias="N_NOTE_DEFAULT_FORM")
    # 확정 간호기록 JSON과 PDF를 저장할 디렉터리다.
    artifact_dir: Path = Path("data/artifacts")
    # 초안·핸드오프·감사 이력·양식 상태를 보관할 SQLite 파일이다. 미설정이면 프로세스 메모리(개발·테스트용)에
    # 보관하므로, 재시작·다중 worker에서 상태를 유지하려면 반드시 지정한다.
    state_db_path: Path | None = Field(None, validation_alias="N_NOTE_STATE_DB")
    # 확정 전 초안(전사문·환자 ID 포함)을 보관하는 시간이다. 지나면 삭제한다.
    draft_retention_hours: PositiveFloat = 72.0

    # 앞단(브리지·인증 프록시)이 인증한 사용자 ID를 전달하는 헤더 이름이다.
    user_header: str = "X-Authenticated-User"
    # 수정·확정·양식 승인 요청에 인증 사용자 헤더를 필수로 요구할지 여부다.
    require_authenticated_user: bool = False

    # 전사문 사람 이름 비식별화용 로컬 NER 모델(이름 또는 경로)이다. 비우면 규칙 기반만 쓴다.
    # 예: ``Leo97/KoELECTRA-small-v3-modu-ner``. ``uv sync --extra ner``가 필요하다.
    ner_model: str | None = None
    # 사람 이름으로 치환할 NER 최소 신뢰도다. 낮출수록 더 많이(오탐 포함) 가린다.
    ner_min_score: float = Field(0.8, gt=0, le=1)
    # 모델이 사람에 붙이는 라벨이다(모두의 말뭉치 ``PS``, 네이버 NER ``PER``).
    ner_person_labels: Names = Field(("PS", "PER", "PERSON"), validation_alias="N_NOTE_NER_LABELS")
    # NER 추론에 쓸 CPU 스레드 수다. worker 수와 곱해 코어 수를 넘지 않게 둔다.
    ner_threads: PositiveInt = 2

    # worker 한 개가 동시에 실행할 초안 생성 작업 수다.
    job_concurrency: PositiveInt = 4
    # worker 한 개의 대기열 상한이다. 넘으면 새 작업을 429로 거부한다.
    job_max_pending: PositiveInt = 32
    # 작업 상태가 이 시간 동안 갱신되지 않으면(서버 중단 등) 실패(``interrupted``)로 본다.
    job_timeout_seconds: PositiveFloat = 600.0
    log_level: Annotated[str, AfterValidator(str.upper)] = "INFO"
    # 구술 시각('오후 2시')과 감사 시각을 해석할 병동 시간대다. 컨테이너 기본(UTC)이면 00~09시 구술이 전날이 된다.
    timezone_name: str = Field(DEFAULT_TIMEZONE, validation_alias="N_NOTE_TIMEZONE")
    # PDF에 포함할 한글 TrueType 글꼴 경로다. 설정이 없으면 알려진 위치를 찾는다.
    pdf_font_path_override: Path | None = Field(None, validation_alias="N_NOTE_PDF_FONT_PATH")

    @model_validator(mode="before")
    @classmethod
    def _blank_is_unset(cls, values: Any) -> Any:
        """compose의 ``${VAR:-}``가 넘기는 빈 문자열은 미설정으로 본다."""
        if not isinstance(values, dict):
            return values
        cleaned = {key: value.strip() if isinstance(value, str) else value for key, value in values.items()}
        return {key: value for key, value in cleaned.items() if value != "" or key.upper() in _EMPTY_MEANS_NONE}

    @property
    def stt_endpoint(self) -> ModelEndpoint:
        return ModelEndpoint(self.stt_model, self.stt_base_url, self.stt_api_key)

    @property
    def llm_endpoint(self) -> ModelEndpoint:
        return ModelEndpoint(self.llm_model, self.llm_base_url, self.llm_api_key)

    @property
    def llm_api(self) -> str:
        """로컬 서버의 Responses API 지원은 버전마다 달라, ``base_url``이 있으면 기본값을 ``chat``으로 둔다."""
        return self.llm_api_override or ("chat" if self.llm_base_url else "responses")

    @property
    def local_timezone(self) -> tzinfo:
        try:
            return ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            if self.timezone_name in _FIXED_OFFSET_ZONES:
                return _FIXED_OFFSET_ZONES[self.timezone_name]
            raise ValueError(f"unknown N_NOTE_TIMEZONE: {self.timezone_name}") from None

    @property
    def pdf_font_path(self) -> Path | None:
        if self.pdf_font_path_override:
            return self.pdf_font_path_override
        return next((Path(path) for path in DEFAULT_PDF_FONT_CANDIDATES if Path(path).is_file()), None)


def get() -> Settings:
    """현재 환경 변수로 설정을 읽는다."""
    return Settings()
