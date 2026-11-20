"""LLM·STT 프롬프트를 ``prompts.toml`` 한 곳에서 읽는다.

지침 문장은 코드에 두지 않고 이 패키지의 ``prompts.toml``에 둔다. 섹션마다 ``version``이 있어 초안 결과·추적 span·
평가 결과에 어떤 지침으로 만든 결과인지 남는다. ``N_NOTE_PROMPTS_FILE``로 다른 파일을 지정해 실험할 수 있다.

파일이 없거나 키가 빠졌으면 기동 시 ``PromptCatalogError``로 멈춘다(빈 지침으로 초안을 만들지 않기 위함).
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS_FILE_ENV = "N_NOTE_PROMPTS_FILE"
DEFAULT_PROMPTS_FILE = Path(__file__).with_name("prompts.toml")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


class PromptCatalogError(RuntimeError):
    """프롬프트 파일을 읽지 못했거나 필요한 키가 없다."""


@dataclass(frozen=True)
class DraftPrompts:
    version: str
    instructions: str
    form_title: str
    form_instructions: str
    tool: str


@dataclass(frozen=True)
class FormExtractionPrompts:
    version: str
    instructions: str


@dataclass(frozen=True)
class StructuredOutputPrompts:
    final_output: str
    final_turn: str


@dataclass(frozen=True)
class SttPrompts:
    version: str
    context: str


@dataclass(frozen=True)
class PromptCatalog:
    source: Path
    draft: DraftPrompts
    form_extraction: FormExtractionPrompts
    structured_output: StructuredOutputPrompts
    stt: SttPrompts


def render(template: str, **values: object) -> str:
    """``{name}`` 자리를 값으로 채운다. 넘기지 않은 이름과 다른 중괄호는 그대로 둔다.

    ``str.format``은 지침의 ``{status, detail}`` 같은 글자 그대로의 중괄호에서 실패하므로 쓰지 않는다.
    """
    return _PLACEHOLDER.sub(lambda match: str(values[match[1]]) if match[1] in values else match[0], template)


def prompts_file() -> Path:
    configured = os.environ.get(PROMPTS_FILE_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_PROMPTS_FILE


@lru_cache(maxsize=4)
def _load(path: Path) -> PromptCatalog:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PromptCatalogError(f"prompt file could not be read: {path}") from exc

    def section(name: str, kind: type) -> object:
        values = raw.get(name)
        names = kind.__dataclass_fields__
        if not isinstance(values, dict) or any(not isinstance(values.get(key), str) for key in names):
            raise PromptCatalogError(f"prompt file {path} needs string keys {sorted(names)} in [{name}]")
        return kind(**{key: values[key] for key in names})

    return PromptCatalog(
        source=path,
        draft=section("draft", DraftPrompts),  # type: ignore[arg-type]
        form_extraction=section("form_extraction", FormExtractionPrompts),  # type: ignore[arg-type]
        structured_output=section("structured_output", StructuredOutputPrompts),  # type: ignore[arg-type]
        stt=section("stt", SttPrompts),  # type: ignore[arg-type]
    )


def get() -> PromptCatalog:
    """현재 프롬프트 모음이다(같은 파일은 한 번만 읽는다)."""
    return _load(prompts_file())


def clear_cache() -> None:
    """테스트·실험에서 파일을 바꾼 뒤 다시 읽게 한다."""
    _load.cache_clear()
