"""구조화 출력으로 간호기록 초안을 생성하는 어댑터다.

LLM 호출(API 형식·도구 왕복)은 ``pydantic_ai_completion``이 맡고, 여기에는 초안에만 해당하는 출력 스키마·결과 검증만 있다.
지침 문장은 ``prompts/prompts.toml``의 ``[draft]``에 있다.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.adapters.llm_completion import StructuredCompletion, StructuredOutput, ToolExecutor, json_text
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema


class DraftGenerationAdapter:
    """양식 허용 필드만 담은 ``DraftResult``를 만든다.

    입력: 구조화 출력 요청기(``completion``), 결과에 남길 모델명, 사전 도구 실행기(없으면 도구 없이)다.
    """

    def __init__(
        self, completion: StructuredCompletion, model_version: str, tools: ToolExecutor | None = None, prompt_version: str | None = None,
    ) -> None:
        self.completion = completion
        self.tools = tools
        self._model = model_version
        # 지정하지 않으면 prompts.toml의 [draft] version이다. 결과(DraftResult.prompt_version)에 그대로 남는다.
        self._prompt_version = prompt_version or prompts.get().draft.version

    def generate(self, payload: dict, schema: FormSchema) -> DraftResult:
        """최소화된 LLM 입력으로 초안을 생성하고 스키마 허용 필드만 반환한다. 잘못된 응답은 ``ValueError``다."""
        output = self._output_format(schema)
        text = self.completion.complete(
            self._instructions(schema), json.dumps(payload, ensure_ascii=False), StructuredOutput(output["name"], output["schema"]),
            self.tools, prompts.get().draft.tool, accept_unconstrained=lambda content: self._unconstrained(content, output["schema"]),
        )
        return self._draft_result(text, schema)

    @staticmethod
    def _instructions(schema: FormSchema) -> str:
        """공통 작성 지침에 양식 제목과 양식별 지침을 덧붙인다(도구 안내는 도구를 줄 때 요청기가 붙인다)."""
        draft = prompts.get().draft
        instructions = draft.instructions + prompts.render(draft.form_title, title=schema.display_title)
        if schema.instructions:
            instructions += prompts.render(draft.form_instructions, instructions=schema.instructions)
        return instructions

    def _draft_result(self, output_text: Any, schema: FormSchema) -> DraftResult:
        """구조화 응답 JSON을 검증하고 스키마의 LLM 허용 필드만 남긴다."""
        try:
            parsed = json.loads(output_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM response did not contain valid structured draft output") from exc
        if not isinstance(parsed, dict):
            raise ValueError("structured draft output must be an object")
        if "proposed_fields" not in parsed:
            # 구조화 출력을 강제하지 못하는 서버에서 모델이 임의 구조를 내면 빈 초안으로 오인하지 않는다.
            raise ValueError("structured draft output did not follow the draft schema")
        allowed = {field.field_id for field in schema.fields if field.llm_allowed}
        proposed = parsed["proposed_fields"]
        if not isinstance(proposed, dict):
            raise ValueError("proposed_fields must be an object")
        evidence = _evidence_by_field(parsed.get("evidence_refs", []))
        return DraftResult(
            proposed_fields={key: value for key, value in proposed.items() if key in allowed and value is not None},
            evidence_refs={key: value for key, value in evidence.items() if key in allowed},
            needs_confirmation=bool(parsed.get("needs_confirmation", False)),
            model_version=self._model,
            prompt_version=self._prompt_version,
        )

    @staticmethod
    def _unconstrained(content: Any, output_schema: dict[str, Any]) -> str | None:
        """문법 제약 없이 생성된 답(Chat 도구 단계)이 출력 스키마를 만족할 때만 쓴다. 아니면 ``None``이다.

        생략된 필드는 null로 본다(strict 출력에서도 null 필드는 초안에 남지 않는다).
        """
        try:
            parsed = json.loads(json_text(content))
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("proposed_fields"), dict):
            return None
        for field_id in output_schema["properties"]["proposed_fields"]["properties"]:
            parsed["proposed_fields"].setdefault(field_id, None)
        evidence = parsed.get("evidence_refs", [])
        if isinstance(evidence, dict):
            # 필드별 객체로 답한 경우도 같은 근거이므로 배열 형식으로 맞춘다.
            evidence = [{"field_id": key, "refs": value} for key, value in evidence.items()]
        parsed["evidence_refs"] = evidence
        parsed.setdefault("needs_confirmation", False)
        if not Draft202012Validator(output_schema).is_valid(parsed):
            return None
        return json.dumps(parsed, ensure_ascii=False)

    @staticmethod
    def _output_format(schema: FormSchema) -> dict:
        """스키마의 LLM 허용 필드에 맞는 OpenAI JSON 스키마를 만든다.

        입력: ``schema``는 필드 ID와 자료형, LLM 허용 여부를 제공한다.
        출력: Responses API ``text.format``에 전달할 JSON 스키마 사전이다.
        """
        allowed_fields = [field for field in schema.fields if field.llm_allowed]
        allowed = [field.field_id for field in allowed_fields]
        value_schemas = {
            field.field_id: DraftGenerationAdapter._value_schema(field)
            for field in allowed_fields
        }
        # 채운 필드의 근거만 배열로 받는다. 필드별 객체(모든 필드 필수)는 null 필드마다 빈 배열을 출력하게 해
        # 출력 토큰이 필드 수에 비례해 늘어난다.
        evidence_item = {
            "type": "object",
            "properties": {
                "field_id": {"type": "string", "enum": allowed} if allowed else {"type": "string"},
                "refs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["field_id", "refs"],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "name": "nursing_draft",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "proposed_fields": {"type": "object", "properties": value_schemas, "required": allowed, "additionalProperties": False},
                    "evidence_refs": {"type": "array", "items": evidence_item},
                    "needs_confirmation": {"type": "boolean"},
                },
                "required": ["proposed_fields", "evidence_refs", "needs_confirmation"],
                "description": "Allowed draft field IDs: " + ", ".join(allowed),
            },
        }

    @staticmethod
    def _value_schema(field: FormField) -> dict:
        """필드 라벨·작성 지침·선택지를 담은 nullable JSON 스키마를 만든다."""
        if field.options:
            value = {"type": "string", "enum": list(field.options)}
        else:
            value = DraftGenerationAdapter._json_type(field.type)
        schema: dict[str, Any] = {"anyOf": [value, {"type": "null"}]}
        description = " - ".join(item for item in (field.label, field.description) if item)
        if field.type == "date":
            description += " (YYYY-MM-DD)"
        elif field.type == "datetime":
            description += " (YYYY-MM-DD HH:MM)"
        if description.strip():
            schema["description"] = description.strip()
        return schema

    @staticmethod
    def _json_type(field_type: str) -> dict:
        """양식 필드 자료형(``FieldType``)을 JSON 스키마 자료형으로 바꾼다. 문자·날짜 계열은 문자열이다."""
        if field_type == "array":
            return {"type": "array", "items": {"type": "string"}}
        if field_type in ("integer", "number", "boolean"):
            return {"type": field_type}
        return {"type": "string"}


def _evidence_by_field(evidence: Any) -> dict[str, tuple[str, ...]]:
    """``[{"field_id", "refs"}]``(현재 schema) 또는 ``{field_id: [refs]}``(구조화 출력을 강제하지 못한 서버)를 필드별 근거로 바꾼다."""
    if isinstance(evidence, dict):
        pairs = [(key, value) for key, value in evidence.items()]
    elif isinstance(evidence, list):
        pairs = [(item.get("field_id"), item.get("refs")) for item in evidence if isinstance(item, dict)]
    else:
        raise ValueError("evidence_refs must be an array or object")
    merged: dict[str, tuple[str, ...]] = {}
    for field_id, refs in pairs:
        if isinstance(field_id, str) and isinstance(refs, list):
            merged[field_id] = merged.get(field_id, ()) + tuple(str(item) for item in refs)
    return merged
