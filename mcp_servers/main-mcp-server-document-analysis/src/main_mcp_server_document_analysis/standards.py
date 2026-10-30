"""버전이 있는 검증 기준 JSON을 공통 IR로 해석하는 구현을 제공한다."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import CriterionSpec, ElementSpec, InputError, ValidationStandard

_RESULT_POLICY = "candidate_only_human_review_required"
_MUTATION_POLICY = "no_automatic_report_modification"


class StandardInterpreter(Protocol):
    """특정 schema_version의 기준을 공통 IR로 변환하는 인터페이스다."""

    schema_version: str

    def interpret(self, raw: dict[str, Any]) -> ValidationStandard:
        """원본 JSON을 검증하고 ValidationStandard로 변환한다."""


def _text(value: Any, field_name: str) -> str:
    """비공백 문자열 필드를 확인해 반환한다."""
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field_name} must be a non-blank string")
    return value.strip()


class StandardInterpreterRegistry:
    """명시적으로 등록된 schema interpreter만 선택하는 레지스트리다."""

    def __init__(self, interpreters: tuple[StandardInterpreter, ...] = ()) -> None:
        """초기 interpreter를 등록한다."""
        self._interpreters: dict[str, StandardInterpreter] = {}
        for interpreter in interpreters:
            self.register(interpreter)

    def register(self, interpreter: StandardInterpreter) -> None:
        """동일 schema_version의 중복 등록을 막고 interpreter를 추가한다."""
        if interpreter.schema_version in self._interpreters:
            raise ValueError(f"interpreter already registered: {interpreter.schema_version}")
        self._interpreters[interpreter.schema_version] = interpreter

    def interpret(self, raw: dict[str, Any]) -> ValidationStandard:
        """schema_version에 맞는 interpreter로 기준 전체를 변환한다."""
        if not isinstance(raw, dict):
            raise InputError("validation_standard must be an object")
        schema_version = raw.get("schema_version")
        interpreter = self._interpreters.get(schema_version)
        if interpreter is None:
            raise InputError("unsupported validation_standard schema_version")
        return interpreter.interpret(raw)


class ReportValidationStandardV1Interpreter:
    """report-validation-standard/v1의 엄격한 interpreter다."""

    schema_version = "report-validation-standard/v1"

    def interpret(self, raw: dict[str, Any]) -> ValidationStandard:
        """기준 ID·정책·criterion 및 required element의 유일성을 검증한다."""
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            raise InputError("usage must be an object")
        result_policy = _text(usage.get("result_policy"), "usage.result_policy")
        mutation_policy = _text(usage.get("mutation_policy"), "usage.mutation_policy")
        if result_policy != _RESULT_POLICY or mutation_policy != _MUTATION_POLICY:
            raise InputError("unsupported usage policy")
        criteria_raw = raw.get("criteria")
        if not isinstance(criteria_raw, list) or not criteria_raw:
            raise InputError("criteria must be a non-empty list")
        criteria: list[CriterionSpec] = []
        seen_criteria: set[str] = set()
        for index, criterion in enumerate(criteria_raw):
            if not isinstance(criterion, dict):
                raise InputError(f"criteria[{index}] must be an object")
            criterion_id = _text(criterion.get("criterion_id"), f"criteria[{index}].criterion_id")
            if criterion_id in seen_criteria:
                raise InputError("duplicate criterion_id")
            seen_criteria.add(criterion_id)
            elements_raw = criterion.get("required_elements")
            if not isinstance(elements_raw, list):
                raise InputError(f"criteria[{index}].required_elements must be a list")
            elements: list[ElementSpec] = []
            seen_elements: set[str] = set()
            for element_index, element in enumerate(elements_raw):
                if not isinstance(element, dict):
                    raise InputError("required element must be an object")
                element_id = _text(element.get("element_id"), f"required_elements[{element_index}].element_id")
                if element_id in seen_elements:
                    raise InputError("duplicate element_id")
                seen_elements.add(element_id)
                description = element.get("description")
                if description is not None and not isinstance(description, str):
                    raise InputError("element description must be a string")
                elements.append(ElementSpec(element_id, description.strip() if isinstance(description, str) else None))
            category = criterion.get("category")
            if category is not None and not isinstance(category, str):
                raise InputError("criterion category must be a string")
            criteria.append(CriterionSpec(criterion_id, _text(criterion.get("description"), "criterion description"), tuple(elements), category))
        return ValidationStandard(
            _text(raw.get("standard_id"), "standard_id"),
            _text(raw.get("standard_version"), "standard_version"),
            _text(raw.get("status"), "status"),
            result_policy,
            mutation_policy,
            tuple(criteria),
        )
