"""HMD 분석 기준 v1을 범용 검증 기준 IR로 변환한다."""

from __future__ import annotations

from typing import Any

from .contracts import CriterionSpec, ElementSpec, InputError, ValidationStandard
from .standards import _MUTATION_POLICY, _RESULT_POLICY, _text


class HmdAnalysisRequirementSetV1Interpreter:
    """hmd-analysis-requirement-set/v1 전용 interpreter다."""

    schema_version = "hmd-analysis-requirement-set/v1"

    def interpret(self, raw: dict[str, Any]) -> ValidationStandard:
        """보고서 검증에 필요한 requirement/evidence만 공통 IR로 변환한다.

        HMD JSON은 여러 소비자가 공유하는 예시 형식이므로 status, usage,
        evidence_rules와 소유·승인 메타데이터는 이 서비스가 해석하거나 검증하지 않는다.
        """
        requirements = raw.get("analysis_requirements")
        if not isinstance(requirements, list) or not requirements:
            raise InputError("analysis_requirements must be a non-empty list")
        criteria: list[CriterionSpec] = []
        seen_requirements: set[str] = set()
        for requirement in requirements:
            if not isinstance(requirement, dict):
                raise InputError("analysis requirement must be an object")
            requirement_id = _text(requirement.get("requirement_id"), "requirement_id")
            if requirement_id in seen_requirements:
                raise InputError("duplicate requirement_id")
            seen_requirements.add(requirement_id)
            evidence = requirement.get("required_evidence")
            if not isinstance(evidence, list):
                raise InputError("required_evidence must be a list")
            evidence_ids = [_text(item, "required_evidence item") for item in evidence]
            if len(evidence_ids) != len(set(evidence_ids)):
                raise InputError("duplicate required evidence")
            criteria.append(CriterionSpec(
                requirement_id,
                _text(requirement.get("description"), "requirement description"),
                tuple(ElementSpec(element_id) for element_id in evidence_ids),
                requirement.get("category") if isinstance(requirement.get("category"), str) else None,
            ))
        return ValidationStandard(
            _metadata_text(raw.get("requirement_set_id"), "hmd-analysis-requirement-set"),
            _metadata_text(raw.get("requirement_set_version"), "v1"),
            "unclassified",
            _RESULT_POLICY,
            _MUTATION_POLICY,
            tuple(criteria),
        )


def _metadata_text(value: Any, fallback: str) -> str:
    """표시에만 쓰는 HMD 메타데이터를 선택적으로 보존한다."""
    return value.strip() if isinstance(value, str) and value.strip() else fallback
