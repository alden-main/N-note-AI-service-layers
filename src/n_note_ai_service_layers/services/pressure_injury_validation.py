"""브레이든 점수에 따라 욕창 위험도와 예방 중재를 결정적으로 검증한다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.clinical_context import Availability, ClinicalContextSnapshot
from n_note_ai_service_layers.contracts.draft import DraftResult, ValidationIssue, ValidationResult
from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus
from n_note_ai_service_layers.policies.pressure_injury import BRADEN_FACT_IDS, INTERVENTIONS, RISK_LEVELS


class PressureInjuryValidator:
    """초안과 임상 사실로 욕창 위험도·예방 중재를 확정한다.

    입력: AI 초안, 임상 문맥 스냅샷과 활성 욕창 양식 스키마이다.
    출력: 점수·위험도·중재 또는 확인 필요 문제를 담은 ``ValidationResult``이다.
    """
    def validate(self, draft: DraftResult, snapshot: ClinicalContextSnapshot, schema: FormSchema) -> ValidationResult:
        """브레이든 사실을 검사해 위험도와 예방 중재를 계산한다.

        입력: ``draft``는 AI 제안값, ``snapshot``은 임상 사실,
        ``schema``는 활성화된 욕창 평가 양식이다.
        출력: 필수 사실이 없거나 충돌하면 확인 필요 결과를, 그렇지 않으면 계산된 검증 결과를 반환한다.
        """
        if schema.status is not SchemaStatus.ACTIVE:
            raise PermissionError("only active schema can be validated")
        by_id = {fact.fact_id: fact for fact in snapshot.facts}
        issues: list[ValidationIssue] = []
        scores: dict[str, int] = {}
        for fact_id in BRADEN_FACT_IDS:
            fact = by_id.get(fact_id)
            if not fact or fact.availability is not Availability.AVAILABLE or fact.value is None:
                code = "CONFLICTING_VALUES" if fact and fact.availability is Availability.CONFLICT else "REQUIRED_FIELD_MISSING"
                issues.append(ValidationIssue(fact_id, code, "confirmation is required before calculating risk"))
            else:
                scores[fact_id] = int(fact.value)
        fields = dict(draft.proposed_fields)
        fields.update(scores)
        if issues:
            status = "error" if any(item.code == "CONFLICTING_VALUES" for item in issues) else "confirmation_required"
            fields.pop("risk.total_score", None); fields.pop("risk.level", None)
            return ValidationResult(status, fields, tuple(issues), True)
        total = sum(scores.values())
        level = next((label for low, high, label in RISK_LEVELS if low <= total <= high), None)
        if level is None:
            return ValidationResult("error", fields, (ValidationIssue("risk.total_score", "OUT_OF_RANGE", "score is outside POC range"),), True)
        fields["risk.total_score"] = total
        fields["risk.level"] = level
        fields["prevention.interventions"] = list(INTERVENTIONS[level])
        return ValidationResult("valid", fields, (), draft.needs_confirmation)
