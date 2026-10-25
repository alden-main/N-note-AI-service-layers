"""검증 결과를 외부 브리지 전달 형식으로 제한해 변환하는 서비스이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.draft import ValidationResult
from n_note_ai_service_layers.contracts.form_schema import DraftHandoff, FormSchema, SchemaStatus


class DraftHandoffProjector:
    """활성 스키마의 브리지 매핑에 따라 검증 결과를 핸드오프로 투영한다.

    입력: 검증 결과, 활성 양식 스키마와 간호사 검토 상태이다.
    출력: 매핑된 필드와 검증 정보를 가진 ``DraftHandoff``이다.
    """
    def project(self, result: ValidationResult, schema: FormSchema, nurse_review_status: str) -> DraftHandoff:
        if schema.status is not SchemaStatus.ACTIVE:
            raise PermissionError("only active schema can create handoff")
        mapping = {item.field_id: item.output_key for item in schema.bridge_mapping}
        fields = {mapping[field_id]: value for field_id, value in result.fields.items() if field_id in mapping}
        return DraftHandoff(schema.schema_id, schema.version, fields, {"status": result.status, "issues": [issue.__dict__ for issue in result.issues], "needs_confirmation": result.needs_confirmation}, nurse_review_status)
