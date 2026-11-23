"""초안 생성 요청의 접근 검사와 활성 양식 확인을 연결하는 흐름이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.clinical_context import AuthorizedContext, ClinicalContextQuery
from n_note_ai_service_layers.services.access_policy import PurposeAccessPolicy
from n_note_ai_service_layers.services.schema_review import FormSchemaRegistry


class ClinicalContextFlow:
    """권한을 먼저 확인한 뒤 요청에 맞는 활성 양식으로 요청 문맥을 만든다.

    입력: 접근 정책과 스키마 레지스트리다.
    출력: ``build`` 호출 시 승인된 요청의 ``AuthorizedContext``다. 목적이 허용되지 않았거나 환자·내원 ID가 없으면
    ``AccessDenied``(``PermissionError``), 활성 양식이 없거나 요청 버전과 다르면 ``LookupError``다.
    """
    def __init__(self, policy: PurposeAccessPolicy, registry: FormSchemaRegistry) -> None:
        self.policy, self.registry = policy, registry

    def build(self, query: ClinicalContextQuery) -> AuthorizedContext:
        self.policy.authorize_read(query)
        schema = self.registry.resolve_active(query.schema_id, query.purpose, query.schema_version)
        return AuthorizedContext(schema.schema_id, schema.version, query.trace_id)
