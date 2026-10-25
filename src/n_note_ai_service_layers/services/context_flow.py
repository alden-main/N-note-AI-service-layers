"""접근 제어부터 저장소 조회·정규화·문맥 조립까지 연결하는 흐름이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery
from n_note_ai_service_layers.services.access_policy import PurposeAccessPolicy
from n_note_ai_service_layers.services.context_assembler import ClinicalContextAssembler
from n_note_ai_service_layers.services.schema_review import FormSchemaRegistry
from n_note_ai_service_layers.adapters.clinical_normalizer import CodeMappingNormalizer


class ClinicalContextFlow:
    """권한을 먼저 확인한 뒤 임상 문맥을 생성하는 오케스트레이션 경계이다.

    입력: 접근 정책, 스키마 레지스트리, 저장소, 조립기와 선택 정규화기이다.
    출력: ``build`` 호출 시 승인된 요청의 ``ClinicalContextSnapshot``이다.
    """
    def __init__(self, policy: PurposeAccessPolicy, registry: FormSchemaRegistry, repository, assembler: ClinicalContextAssembler, normalizer=None) -> None:
        self.policy, self.registry, self.repository, self.assembler = policy, registry, repository, assembler
        self.normalizer = normalizer or CodeMappingNormalizer()

    def build(self, query: ClinicalContextQuery):
        self.policy.authorize_read(query)
        schema = self.registry.resolve_active(query.schema_id, query.purpose, query.schema_version)
        records = self.repository.fetch(query, schema.requested_fact_ids)
        return self.assembler.build(query, self.normalizer.normalize(records), schema)
