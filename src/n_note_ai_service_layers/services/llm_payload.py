"""임상 문맥에서 LLM이 허용된 최소 입력 페이로드만 만드는 서비스이다."""

from __future__ import annotations

from datetime import datetime

from n_note_ai_service_layers.contracts.clinical_context import Availability, ClinicalContextSnapshot
from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus
from n_note_ai_service_layers.services.clock import now_local


class LlmPayloadProjector:
    """활성 스키마가 허용한 사실과 전사문만 LLM 입력으로 투영한다.

    입력: 임상 문맥 스냅샷, 전사문과 일치하는 활성 양식 스키마이다.
    출력: 가용 상태와 확인 필요 여부를 포함하는 최소 사전 페이로드이다.
    """
    def project(
        self,
        snapshot: ClinicalContextSnapshot,
        transcript: str,
        schema: FormSchema,
        approved_term_corrections: tuple[dict[str, str], ...] = (),
        term_review_hints: tuple[dict, ...] = (),
        reference_time: datetime | None = None,
    ) -> dict:
        """활성 스키마에 허용된 사실만 전사문과 함께 LLM 입력으로 구성한다.

        입력: ``snapshot``은 확정 임상 사실, ``transcript``는 음성 전사문,
        ``schema``는 스냅샷과 일치하는 활성 양식 스키마이다.
        출력: 사용 가능 사실의 값과 나머지 사실의 상태만 담은 사전이며,
        스키마가 활성화되지 않았거나 일치하지 않으면 ``PermissionError``를 발생시킨다.
        """
        if schema.status is not SchemaStatus.ACTIVE or (snapshot.schema_id, snapshot.schema_version) != (schema.schema_id, schema.version):
            raise PermissionError("only the matching active schema can be projected")
        allowed = {field.fact_id for field in schema.fields if field.llm_allowed and field.fact_id}
        facts = []
        confirmation = False
        for fact in snapshot.facts:
            if fact.fact_id not in allowed:
                continue
            item = {"fact_id": fact.fact_id, "availability": fact.availability.value, "observed_at": fact.observed_at.isoformat() if fact.observed_at else None}
            if fact.availability is Availability.AVAILABLE:
                item["value"] = fact.value
            else:
                confirmation = True
            facts.append(item)
        return {
            "schema_id": schema.schema_id,
            "schema_version": schema.version,
            "form_title": schema.display_title,
            # 구술 시점이다. '오후 2시' 같은 상대 표현의 날짜를 해석하는 기준으로만 쓴다.
            "reference_time": (reference_time or now_local()).isoformat(timespec="minutes"),
            "transcript": transcript,
            # Gateway가 전사문에서 찾은 표준어·의학용어 후보다. 전사문을 대체하지 않는다.
            "term_review_hints": list(term_review_hints),
            "facts": facts,
            "needs_confirmation": confirmation,
            # This is an explicit review artifact, never a replacement transcript.
            "approved_term_corrections": list(approved_term_corrections),
        }
