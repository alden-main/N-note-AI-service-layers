"""요청 문맥에서 LLM에 허용된 최소 입력 페이로드만 만드는 서비스이다."""

from __future__ import annotations

from datetime import datetime

from n_note_ai_service_layers.contracts.clinical_context import AuthorizedContext
from n_note_ai_service_layers.contracts.form_schema import FormSchema, SchemaStatus
from n_note_ai_service_layers.services.clock import now_local
from n_note_ai_service_layers.services.emr_context import EmrContext


class LlmPayloadProjector:
    """활성 양식에 맞춰 전사문·EMR 참고 문맥·용어 힌트만 LLM 입력으로 투영한다.

    입력: 승인된 요청 문맥, 비식별 전사문, 문맥과 같은 활성 양식 스키마다.
    출력: LLM이 쓰는 값만 담은 사전이다(빈 항목은 넣지 않는다).
    """

    def project(
        self,
        context: AuthorizedContext,
        transcript: str,
        schema: FormSchema,
        term_review_hints: tuple[dict, ...] = (),
        reference_time: datetime | None = None,
        emr: EmrContext | None = None,
    ) -> dict:
        """문맥과 같은 활성 양식일 때만 LLM 입력을 만든다. 아니면 ``PermissionError``다."""
        if schema.status is not SchemaStatus.ACTIVE or (context.schema_id, context.schema_version) != (schema.schema_id, schema.version):
            raise PermissionError("only the matching active schema can be projected")
        # LLM이 쓰지 않는 식별자(schema_id·version)와 빈 목록은 넣지 않는다. 토큰만 쓰고 모델이 값을 따라 쓸 수 있다.
        payload: dict = {
            "form_title": schema.display_title,
            # 구술 시점이다. '오후 2시' 같은 상대 표현의 날짜를 해석하는 기준으로만 쓴다.
            "reference_time": (reference_time or now_local()).isoformat(timespec="minutes"),
            "transcript": transcript,
        }
        if emr is not None and emr.entries:
            # bridge가 보낸 EMR(식별 정보 제거·비식별화 후)이다. 필드 매칭 없이 전체를 참고 문맥으로 준다.
            payload["emr_context"] = emr.payload()
        if term_review_hints:
            # Gateway가 전사문에서 찾은 표준어·의학용어 후보다. 전사문을 대체하지 않는다.
            payload["term_review_hints"] = list(term_review_hints)
        return payload
