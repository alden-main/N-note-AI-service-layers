"""임상 문맥 조회 목적과 필수 식별자를 검사하는 접근 제어 정책이다."""

from __future__ import annotations

from n_note_ai_service_layers.contracts.clinical_context import ClinicalContextQuery


class AccessDenied(PermissionError):
    """요청 목적 또는 환자·내원 식별자가 허용되지 않을 때 발생한다."""
    pass


class PurposeAccessPolicy:
    """허용된 업무 목적에 대해서만 임상 사실 조회를 승인한다.

    입력: 허용할 업무 목적 문자열 집합이다.
    출력: 조회 요청을 통과시키거나 ``AccessDenied`` 예외를 발생시킨다.
    """
    def __init__(self, allowed_purposes: set[str]) -> None:
        self.allowed_purposes = allowed_purposes

    def authorize_read(self, query: ClinicalContextQuery) -> None:
        if query.purpose not in self.allowed_purposes or not query.patient_id or not query.encounter_id:
            raise AccessDenied("read access is not authorized")
