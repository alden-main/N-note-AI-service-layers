"""bridge가 음성과 함께 보내는 EMR JSON(선택)의 검증·식별 정보 제거·평탄화·비식별화와 초안 흐름 연결을 검증한다."""

from __future__ import annotations

import json
import threading
import unittest

from fastapi.testclient import TestClient

from n_note_ai_service_layers.api import app, get_nursing_note_workflow, get_services, get_workflow_context_flow
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.services.emr_context import EmrInputError, flatten, parse_emr, prepare_emr
from n_note_ai_service_layers.services.llm_payload import LlmPayloadProjector
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor
from support import SCORE_FIELD
from test_multi_form_workflow import BASE, _Context, _Gateway, _job_result, _Stt

# 사용자가 준 다른 프로젝트의 합성 EMR과 같은 구조다(중첩 객체 + 측정 기록 배열 + null).
EMR = {
    "patient": {
        "patientId": "pid-4356", "patientName": "홍길동", "startDate": "2026-09-16 11:17:30",
        "doctorId": "doc-5567", "doctorName": "임꺽정", "admissionDate": "2026-09-01", "dischargeDate": None,
    },
    "bloodPressureList": [
        {"measuredAt": "2026-09-01 17:31:25", "sbp": 120, "dbp": 90},
        {"measuredAt": "2026-09-03 08:14:33", "sbp": None, "dbp": 83},
    ],
    "bloodSugarList": [{"measuredAt": "2026-09-02 09:34:42", "timingType": "식후", "glucoseValue": 200}],
    "labResultList": [{"examCode": "HbA1c", "itemName": "당화혈색소", "value": "6.8", "testDate": "2026-09-04 09:00:00"}],
    "medicineDtailList": [{"medCode": "med-011", "medName": "약이름1", "dosage": 15, "frequency": 3, "instructions": "홍길동님 하루 3번 15mg 투약하세요"}],
}


class ParseTests(unittest.TestCase):
    def test_object_is_accepted_and_empty_means_no_emr(self) -> None:
        self.assertEqual(parse_emr(json.dumps(EMR, ensure_ascii=False)), EMR)
        self.assertEqual(parse_emr(EMR), EMR)
        for empty in (None, "", "  ", "{}", {}):
            self.assertIsNone(parse_emr(empty))

    def test_limits_and_shape_are_enforced(self) -> None:
        cases = {
            "not json": "{",
            "not an object": "[1, 2]",
            "too large": json.dumps({"a": "x" * 100}),
            "too deep": json.dumps({"a": {"b": {"c": {"d": 1}}}}),
            "too many values": json.dumps({"a": list(range(20))}),
        }
        for name, raw in cases.items():
            with self.subTest(case=name), self.assertRaises(EmrInputError):
                parse_emr(raw, max_bytes=80, max_entries=10, max_depth=3)


class PrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.found = prepare_emr(EMR, TranscriptRedactor(), ("P1", "E1"))
        self.lines = {entry.key: entry.value for entry in self.context.entries}

    def test_identifier_keys_are_removed_everywhere_and_their_values_never_leave(self) -> None:
        text = json.dumps(self.context.payload(), ensure_ascii=False)
        for value in ("pid-4356", "홍길동", "doc-5567", "임꺽정"):
            self.assertNotIn(value, text)
        self.assertEqual(self.context.removed_identifier_keys, 4)
        self.assertEqual(set(self.found.names), {"홍길동", "임꺽정"})

    def test_records_keep_measurements_with_their_time_and_skip_nulls(self) -> None:
        self.assertEqual(self.lines["bloodPressureList[0]"], "measuredAt=2026-09-01 17:31:25, sbp=120, dbp=90")
        self.assertEqual(self.lines["bloodPressureList[1]"], "measuredAt=2026-09-03 08:14:33, dbp=83")
        self.assertNotIn("patient.dischargeDate", self.lines)
        self.assertEqual(self.lines["patient.admissionDate"], "2026-09-01")

    def test_clinical_names_are_kept_and_names_inside_values_are_masked(self) -> None:
        medication = self.lines["medicineDtailList[0]"]
        self.assertIn("medName=약이름1", medication)  # 약 이름은 식별 정보가 아니다
        self.assertIn("[[PERSON_1]]님", medication)
        self.assertEqual(self.context.replacements, 1)

    def test_emr_names_are_masked_in_the_transcript_even_when_short(self) -> None:
        redacted = TranscriptRedactor().redact("홍길동 환자 혈압 120/90, 임꺽정 선생님께 보고함", self.found.identifiers, self.found.names)
        self.assertNotIn("홍길동", redacted.text)
        self.assertNotIn("임꺽정", redacted.text)

    def test_flatten_handles_scalar_arrays_and_nested_lists(self) -> None:
        self.assertEqual(flatten({"allergies": ["페니실린", None, "조영제"], "a": {"b": [{"c": [1, 2]}]}}),
                         [("allergies", "페니실린, 조영제"), ("a.b[0].c", "1, 2")])

    def test_custom_identifier_keys_replace_the_defaults(self) -> None:
        context, _ = prepare_emr({"wardName": "7병동", "bed": "3"}, TranscriptRedactor(), identifier_keys=("ward_name",))
        self.assertEqual([entry.key for entry in context.entries], ["bed"])


class _Generator:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, payload, schema):
        with self._lock:
            self.payloads.append(payload)
        # 전사문에는 없고 EMR에만 있는 수치(200, 식후 혈당)를 쓴다. EMR이 왔으면 출처 있는 수치다.
        return DraftResult({SCORE_FIELD: 2, "assessment.mobility_score": 200})


class _Projector(LlmPayloadProjector):
    def project(self, snapshot, transcript, schema, term_review_hints=(), reference_time=None, emr=None):
        return {"transcript": transcript, **({"emr_context": emr.payload()} if emr else {})}


class EmrWorkflowApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_workflow_context_flow] = _Context
        self.generator = _Generator()
        workflow = NursingNoteWorkflow(_Stt(), self.generator, _Gateway(), projector=_Projector())
        app.dependency_overrides[get_nursing_note_workflow] = lambda: workflow

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def _draft(self, job) -> dict:
        """초안 조회 API와 같은 응답이다(테스트 양식은 운영 레지스트리에 없으므로 저장소·매퍼로 만든다)."""
        from n_note_ai_service_layers.api import mappers
        from test_multi_form_workflow import SCHEMAS

        store = get_services().draft_review_store
        record = store.get(job["result"]["draft_id"])
        schema = SCHEMAS[record.schema_id]
        return mappers.draft_review_response(record, schema, store.numeric_findings(record, schema))

    def _post(self, **data):
        return self.client.post(
            "/v1/nursing-note-workflows",
            files={"audio": ("note.wav", b"audio", "audio/wav")},
            data={"patient_id": "pid-4356", "encounter_id": "enc-0001", "purpose": BASE.purpose, "schema_id": "form-a", **data},
        )

    def test_emr_reaches_the_llm_payload_deidentified_and_is_kept_on_the_draft(self) -> None:
        job = _job_result(self.client, self._post(emr=json.dumps(EMR, ensure_ascii=False)))
        self.assertEqual(job["status"], "succeeded", job)
        (payload,) = self.generator.payloads
        sent = json.dumps(payload, ensure_ascii=False)
        self.assertIn("bloodSugarList[0]", sent)
        self.assertNotIn("홍길동", sent)
        self.assertNotIn("pid-4356", sent)
        # 원본 EMR은 작업 문서에 남기지 않는다.
        self.assertNotIn("bloodSugarList", json.dumps(get_services().jobs.get(job["job_id"]), ensure_ascii=False))
        draft = self._draft(job)
        self.assertIn({"key": "bloodSugarList[0]", "value": "measuredAt=2026-09-02 09:34:42, timingType=식후, glucoseValue=200"}, draft["emr_context"])
        self.assertEqual(draft["numeric_verification"]["status"], "ok", draft["numeric_verification"])

    def test_without_emr_the_payload_is_unchanged(self) -> None:
        job = _job_result(self.client, self._post())
        self.assertEqual(job["status"], "succeeded", job)
        self.assertNotIn("emr_context", self.generator.payloads[0])
        draft = self._draft(job)
        self.assertEqual(draft["emr_context"], [])
        # EMR이 없으면 같은 200은 전사문에 없는 수치로 잡힌다(위 테스트와 대비).
        self.assertEqual(draft["numeric_verification"]["status"], "needs_review")

    def test_invalid_emr_is_rejected_before_a_job_is_created(self) -> None:
        for raw in ("{", "[1]"):
            with self.subTest(raw=raw):
                response = self._post(emr=raw)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"], "emr must be a JSON object")
        self.assertEqual(self.generator.payloads, [])


if __name__ == "__main__":
    unittest.main()


class EmrOtherEntryPointTests(unittest.TestCase):
    """녹음 세션 종료와 동기 서식 추출(/internal/v1)도 같은 EMR을 받는다."""

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_recording_session_finish_passes_emr_to_the_draft(self) -> None:
        from n_note_ai_service_layers.api import get_speech_to_text_provider
        from test_recording_sessions import _ChunkStt

        client = TestClient(app)
        generator = _Generator()
        workflow = NursingNoteWorkflow(_Stt(), generator, _Gateway(), projector=_Projector())
        app.dependency_overrides[get_speech_to_text_provider] = lambda: _ChunkStt()
        app.dependency_overrides[get_workflow_context_flow] = _Context
        app.dependency_overrides[get_nursing_note_workflow] = lambda: workflow
        session_id = client.post("/v1/recording-sessions", json={"patient_id": "pid-4356", "encounter_id": "enc-0001"}).json()["session_id"]
        chunk = client.post(f"/v1/recording-sessions/{session_id}/chunks", files={"audio": ("c.webm", "홍길동 환자 통증 2점".encode(), "audio/webm")}, data={"sequence": "0"})
        self.assertEqual(chunk.status_code, 200, chunk.text)
        body = {"purpose": BASE.purpose, "schema_id": "form-a"}
        self.assertEqual(client.post(f"/v1/recording-sessions/{session_id}/workflows", json={**body, "emr": [1]}).status_code, 422)
        accepted = client.post(f"/v1/recording-sessions/{session_id}/workflows", json={**body, "emr": EMR})
        job = _job_result(client, accepted)
        self.assertEqual(job["status"], "succeeded", job)
        (payload,) = generator.payloads
        self.assertIn("emr_context", payload)
        # EMR의 환자 이름은 전사문에서도 가린다.
        self.assertNotIn("홍길동", json.dumps(payload, ensure_ascii=False))
