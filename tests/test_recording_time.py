"""녹음 시각(요청 값 → 음성 메타데이터 → 접수 시각)과 기록 시각 항목 기본값을 검증한다."""

from __future__ import annotations

import io
import time
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from n_note_ai_service_layers.adapters.audio_preprocessing import audio_creation_time, encode_wav
from n_note_ai_service_layers.api import app, get_nursing_note_workflow, get_services, get_workflow_context_flow
from n_note_ai_service_layers.contracts.audio import PreTranscribedAudio, Transcript
from n_note_ai_service_layers.contracts.draft import DraftResult
from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.contracts.term_verification import GatewayTermVerificationResult
from n_note_ai_service_layers.services.nursing_note_workflow import NursingNoteWorkflow, NursingNoteWorkflowResult, fill_recording_time
from n_note_ai_service_layers.services.recording_sessions import RecordingChunk
from n_note_ai_service_layers.services.recording_time import format_for_field, parse_recorded_at, resolve_recording_time

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 11, 18, 9, 30, tzinfo=KST)

SCHEMA = FormSchema(
    "rt-form", "rt-form/v1", "RT",
    (
        FormSection("meta", (FormField("recorded_at", "datetime", default_from="recording_time"),)),
        FormSection("note", (FormField("assessment", "text", required=True),)),
    ),
    status=SchemaStatus.ACTIVE,
)


def _m4a(creation_time: str | None) -> bytes:
    import av

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="ipod") as output:
        if creation_time:
            output.metadata["creation_time"] = creation_time
        stream = output.add_stream("aac", rate=16000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype=np.float32), format="fltp", layout="mono")
        frame.sample_rate = 16000
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            output.mux(packet)
    return buffer.getvalue()


class ResolveRecordingTimeTests(unittest.TestCase):
    def test_priority_is_request_then_metadata_then_received(self) -> None:
        requested = datetime(2026, 11, 18, 8, 0, tzinfo=KST)
        self.assertEqual(resolve_recording_time(requested, "2026-11-17T23:05:00Z", NOW).source, "request")
        metadata = resolve_recording_time(None, "2026-11-17T23:05:00.000000Z", NOW)
        self.assertEqual((metadata.source, metadata.value), ("audio_metadata", datetime(2026, 11, 18, 8, 5, tzinfo=KST)))
        self.assertEqual(resolve_recording_time(None, None, NOW), resolve_recording_time(None, "", NOW))
        self.assertEqual(resolve_recording_time(None, None, NOW).source, "received_at")

    def test_implausible_metadata_falls_back_to_received_time(self) -> None:
        # mp4 기본값(1904년), 미래 시각, 읽을 수 없는 값은 쓰지 않는다.
        for value in ("1904-01-01T00:00:00Z", "1970-01-01T00:00:00Z", "2026-11-19T09:00:00+09:00", "not-a-date"):
            with self.subTest(value=value):
                result = resolve_recording_time(None, value, NOW)
                self.assertEqual((result.source, result.value), ("received_at", NOW))

    def test_parse_uses_ward_timezone_for_naive_values(self) -> None:
        self.assertEqual(parse_recorded_at("2026-11-18T09:00:00"), datetime(2026, 11, 18, 9, 0, tzinfo=KST))
        self.assertEqual(parse_recorded_at("2026-11-18T00:00:00Z"), datetime(2026, 11, 18, 9, 0, tzinfo=KST))
        with self.assertRaises(ValueError):
            parse_recorded_at("어제 오후")

    def test_field_format_matches_llm_format(self) -> None:
        value = datetime(2026, 11, 18, 0, 7, tzinfo=timezone.utc)
        self.assertEqual(format_for_field(value, "datetime"), "2026-11-18 09:07")
        self.assertEqual(format_for_field(value, "date"), "2026-11-18")


class AudioMetadataTests(unittest.TestCase):
    def test_reads_creation_time_from_m4a(self) -> None:
        self.assertEqual(audio_creation_time(_m4a("2026-11-18T00:05:00.000000Z")), "2026-11-18T00:05:00.000000Z")

    def test_returns_none_without_metadata_or_for_undecodable_bytes(self) -> None:
        self.assertIsNone(audio_creation_time(encode_wav(np.zeros(1600, dtype=np.int16))))
        self.assertIsNone(audio_creation_time(b"not audio"))


class FillRecordingTimeTests(unittest.TestCase):
    def test_fills_empty_field_and_marks_evidence(self) -> None:
        draft = DraftResult({"assessment": "통증 호소함."}, {"assessment": ("transcript",)})
        filled = fill_recording_time(draft, SCHEMA, NOW)
        self.assertEqual(filled.proposed_fields, {"assessment": "통증 호소함.", "recorded_at": "2026-11-18 09:30"})
        self.assertEqual(filled.evidence_refs["recorded_at"], ("recording_time",))

    def test_keeps_time_the_llm_read_from_the_transcript(self) -> None:
        draft = DraftResult({"recorded_at": "2026-11-18 14:00", "assessment": "통증 호소함."})
        self.assertIs(fill_recording_time(draft, SCHEMA, NOW), draft)

    def test_default_from_requires_date_or_datetime_field(self) -> None:
        with self.assertRaises(ValidationError):
            FormField("note", "text", default_from="recording_time")

    def test_daily_note_form_opts_in(self) -> None:
        schema = get_services().template_registry.resolve_active("nursing-daily-note", "NURSING_DAILY_NOTE")
        field = next(item for item in schema.fields if item.field_id == "recorded_at")
        self.assertEqual(field.default_from, "recording_time")


class WorkflowRecordingTimeTests(unittest.TestCase):
    def test_recorded_at_is_reference_time_and_fills_record_time(self) -> None:
        class Generator:
            def generate(self, payload, schema):
                self.payload = payload
                return DraftResult({"assessment": "통증 호소함."})

        class Gateway:
            class Client:
                def discover(self, request_id): return ()
            gateway = Client()
            def verify(self, phase, text, trace_id=None, catalog=None): return GatewayTermVerificationResult(phase, "ok")
            def verify_document(self, phase, document, trace_id=None, catalog=None): return GatewayTermVerificationResult(phase, "ok")

        class Projector:
            def project(self, snapshot, transcript, schema, term_review_hints=(), reference_time=None):
                return {"transcript": transcript, "reference_time": reference_time.isoformat(timespec="minutes")}

        generator = Generator()
        recorded = datetime(2026, 11, 18, 7, 45, tzinfo=KST)
        audio = PreTranscribedAudio(Transcript("통증을 호소합니다.", "fake", "fake"), "trace", recorded_at=recorded)
        result = NursingNoteWorkflow(None, generator, Gateway(), projector=Projector()).generate(object(), audio, SCHEMA)
        self.assertEqual(result.reference_time, recorded)
        self.assertTrue(generator.payload["reference_time"].startswith("2026-11-18T07:45"))
        self.assertEqual(result.draft.proposed_fields["recorded_at"], "2026-11-18 07:45")


class _CapturingWorkflow:
    """API가 워크플로에 넘긴 음성 입력을 기록한다."""

    audio = None

    def generate(self, snapshot, audio, selected_schema, known_identifiers=(), progress=None):
        _CapturingWorkflow.audio = audio
        return NursingNoteWorkflowResult(
            Transcript("통증 호소", "fake", "fake-stt", "ko"), DraftResult({"assessment": "통증 호소함."}),
            GatewayTermVerificationResult("transcript", "ok"), GatewayTermVerificationResult("draft", "ok"),
        )


class _Context:
    class Registry:
        def resolve_active(self, schema_id, purpose, version): return SCHEMA
    registry = Registry()
    def build(self, query): return object()


class RecordingTimeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        _CapturingWorkflow.audio = None
        app.dependency_overrides[get_workflow_context_flow] = _Context
        app.dependency_overrides[get_nursing_note_workflow] = _CapturingWorkflow

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def _post(self, content: bytes, filename: str, media_type: str, **data: str):
        form = {"patient_id": "p", "encounter_id": "e", "purpose": SCHEMA.purpose, "schema_id": SCHEMA.schema_id, **data}
        return self.client.post("/v1/nursing-note-workflows", files={"audio": (filename, content, media_type)}, data=form)

    def _recorded_at(self, response) -> datetime:
        self.assertEqual(response.status_code, 202, response.text)
        for _ in range(500):
            if self.client.get(response.headers["Location"]).json()["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.01)
        return _CapturingWorkflow.audio.recorded_at

    def test_request_value_wins(self) -> None:
        value = self._recorded_at(self._post(_m4a("2026-11-17T23:05:00Z"), "a.m4a", "audio/mp4", recorded_at="2026-11-18T08:00:00+09:00"))
        self.assertEqual(value, datetime(2026, 11, 18, 8, 0, tzinfo=KST))

    def test_audio_metadata_is_used_without_request_value(self) -> None:
        value = self._recorded_at(self._post(_m4a("2026-11-17T23:05:00Z"), "a.m4a", "audio/mp4"))
        self.assertEqual(value, datetime(2026, 11, 18, 8, 5, tzinfo=KST))

    def test_received_time_is_used_when_audio_has_no_metadata(self) -> None:
        before = datetime.now(timezone.utc)
        value = self._recorded_at(self._post(b"audio", "a.wav", "audio/wav"))
        self.assertLessEqual(before - timedelta(seconds=1), value)
        self.assertLessEqual(value, datetime.now(timezone.utc))

    def test_invalid_request_value_is_rejected(self) -> None:
        self.assertEqual(self._post(b"audio", "a.wav", "audio/wav", recorded_at="어제").status_code, 422)

    def test_recording_session_uses_its_start_time(self) -> None:
        created = self.client.post("/v1/recording-sessions", json={"patient_id": "p", "encounter_id": "e"})
        self.assertEqual(created.status_code, 201, created.text)
        session = get_services().recording_sessions.get(created.json()["session_id"])
        self.assertIsNotNone(created.json()["started_at"])
        session.chunks.append(RecordingChunk(0, "통증 호소", "통증 호소", 0, "fake", "fake", "ko"))
        accepted = self.client.post(f"/v1/recording-sessions/{session.session_id}/workflows", json={"purpose": SCHEMA.purpose, "schema_id": SCHEMA.schema_id})
        self.assertEqual(self._recorded_at(accepted), session.started_at)
        self.assertIsInstance(_CapturingWorkflow.audio, PreTranscribedAudio)


if __name__ == "__main__":
    unittest.main()
