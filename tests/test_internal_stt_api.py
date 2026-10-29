import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from n_note_ai_service_layers.api.app import create_app
from n_note_ai_service_layers.api.deps import get_gateway_client, get_speech_to_text_provider, get_transcript_redactor
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor
from n_note_ai_service_layers.contracts.term_verification import GatewayTermOccurrence, GatewaySuggestion, GatewayTermVerificationResult
from n_note_ai_service_layers.api.routes import internal_stt


def setup_review(app, monkeypatch, result=None):
    app.dependency_overrides[get_transcript_redactor] = lambda: TranscriptRedactor()
    app.dependency_overrides[get_gateway_client] = lambda: object()
    monkeypatch.setattr(internal_stt, "GatewayTermVerificationFlow", lambda gateway: SimpleNamespace(
        verify=lambda *args: result or GatewayTermVerificationResult("transcript", "ok")))


def test_internal_stt_multipart_and_output_validation(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text="고혈압이 있습니다."))
    monkeypatch.setattr(internal_stt, "_extract", lambda transcript, template, hints: {
        "fields": {"history": "고혈압", "choice": {"status": "없음", "detail": "잘못된 상세"}, "allergy": ["약", "허위"]},
        "relatedSentences": {"history": ["고혈압이 있습니다.", "없는 문장"]},
    })
    template = {"templateType": "CUSTOM", "templateName": "서식", "schemaDefinition": {"sections": [{"fields": [
        {"field_id": "history", "field_type": "text"},
        {"field_id": "choice", "field_type": "radio", "options": ["없음", "있음"], "has_detail_text": True},
        {"field_id": "allergy", "field_type": "checkbox", "options": ["약"]},
    ]}]}}
    client = TestClient(app)
    response = client.post("/internal/v1", files={"file": ("test.aac", b"audio", "audio/aac"), "templates": ("templates.json", json.dumps([template]), "application/json")})
    assert response.status_code == 200
    assert response.json() == {"transcript": "고혈압이 있습니다.", "forms": [{"templateType": "CUSTOM", "templateName": "서식", "fields": {"history": "고혈압", "choice": {"status": "없음", "detail": ""}, "allergy": ["약"]}, "relatedSentences": {"history": ["고혈압이 있습니다."]}}]}


def test_internal_stt_extraction_failure_preserves_transcript(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text="전사문"))
    def fail(*args):
        raise RuntimeError()
    monkeypatch.setattr(internal_stt, "_extract", fail)
    template = {"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": []}}
    response = TestClient(app).post("/internal/v1", files={"file": ("test.wav", b"audio", "audio/wav")}, data={"templates": json.dumps([template])})
    assert response.json() == {"transcript": "전사문", "forms": [{"templateType": "X", "templateName": "Y", "fields": {}, "relatedSentences": {}}]}
    assert TestClient(app).post("/internal/v1", files={"file": ("bad.txt", b"x")}, data={"templates": "[]"}).status_code == 415


def test_redaction_and_term_hints_precede_extraction(monkeypatch):
    app = create_app()
    occurrence = GatewayTermOccurrence("transcript", "medical", 0, 3, "고혈압", "고혈압", (GatewaySuggestion("고혈압증"),), category="medical_term")
    setup_review(app, monkeypatch, GatewayTermVerificationResult("transcript", "ok", (occurrence,)))
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text="홍길동님 고혈압"))
    seen = {}
    def extract(transcript, template, hints):
        seen.update(transcript=transcript, hints=hints)
        return {"fields": {"history": "고혈압증"}, "relatedSentences": {"history": [transcript, "홍길동님 고혈압"]}}
    monkeypatch.setattr(internal_stt, "_extract", extract)
    template = {"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": [{"fields": [{"field_id": "history", "field_type": "text"}]}]}}
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([template])})
    assert response.status_code == 200
    body = response.json()
    assert "홍길동" not in body["transcript"] and seen["transcript"] == body["transcript"]
    assert seen["hints"][0]["suggestions"] == ["고혈압증"]
    assert body["forms"][0]["relatedSentences"]["history"] == [body["transcript"]]


def test_redaction_or_gateway_failure_stops_extraction(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch, GatewayTermVerificationResult("transcript", "unavailable"))
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text="홍길동님"))
    def fail(*args):
        raise AssertionError("extraction must not run")
    monkeypatch.setattr(internal_stt, "_extract", fail)
    template = [{"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": []}}]
    def send():
        return TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(template)})
    assert send().status_code == 503
    class BrokenRedactor:
        def redact(self, text):
            raise ValueError("private text")
    app.dependency_overrides[get_transcript_redactor] = lambda: BrokenRedactor()
    response = send()
    assert response.status_code == 503 and "private text" not in response.text
