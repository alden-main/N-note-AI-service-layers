import json
from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

from n_note_ai_service_layers import settings
from n_note_ai_service_layers.adapters import model_providers
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


FORM = {"templateType": "HAND_OVER", "templateName": "인계", "schemaDefinition": {"sections": [{"section_id": "sbar", "fields": [
    {"field_id": "situation", "label": "Situation", "field_type": "text", "options": None},
    {"field_id": "smoking", "label": "흡연력", "sub_label": "현재", "field_type": "radio", "has_detail_text": True, "options": ["nonsmoker", "ex-smoker"]},
    {"field_id": "age", "field_type": "radio", "options": ["성인"]},
    {"field_id": "allergy", "field_type": "checkbox", "has_detail_text": True, "options": ["없음", "약"]},
    {"field_id": "tags", "field_type": "checkbox", "options": []},
]}]}}
VALID = {
    "fields": {"situation": "흑색변으로 입원함", "smoking": {"status": "nonsmoker", "detail": ""}, "age": "성인", "allergy": {"selected": ["약"], "detail": "두드러기"}, "tags": []},
    "relatedSentences": {"situation": ["흑색변으로 입원했습니다."], "smoking": [], "age": [], "allergy": [], "tags": []},
}


def chat_client(monkeypatch, content):
    monkeypatch.setenv("N_NOTE_LLM_API", "chat")
    monkeypatch.setenv("N_NOTE_LLM_BASE_URL", "http://local-llm/v1")
    calls = []
    def create(**request):
        calls.append(request)
        return {"choices": [{"message": {"content": content}}]}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(internal_stt.model_providers, "openai_client", lambda *args: client)
    return calls


def test_output_schema_is_flat_and_matches_response_contract():
    schema = internal_stt._output_schema(FORM)
    fields = schema["properties"]["fields"]
    assert fields["required"] == ["situation", "smoking", "age", "allergy", "tags"] and fields["additionalProperties"] is False
    assert fields["properties"]["smoking"]["description"] == "흡연력 - 현재"
    assert fields["properties"]["smoking"]["properties"]["status"]["enum"] == ["nonsmoker", "ex-smoker", ""]
    assert fields["properties"]["allergy"]["required"] == ["selected", "detail"]
    assert fields["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["relatedSentences"]["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert internal_stt.Draft202012Validator(schema).is_valid(VALID)


def test_chat_extraction_requests_structured_output_with_correction_rules(monkeypatch):
    calls = chat_client(monkeypatch, json.dumps(VALID, ensure_ascii=False))
    assert internal_stt._extract("전사문", FORM, ()) == VALID
    request = calls[0]
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == internal_stt._output_schema(FORM)
    assert "standard Korean" in request["messages"][0]["content"] and "JSON schema:" in request["messages"][0]["content"]


def test_responses_extraction_requests_structured_output(monkeypatch):
    monkeypatch.setenv("N_NOTE_LLM_API", "responses")
    monkeypatch.delenv("N_NOTE_LLM_BASE_URL", raising=False)
    seen = {}
    def create(**request):
        seen.update(request)
        return SimpleNamespace(output_text=json.dumps(VALID, ensure_ascii=False))
    monkeypatch.setattr(internal_stt.model_providers, "openai_client", lambda *args: SimpleNamespace(responses=SimpleNamespace(create=create)))
    assert internal_stt._extract("전사문", FORM, ()) == VALID
    assert seen["text"]["format"]["strict"] is True and seen["text"]["format"]["schema"] == internal_stt._output_schema(FORM)


NESTED = {"fields": {"sbar": VALID["fields"]}, "relatedSentences": VALID["relatedSentences"]}
ARRAY = {"fields": [{"field_id": "situation", "value": "입원"}], "relatedSentences": VALID["relatedSentences"]}
WRAPPED_TEXT = {"fields": {**VALID["fields"], "situation": {"value": "입원"}}, "relatedSentences": VALID["relatedSentences"]}
NO_EVIDENCE = {"fields": VALID["fields"]}


def test_extraction_rejects_structure_that_would_silently_empty_fields(monkeypatch):
    for result in (NESTED, ARRAY, WRAPPED_TEXT, NO_EVIDENCE, []):
        chat_client(monkeypatch, json.dumps(result, ensure_ascii=False))
        try:
            internal_stt._extract("전사문", FORM, ())
        except ValueError:
            continue
        raise AssertionError(f"accepted {result}")


def test_out_of_option_value_stays_a_blank_field_not_a_failed_form(monkeypatch):
    result = {**VALID, "fields": {**VALID["fields"], "age": "노인", "smoking": {"status": "비흡연", "detail": ""}}}
    chat_client(monkeypatch, json.dumps(result, ensure_ascii=False))
    normalized = internal_stt._normalize(FORM, "전사문", internal_stt._extract("전사문", FORM, ()))
    assert normalized["fields"]["age"] == "" and normalized["fields"]["smoking"] == {"status": "", "detail": ""}
    assert normalized["fields"]["situation"] == "흑색변으로 입원함"


def test_malformed_extraction_returns_empty_form_and_logs_failure(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text="흑색변으로 입원했습니다."))
    chat_client(monkeypatch, json.dumps(NESTED, ensure_ascii=False))
    logs = []
    monkeypatch.setattr(internal_stt, "stage", lambda *args, **kwargs: observed_stage(logs, *args, **kwargs))
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([FORM])})
    assert response.status_code == 200
    assert response.json()["forms"] == [{"templateType": "HAND_OVER", "templateName": "인계", "fields": {}, "relatedSentences": {}}]
    assert logs == [("transcript_term_review", {"status": "ok", "candidates": 0}), ("form_extraction", "ValueError")]


def test_extraction_metrics_count_filled_fields_only(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text="흑색변으로 입원했습니다."))
    chat_client(monkeypatch, json.dumps(VALID, ensure_ascii=False))
    logs = []
    monkeypatch.setattr(internal_stt, "stage", lambda *args, **kwargs: observed_stage(logs, *args, **kwargs))
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([FORM])})
    assert response.status_code == 200
    assert logs[-1] == ("form_extraction", {"fields": 5, "filled": 4, "with_evidence": 1})


@contextmanager
def observed_stage(logs, logger, trace_id, name, **fields):
    metrics = {}
    try:
        yield metrics
    except Exception as exc:
        logs.append((name, type(exc).__name__))
        raise
    logs.append((name, metrics))


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


def test_stt_uses_nursing_context_prompt_when_setting_is_blank(monkeypatch):
    monkeypatch.setenv("N_NOTE_STT_PROMPT", "")
    monkeypatch.setenv("N_NOTE_STT_BASE_URL", "http://local-stt/v1")
    monkeypatch.setattr(model_providers, "openai_client", lambda *args: object())
    assert model_providers.speech_to_text_provider()._prompt == settings.DEFAULT_STT_PROMPT
    monkeypatch.setenv("N_NOTE_STT_PROMPT", "병동 문맥")
    assert model_providers.speech_to_text_provider()._prompt == "병동 문맥"


def test_term_review_status_is_logged_when_it_stops_extraction(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch, GatewayTermVerificationResult("transcript", "unsupported"))
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text="전사문"))
    logs = []
    monkeypatch.setattr(internal_stt, "stage", lambda *args, **kwargs: observed_stage(logs, *args, **kwargs))
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([FORM])})
    assert response.status_code == 503 and response.json() == {"detail": "term review is unavailable"}
    assert logs == [("transcript_term_review", {"status": "unsupported", "candidates": 0})]
