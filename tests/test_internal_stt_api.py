import json
from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

from n_note_ai_service_layers import prompts
from n_note_ai_service_layers.adapters import model_providers
from n_note_ai_service_layers.api.app import create_app
from n_note_ai_service_layers.api.deps import get_gateway_client, get_speech_to_text_provider, get_transcript_redactor
from n_note_ai_service_layers.services.transcript_redaction import TranscriptRedactor
from n_note_ai_service_layers.contracts.term_verification import GatewayTermOccurrence, GatewaySuggestion, GatewayTermVerificationResult
from n_note_ai_service_layers.api import deps
from n_note_ai_service_layers.services import form_extraction
from n_note_ai_service_layers.contracts.audio import Transcript
from n_note_ai_service_layers.services import nursing_note_workflow
from llm_support import FakeServer, chat, responses


def fake_review(verify):
    """Gateway 용어 검토 흐름 대역이다(catalog 조회는 항상 성공)."""
    return lambda gateway: SimpleNamespace(gateway=SimpleNamespace(discover=lambda request_id=None: ()), verify=verify)


def setup_review(app, monkeypatch, result=None):
    app.dependency_overrides[get_transcript_redactor] = lambda: TranscriptRedactor()
    app.dependency_overrides[get_gateway_client] = lambda: object()
    monkeypatch.setattr(deps, "get_term_verification", fake_review(lambda *args: result or GatewayTermVerificationResult("transcript", "ok")))


def test_internal_stt_multipart_and_output_validation(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("고혈압이 있습니다.", "fake", "fake-stt"))
    monkeypatch.setattr(form_extraction, "extract", lambda transcript, template, hints, *_: {
        "fields": {"history": "고혈압", "choice": {"status": "없음", "detail": "잘못된 상세"}, "allergy": ["약", "허위"]},
        "relatedSentences": {"history": [0, 7]},
    })
    template = {"templateType": "CUSTOM", "templateName": "서식", "schemaDefinition": {"sections": [{"fields": [
        {"field_id": "history", "field_type": "text"},
        {"field_id": "choice", "field_type": "radio", "options": ["없음", "있음"], "has_detail_text": True},
        {"field_id": "allergy", "field_type": "checkbox", "options": ["약"]},
    ]}]}}
    client = TestClient(app)
    response = client.post("/internal/v1", files={"file": ("test.aac", b"audio", "audio/aac"), "templates": ("templates.json", json.dumps([template]), "application/json")})
    assert response.status_code == 200
    body = response.json()
    assert body.pop("recordedAt")
    assert body == {"transcript": "고혈압이 있습니다.", "forms": [{"templateType": "CUSTOM", "templateName": "서식", "fields": {"history": "고혈압", "choice": {"status": "없음", "detail": ""}, "allergy": ["약"]}, "relatedSentences": {"history": ["고혈압이 있습니다."]}, "numericVerification": {"status": "ok", "findings": []}}]}


def test_internal_stt_extraction_failure_preserves_transcript(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("전사문", "fake", "fake-stt"))
    def fail(*args):
        raise RuntimeError()
    monkeypatch.setattr(form_extraction, "extract", fail)
    template = {"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": []}}
    response = TestClient(app).post("/internal/v1", files={"file": ("test.wav", b"audio", "audio/wav")}, data={"templates": json.dumps([template])})
    body = response.json()
    assert body.pop("recordedAt")
    assert body == {"transcript": "전사문", "forms": [{"templateType": "X", "templateName": "Y", "fields": {}, "relatedSentences": {}, "numericVerification": {"status": "skipped", "findings": []}}]}
    assert TestClient(app).post("/internal/v1", files={"file": ("bad.txt", b"x")}, data={"templates": "[]"}).status_code == 415


def test_redaction_and_term_hints_precede_extraction(monkeypatch):
    app = create_app()
    occurrence = GatewayTermOccurrence("transcript", "medical", 0, 3, "고혈압", "고혈압", (GatewaySuggestion("고혈압증"),), category="medical_term")
    setup_review(app, monkeypatch, GatewayTermVerificationResult("transcript", "ok", (occurrence,)))
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("홍길동님 고혈압", "fake", "fake-stt"))
    seen = {}
    def extract(sentences, template, hints, *_):
        seen.update(sentences=sentences, hints=hints)
        return {"fields": {"history": "고혈압증"}, "relatedSentences": {"history": [0, 0]}}
    monkeypatch.setattr(form_extraction, "extract", extract)
    template = {"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": [{"fields": [{"field_id": "history", "field_type": "text"}]}]}}
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([template])})
    assert response.status_code == 200
    body = response.json()
    assert "홍길동" not in body["transcript"] and seen["sentences"] == [body["transcript"]]
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
    "relatedSentences": {"situation": [0], "smoking": [], "age": [], "allergy": [], "tags": []},
}


def llm_server(monkeypatch, reply):
    """LLM 서버를 가짜 HTTP 서버로 바꾼다. 받은 요청 본문 목록을 돌려준다(모든 요청에 같은 응답)."""
    server = FakeServer(lambda request: reply)
    monkeypatch.setattr(model_providers, "async_openai_client", lambda *args: server.client())
    return server.requests


def chat_client(monkeypatch, content):
    monkeypatch.setenv("N_NOTE_LLM_API", "chat")
    monkeypatch.setenv("N_NOTE_LLM_BASE_URL", "http://local-llm/v1")
    return llm_server(monkeypatch, chat(content))


def test_output_schema_is_flat_and_matches_response_contract():
    schema = form_extraction.output_schema(FORM, 1)
    fields = schema["properties"]["fields"]
    assert fields["required"] == ["situation", "smoking", "age", "allergy", "tags"] and fields["additionalProperties"] is False
    assert fields["properties"]["smoking"]["description"] == "흡연력 - 현재"
    assert fields["properties"]["smoking"]["properties"]["status"]["enum"] == ["nonsmoker", "ex-smoker", ""]
    assert fields["properties"]["allergy"]["required"] == ["selected", "detail"]
    assert fields["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["relatedSentences"]["properties"]["tags"] == {"type": "array", "items": {"type": "integer", "enum": [0]}, "maxItems": 3}
    assert form_extraction.Draft202012Validator(schema).is_valid(VALID)


def test_chat_extraction_requests_structured_output_with_correction_rules(monkeypatch):
    calls = chat_client(monkeypatch, json.dumps(VALID, ensure_ascii=False))
    assert form_extraction.extract(["전사문"], FORM, ()) == VALID
    request = calls[0]
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == form_extraction.output_schema(FORM, 1)
    assert "standard Korean" in request["messages"][0]["content"] and "JSON schema:" in request["messages"][0]["content"]


def test_responses_extraction_requests_structured_output(monkeypatch):
    monkeypatch.setenv("N_NOTE_LLM_API", "responses")
    monkeypatch.delenv("N_NOTE_LLM_BASE_URL", raising=False)
    requests = llm_server(monkeypatch, responses(json.dumps(VALID, ensure_ascii=False)))
    assert form_extraction.extract(["전사문"], FORM, ()) == VALID
    seen = requests[0]
    assert seen["text"]["format"]["strict"] is True and seen["text"]["format"]["schema"] == form_extraction.output_schema(FORM, 1)
    # 전사문을 OpenAI 쪽에 보관하지 않는다.
    assert seen["store"] is False


def test_extraction_uses_shared_generation_options(monkeypatch):
    monkeypatch.setenv("N_NOTE_LLM_TEMPERATURE", "0")
    monkeypatch.setenv("N_NOTE_LLM_MAX_TOKENS", "1024")
    calls = chat_client(monkeypatch, json.dumps(VALID, ensure_ascii=False))
    form_extraction.extract(["전사문"], FORM, ())
    assert (calls[0]["temperature"], calls[0]["max_tokens"]) == (0.0, 1024)


def test_privacy_tokens_are_scrubbed_from_field_values_but_kept_in_evidence():
    sentences = ["[[PERSON_1]]님은 흑색변으로 입원함."]
    result = {**VALID, "fields": {**VALID["fields"], "situation": "[[PERSON_1]]님은 흑색변으로 입원함", "allergy": {"selected": ["약"], "detail": "[[PERSON_2]]가 두드러기"}},
              "relatedSentences": {**VALID["relatedSentences"], "situation": [0]}}
    normalized = form_extraction.normalize(FORM, sentences, result)
    assert normalized["fields"]["situation"] == "흑색변으로 입원함"
    assert normalized["fields"]["allergy"] == {"selected": ["약"], "detail": "두드러기"}
    assert normalized["relatedSentences"]["situation"] == sentences


NESTED = {"fields": {"sbar": VALID["fields"]}, "relatedSentences": VALID["relatedSentences"]}
ARRAY = {"fields": [{"field_id": "situation", "value": "입원"}], "relatedSentences": VALID["relatedSentences"]}
WRAPPED_TEXT = {"fields": {**VALID["fields"], "situation": {"value": "입원"}}, "relatedSentences": VALID["relatedSentences"]}
NO_EVIDENCE = {"fields": VALID["fields"]}


def test_extraction_rejects_structure_that_would_silently_empty_fields(monkeypatch):
    for result in (NESTED, ARRAY, WRAPPED_TEXT, NO_EVIDENCE, []):
        chat_client(monkeypatch, json.dumps(result, ensure_ascii=False))
        try:
            form_extraction.extract(["전사문"], FORM, ())
        except ValueError:
            continue
        raise AssertionError(f"accepted {result}")


def test_out_of_option_value_stays_a_blank_field_not_a_failed_form(monkeypatch):
    result = {**VALID, "fields": {**VALID["fields"], "age": "노인", "smoking": {"status": "비흡연", "detail": ""}}}
    chat_client(monkeypatch, json.dumps(result, ensure_ascii=False))
    normalized = form_extraction.normalize(FORM, ["전사문"], form_extraction.extract(["전사문"], FORM, ()))
    assert normalized["fields"]["age"] == "" and normalized["fields"]["smoking"] == {"status": "", "detail": ""}
    assert normalized["fields"]["situation"] == "흑색변으로 입원함"


def test_malformed_extraction_returns_empty_form_and_logs_failure(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("흑색변으로 입원했습니다.", "fake", "fake-stt"))
    chat_client(monkeypatch, json.dumps(NESTED, ensure_ascii=False))
    logs = []
    capture_stages(monkeypatch, logs)
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([FORM])})
    assert response.status_code == 200
    assert response.json()["forms"] == [{"templateType": "HAND_OVER", "templateName": "인계", "fields": {}, "relatedSentences": {}, "numericVerification": {"status": "skipped", "findings": []}}]
    assert logs == [
        ("stt", {"chars": 13}), ("redaction", {"replacements": 0}),
        ("transcript_term_review", {"status": "ok", "candidates": 0, "hints_deferred": False}), ("form_extraction", "ValueError"),
    ]


def test_extraction_metrics_count_filled_fields_only(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("흑색변으로 입원했습니다.", "fake", "fake-stt"))
    chat_client(monkeypatch, json.dumps(VALID, ensure_ascii=False))
    logs = []
    capture_stages(monkeypatch, logs)
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([FORM])})
    assert response.status_code == 200
    assert logs[-1] == ("form_extraction", {"fields": 5, "filled": 4, "with_evidence": 1})


def capture_stages(monkeypatch, logs):
    """공통 단계(TranscriptPreparation)와 서식 추출(form_extraction)의 단계 로그를 함께 모은다."""
    for module in (nursing_note_workflow, form_extraction):
        monkeypatch.setattr(module, "stage", lambda *args, **kwargs: observed_stage(logs, *args, **kwargs))


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
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("홍길동님", "fake", "fake-stt"))
    def fail(*args):
        raise AssertionError("extraction must not run")
    monkeypatch.setattr(form_extraction, "extract", fail)
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
    assert model_providers.speech_to_text_provider().inner._prompt == prompts.get().stt.context
    monkeypatch.setenv("N_NOTE_STT_PROMPT", "병동 문맥")
    assert model_providers.speech_to_text_provider().inner._prompt == "병동 문맥"


def test_term_review_status_is_logged_when_it_stops_extraction(monkeypatch):
    app = create_app()
    setup_review(app, monkeypatch, GatewayTermVerificationResult("transcript", "unsupported"))
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("전사문", "fake", "fake-stt"))
    logs = []
    # 공통 단계(STT·비식별화·검토) 로그는 TranscriptPreparation이 남긴다.
    monkeypatch.setattr(nursing_note_workflow, "stage", lambda *args, **kwargs: observed_stage(logs, *args, **kwargs))
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps([FORM])})
    assert response.status_code == 503 and response.json() == {"detail": "term review is unavailable"}
    assert logs == [("stt", {"chars": 3}), ("redaction", {"replacements": 0}), ("transcript_term_review", {"status": "unsupported", "candidates": 0, "hints_deferred": False})]


def test_forms_are_extracted_concurrently_and_keep_request_order(monkeypatch):
    import threading
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("전사문", "fake", "fake-stt"))
    # 두 양식이 동시에 실행 중이어야 barrier를 통과한다. 순차 실행이면 BrokenBarrierError로 빈 서식이 된다.
    barrier = threading.Barrier(2, timeout=5)
    def extract(sentences, template, hints, *_):
        barrier.wait()
        return {"fields": {"note": template["templateType"]}, "relatedSentences": {}}
    monkeypatch.setattr(form_extraction, "extract", extract)
    templates = [
        {"templateType": kind, "templateName": kind, "schemaDefinition": {"sections": [{"fields": [{"field_id": "note", "field_type": "text"}]}]}}
        for kind in ("A", "B")
    ]
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(templates)})
    assert response.status_code == 200
    assert [form["fields"] for form in response.json()["forms"]] == [{"note": "A"}, {"note": "B"}]


def test_sentences_split_on_endings_and_long_clauses_and_stay_transcript_substrings():
    long_run = ", ".join(["혈압 측정함"] * 40)
    text = f"흑색변으로 입원했습니다. 통증 있나요?\n{long_run}"
    sentences = form_extraction.split_sentences(text)
    assert sentences[:2] == ["흑색변으로 입원했습니다.", "통증 있나요?"]
    assert all(len(sentence) <= form_extraction.MAX_SENTENCE_CHARS and sentence in text for sentence in sentences)
    assert form_extraction.split_sentences("") == []


def test_evidence_ids_map_to_sentences_deduplicated_and_capped():
    template = {"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": [{"fields": [{"field_id": "a", "field_type": "text"}]}]}}
    sentences = ["s0", "s1", "s2", "s3"]
    result = {"fields": {"a": "값"}, "relatedSentences": {"a": [2, 2, True, -1, 9, "1", 0, 1, 3]}}
    assert form_extraction.normalize(template, sentences, result)["relatedSentences"]["a"] == ["s2", "s0", "s1"]


def test_extraction_sends_numbered_sentences_instead_of_raw_transcript(monkeypatch):
    calls = chat_client(monkeypatch, json.dumps(VALID, ensure_ascii=False))
    form_extraction.extract(["첫 문장.", "둘째 문장."], FORM, ())
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["transcript_sentences"] == [{"id": 0, "text": "첫 문장."}, {"id": 1, "text": "둘째 문장."}] and "transcript" not in payload
    assert "transcript_sentences" in calls[0]["messages"][0]["content"]


def _slow_review(app, monkeypatch, status):
    import time
    def verify(*args):
        time.sleep(0.3)
        return GatewayTermVerificationResult("transcript", status)
    app.dependency_overrides[get_transcript_redactor] = lambda: TranscriptRedactor()
    app.dependency_overrides[get_gateway_client] = lambda: object()
    monkeypatch.setattr(deps, "get_term_verification", fake_review(verify))
    monkeypatch.setenv("N_NOTE_TERM_HINT_BUDGET_SECONDS", "0.05")
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript("전사문", "fake", "fake-stt"))
    seen = []
    def extract(sentences, template, hints, *_):
        seen.append(hints)
        return {"fields": {"note": "값"}, "relatedSentences": {"note": [0]}}
    monkeypatch.setattr(form_extraction, "extract", extract)
    return seen


SIMPLE = [{"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": [{"fields": [{"field_id": "note", "field_type": "text"}]}]}}]


def test_late_term_review_starts_extraction_without_hints(monkeypatch):
    app = create_app()
    seen = _slow_review(app, monkeypatch, "ok")
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(SIMPLE)})
    assert response.status_code == 200 and seen == [()]
    assert response.json()["forms"][0]["relatedSentences"] == {"note": ["전사문"]}


def test_late_term_review_failure_still_returns_503_after_extraction(monkeypatch):
    app = create_app()
    seen = _slow_review(app, monkeypatch, "unavailable")
    response = TestClient(app).post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(SIMPLE)})
    assert response.status_code == 503 and response.json() == {"detail": "term review is unavailable"}
    assert seen == [()]


# --- 녹음 시각·기록 시각 기본값·수치 대조(초안 작업 API와 같은 규칙) ---

TIMED = [{"templateType": "HAND_OVER", "templateName": "인계", "schemaDefinition": {"sections": [{"fields": [
    {"field_id": "recorded", "label": "기록 일시", "field_type": "text", "default_from": "recording_time"},
    {"field_id": "status", "label": "상태", "field_type": "text"},
]}]}}]


def _timed_app(monkeypatch, transcript, fields):
    app = create_app()
    setup_review(app, monkeypatch)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: SimpleNamespace(transcribe=lambda audio: Transcript(transcript, "fake", "fake-stt"))
    seen = {}
    def extract(sentences, template, hints, reference_time=None):
        seen["reference_time"] = reference_time
        return {"fields": fields, "relatedSentences": {"status": [0]}}
    monkeypatch.setattr(form_extraction, "extract", extract)
    return TestClient(app), seen


def test_recorded_at_part_sets_reference_time_and_fills_default_field(monkeypatch):
    client, seen = _timed_app(monkeypatch, "맥박 92회입니다.", {"recorded": "", "status": "맥박 92회임."})
    response = client.post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(TIMED), "recorded_at": "2026-11-18T08:00:00+09:00"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recordedAt"] == "2026-11-18T08:00:00+09:00"
    assert seen["reference_time"].isoformat() == "2026-11-18T08:00:00+09:00"
    assert body["forms"][0]["fields"] == {"recorded": "2026-11-18 08:00", "status": "맥박 92회임."}
    # 서비스가 채운 녹음 시각의 숫자는 출처 대조에서 제외한다.
    assert body["forms"][0]["numericVerification"] == {"status": "ok", "findings": []}


def test_time_read_from_the_transcript_is_kept(monkeypatch):
    client, _ = _timed_app(monkeypatch, "14시 30분 맥박 92회입니다.", {"recorded": "2026-11-18 14:30", "status": "맥박 92회임."})
    response = client.post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(TIMED), "recorded_at": "2026-11-18T08:00:00+09:00"})
    assert response.json()["forms"][0]["fields"]["recorded"] == "2026-11-18 14:30"


def test_without_recorded_at_the_receipt_time_is_used(monkeypatch):
    client, seen = _timed_app(monkeypatch, "통증 호소", {"recorded": "", "status": "통증 호소함."})
    body = client.post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(TIMED)}).json()
    assert seen["reference_time"] is not None and body["recordedAt"] == seen["reference_time"].isoformat()


def test_numeric_verification_reports_missing_and_unsupported_values(monkeypatch):
    client, _ = _timed_app(monkeypatch, "맥박은 분당 92회, 체온 38.2도입니다.", {"recorded": "", "status": "맥박 정상 범위(60-100회/분)임. 체온 38.2도."})
    form = client.post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(TIMED), "recorded_at": "2026-11-18T08:00:00+09:00"}).json()["forms"][0]
    verification = form["numericVerification"]
    assert verification["status"] == "needs_review"
    assert [(item["fieldId"], item["text"], item["reason"]) for item in verification["findings"]] == [
        ("status", "60", "not_in_source"), ("status", "100", "not_in_source"), (None, "92", "missing_from_draft"),
    ]


def test_invalid_recorded_at_and_default_from_are_rejected(monkeypatch):
    client, _ = _timed_app(monkeypatch, "통증", {"recorded": "", "status": ""})
    bad_time = client.post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(TIMED), "recorded_at": "어제"})
    assert bad_time.status_code == 422
    radio = [{"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": [{"fields": [
        {"field_id": "r", "field_type": "radio", "options": ["예"], "default_from": "recording_time"}]}]}}]
    assert client.post("/internal/v1", files={"file": ("a.wav", b"x")}, data={"templates": json.dumps(radio)}).status_code == 422


def test_label_left_after_token_scrub_becomes_an_empty_value():
    form = {"templateType": "X", "templateName": "Y", "schemaDefinition": {"sections": [{"fields": [
        {"field_id": "summary", "field_type": "text"},
        {"field_id": "med", "field_type": "radio", "has_detail_text": True, "options": ["없음", "있음"]}]}]}}
    result = {"fields": {"summary": "환자명: [[LABELED_PERSON_1]].", "med": {"status": "있음", "detail": "환자명: [[LABELED_PERSON_1]]"}}, "relatedSentences": {}}
    assert form_extraction.normalize(form, ["문장"], result)["fields"] == {"summary": "", "med": {"status": "있음", "detail": ""}}


def test_extraction_payload_carries_reference_time_and_prompt_rules(monkeypatch):
    from datetime import datetime, timedelta, timezone
    calls = chat_client(monkeypatch, json.dumps(VALID, ensure_ascii=False))
    form_extraction.extract(["전사문"], FORM, (), datetime(2026, 11, 18, 8, 0, tzinfo=timezone(timedelta(hours=9))))
    assert json.loads(calls[0]["messages"][1]["content"])["reference_time"] == "2026-11-18T08:00+09:00"
    instructions = calls[0]["messages"][0]["content"]
    assert "never drop a measurement" in instructions and "Do not add clinical interpretations" in instructions


def test_parse_templates_rejects_each_contract_violation():
    import pytest

    field = {"field_id": "a", "field_type": "text"}
    def template(*fields, **extra):
        return {"templateType": "T", "templateName": "N", "schemaDefinition": {"sections": [{"fields": list(fields)}]}, **extra}
    assert form_extraction.parse_templates(json.dumps([template(field, label="x")]))[0]["label"] == "x"  # 추가 키는 그대로 둔다
    bad = [
        "not json", json.dumps({}), json.dumps([template(field, templateType="")]),
        json.dumps([template(field, field)]),  # field_id 중복
        json.dumps([template({"field_id": "a", "field_type": "date"})]),
        json.dumps([template({"field_id": "a", "field_type": "radio"})]),  # options 없음
        json.dumps([template({"field_id": "a", "field_type": "radio", "options": ["x"], "default_from": "recording_time"})]),
        json.dumps([template({"field_id": "a", "field_type": "text", "default_from": "now"})]),
    ]
    for raw in bad:
        with pytest.raises(ValueError):
            form_extraction.parse_templates(raw)
