from __future__ import annotations

import json
import unittest

from n_note_ai_service_layers.contracts.form_schema import FormField, FormSchema, FormSection, SchemaStatus
from n_note_ai_service_layers.services.draft_document import DraftDocument
from n_note_ai_service_layers.services.gateway_term_verification import GatewayTermVerificationFlow
from n_note_ai_service_layers.services.mcp_gateway_client import McpGatewayClient

KOREAN_TOOL = "korean-standard-dictionary__review_standard_korean_text"
MEDICAL_TOOL = "medical-term-review__analyze_document_terms"
TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}
MEDICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "document_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "document_offset": {"type": "integer", "default": 0},
    },
    "required": ["text"],
}


class _Gateway:
    """Gateway가 upstream 결과를 그대로 중계하는 응답을 흉내 낸다."""

    def __init__(self, results, batch=True):
        self.results = results
        self.calls = []
        self.batch = batch
        self.batch_requests = 0

    def _tool(self, name, arguments):
        self.calls.append((name, arguments["text"]))
        return self.results[name](arguments["text"])

    def __call__(self, url, body, headers, timeout):
        message = json.loads(body)
        if url.endswith("/v1/tools/batch-call"):
            # 일괄 호출을 모르는 구버전 Gateway는 404를 반환한다.
            if not self.batch:
                return 404, {}, b"not found"
            self.batch_requests += 1
            results = [self._tool(item["name"], item["arguments"]) for item in message["calls"]]
            return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps({"results": results}, ensure_ascii=False).encode()
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        elif message["method"] == "tools/list":
            result = {"tools": [
                {"name": KOREAN_TOOL, "description": "review", "inputSchema": TEXT_SCHEMA},
                {"name": MEDICAL_TOOL, "description": "analyze", "inputSchema": MEDICAL_SCHEMA},
                {"name": "medical-term-review__submit_review_suggestions", "description": "submit", "inputSchema": {"type": "object", "properties": {"analysis_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["analysis_id", "text"]}},
            ]}
        else:
            result = self._tool(message["params"]["name"], message["params"]["arguments"])
        return 200, {"X-Request-ID": headers["X-Request-ID"]}, json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}, ensure_ascii=False).encode()


def _korean(findings):
    return lambda text: {"structuredContent": findings(text), "isError": False}


def _medical(candidates):
    # FastMCP는 Union 반환 타입을 {"result": ...}로 감싼다.
    return lambda text: {"structuredContent": {"result": {"analysis_id": "a1", "dictionary_version": "v", "status": "completed", "suggestions": [], "host_reviewed": False, "candidates": candidates(text)}}, "isError": False}


def _no_candidates(text):
    return []


def _finding(text, source, replacement, status="needs_review"):
    start = text.index(source)
    return {"start": start, "end": start + len(source), "source_text": source, "analysis_form": source, "status": status, "candidates": [{"replacement": replacement, "confidence": 0.98, "reason": "rule"}]}


def _term(kor, eng):
    return {"source_id": "s", "term_code": "T1", "concept_code": "C1", "eng_name": eng, "kor_name": kor}


class GatewayTermVerificationTests(unittest.TestCase):
    def _flow(self, gateway, tool_names=(KOREAN_TOOL, MEDICAL_TOOL), chunk_chars=8000):
        return GatewayTermVerificationFlow(McpGatewayClient(transport=gateway), tool_names, chunk_chars)

    def test_returns_only_valid_structured_candidates_with_tool_provenance(self):
        gateway = _Gateway({KOREAN_TOOL: _korean(lambda text: {"findings": [_finding(text, "금새", "금세")], "warnings": []}), MEDICAL_TOOL: _medical(_no_candidates)})
        result = self._flow(gateway).verify("transcript", "금새 확인")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.occurrences[0].tool_name, KOREAN_TOOL)
        self.assertEqual(result.occurrences[0].category, "standard_korean")
        self.assertEqual(result.occurrences[0].suggestions[0].text, "금세")
        self.assertFalse(result.auto_apply)
        self.assertNotIn("medical-term-review__submit_review_suggestions", [name for name, _ in gateway.calls])

    def test_rejects_offset_or_source_text_mismatch_instead_of_inventing_candidates(self):
        bad = {"findings": [{"start": 0, "end": 2, "source_text": "금세", "analysis_form": "금세", "status": "needs_review", "candidates": []}]}
        result = self._flow(_Gateway({KOREAN_TOOL: _korean(lambda text: bad)}), (KOREAN_TOOL,)).verify("transcript", "금새 확인")
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.occurrences, ())

    def test_skips_dictionary_matches_that_need_no_change(self):
        matched = lambda text: {"findings": [_finding(text, "확인", "확인", "matched")]}
        result = self._flow(_Gateway({KOREAN_TOOL: _korean(matched)}), (KOREAN_TOOL,)).verify("transcript", "금새 확인")
        self.assertEqual((result.status, result.occurrences), ("ok", ()))

    def test_unwraps_medical_result_and_skips_exact_matches(self):
        def candidates(text):
            return [
                {"candidate_id": "c1", "source_text": "헤모글로빈", "start_offset": 0, "end_offset": 5, "candidate_type": "exact_match", "terms": [_term("헤모글로빈", "Hemoglobin")], "automatic_correction": False},
                {"candidate_id": "c2", "source_text": "hemoglobn", "start_offset": 6, "end_offset": 15, "candidate_type": "unknown_or_possible_typo", "terms": [_term("헤모글로빈", "Hemoglobin")], "automatic_correction": False},
            ]
        result = self._flow(_Gateway({MEDICAL_TOOL: _medical(candidates)}), (MEDICAL_TOOL,)).verify("transcript", "헤모글로빈 hemoglobn 확인")
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.occurrences), 1)
        self.assertEqual(result.occurrences[0].input_text, "hemoglobn")
        self.assertEqual(result.occurrences[0].suggestions[0].text, "Hemoglobin")
        self.assertEqual(result.occurrences[0].category, "medical_term")

    def test_medical_inflections_and_compounds_are_not_corrections(self):
        def candidates(text):
            return [
                {"candidate_id": "c1", "source_text": "통증은", "start_offset": 0, "end_offset": 3, "candidate_type": "variant_match", "terms": [_term("통증", "Pain")], "automatic_correction": False},
                {"candidate_id": "c2", "source_text": "수치", "start_offset": 4, "end_offset": 6, "candidate_type": "unknown_or_possible_typo", "terms": [_term("수치값", "Value"), _term("수치", "Level")], "automatic_correction": False},
            ]
        result = self._flow(_Gateway({MEDICAL_TOOL: _medical(candidates)}), (MEDICAL_TOOL,)).verify("transcript", "통증은 수치 확인")
        self.assertEqual(result.occurrences, ())

    def test_medical_typo_guesses_on_standard_korean_words_are_dropped(self):
        text = "가지 말라고 교육함. 헤모글로반 확인"
        def findings(chunk):
            start = chunk.index("말라고")
            return {"findings": [{"start": start, "end": start + 1, "source_text": "말", "analysis_form": "말", "status": "matched", "candidates": [{"replacement": "말", "confidence": 1.0}]}]}
        def candidates(chunk):
            return [
                {"candidate_id": "c1", "source_text": "말라고", "start_offset": chunk.index("말라고"), "end_offset": chunk.index("말라고") + 3, "candidate_type": "unknown_or_possible_typo", "terms": [_term("말라리아", "Malaria")], "automatic_correction": False},
                {"candidate_id": "c2", "source_text": "헤모글로반", "start_offset": chunk.index("헤모"), "end_offset": chunk.index("헤모") + 5, "candidate_type": "unknown_or_possible_typo", "terms": [_term("헤모글로빈", "Hemoglobin")], "automatic_correction": False},
            ]
        result = self._flow(_Gateway({KOREAN_TOOL: _korean(findings), MEDICAL_TOOL: _medical(candidates)})).verify("transcript", text)
        self.assertEqual([item.input_text for item in result.occurrences], ["헤모글로반"])

    def test_long_text_is_split_and_offsets_refer_to_the_whole_text(self):
        text = "통증 호소함\n" * 5 + "금새 호전됨\n"
        gateway = _Gateway({KOREAN_TOOL: _korean(lambda chunk: {"findings": [_finding(chunk, "금새", "금세")] if "금새" in chunk else []})})
        result = self._flow(gateway, (KOREAN_TOOL,), chunk_chars=20).verify("transcript", text)
        self.assertGreater(len(gateway.calls), 1)
        self.assertTrue(all(len(chunk) <= 20 for _, chunk in gateway.calls))
        occurrence = result.occurrences[0]
        self.assertEqual(text[occurrence.start_offset:occurrence.end_offset], "금새")

    def test_draft_candidates_are_mapped_to_field_relative_offsets(self):
        schema = FormSchema("t", "v1", "T", (FormSection("s", (FormField("a", "text", label="사정"), FormField("b", "text", label="중재"))),), status=SchemaStatus.ACTIVE)
        document = DraftDocument.build(schema, {"a": "통증 없음", "b": "금새 안정됨"})
        gateway = _Gateway({KOREAN_TOOL: _korean(lambda text: {"findings": [_finding(text, "금새", "금세"), _finding(text, "중재", "중제")]})})
        result = self._flow(gateway, (KOREAN_TOOL,)).verify_document("draft", document)
        # 라벨("중재")의 후보는 필드 값이 아니므로 적용 대상에서 제외한다.
        self.assertEqual([(item.field_id, item.start_offset, item.end_offset) for item in result.occurrences], [("b", 0, 2)])

    def test_review_calls_go_out_in_one_gateway_batch(self):
        findings = lambda text: {"findings": [_finding(text, "금새", "금세")] if "금새" in text else []}
        gateway = _Gateway({KOREAN_TOOL: _korean(findings), MEDICAL_TOOL: _medical(_no_candidates)})
        result = self._flow(gateway, chunk_chars=4).verify("transcript", "금새\n확인 필요")
        chunks = len(self._flow(gateway, chunk_chars=4)._chunks("금새\n확인 필요"))
        self.assertGreater(chunks, 1)
        self.assertEqual(gateway.batch_requests, 1)
        self.assertEqual(len(gateway.calls), 2 * chunks)
        self.assertEqual([item.input_text for item in result.occurrences], ["금새"])

    def test_falls_back_to_single_calls_when_gateway_has_no_batch_endpoint(self):
        gateway = _Gateway({KOREAN_TOOL: _korean(lambda text: {"findings": [_finding(text, "금새", "금세")]}), MEDICAL_TOOL: _medical(_no_candidates)}, batch=False)
        flow = self._flow(gateway)
        first = flow.verify("transcript", "금새 확인")
        second = flow.verify("transcript", "금새 확인")
        self.assertEqual(gateway.batch_requests, 0)
        self.assertEqual(len(gateway.calls), 4)
        self.assertEqual(first.occurrences, second.occurrences)
        self.assertEqual(first.occurrences[0].suggestions[0].text, "금세")

    def test_one_failed_tool_in_a_batch_keeps_the_other_candidates(self):
        failed = lambda text: {"isError": True, "structuredContent": {"error": {"code": "upstream_unavailable", "message": "down", "retryable": True}}}
        gateway = _Gateway({KOREAN_TOOL: _korean(lambda text: {"findings": [_finding(text, "금새", "금세")]}), MEDICAL_TOOL: failed})
        result = self._flow(gateway).verify("transcript", "금새 확인")
        self.assertEqual(result.status, "ok")
        self.assertEqual([item.tool_name for item in result.occurrences], [KOREAN_TOOL])


if __name__ == "__main__":
    unittest.main()
