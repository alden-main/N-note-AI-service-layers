"""표준 해석과 결정적 후보 검증의 핵심 계약을 확인한다."""

import json
from pathlib import Path
import unittest

from main_mcp_server_document_analysis.contracts import InputError
from main_mcp_server_document_analysis.hmd_standard import HmdAnalysisRequirementSetV1Interpreter
from main_mcp_server_document_analysis.standards import ReportValidationStandardV1Interpreter, StandardInterpreterRegistry
from main_mcp_server_document_analysis.validation import ReportValidationService


def standard() -> dict:
    """테스트에 쓰는 작고 유효한 범용 v1 기준을 반환한다."""
    return {
        "schema_version": "report-validation-standard/v1", "standard_id": "example", "standard_version": "v1", "status": "approved",
        "usage": {"result_policy": "candidate_only_human_review_required", "mutation_policy": "no_automatic_report_modification"},
        "criteria": [{"criterion_id": "endpoint", "description": "endpoint result", "required_elements": [{"element_id": "p_value", "description": "p-value"}]}],
    }


def service() -> ReportValidationService:
    """범용 및 HMD interpreter가 등록된 검증 서비스를 만든다."""
    return ReportValidationService(StandardInterpreterRegistry((ReportValidationStandardV1Interpreter(), HmdAnalysisRequirementSetV1Interpreter())))


class ValidationTest(unittest.TestCase):
    """기준 해석과 후보 검증의 관찰 가능한 결과를 검증한다."""

    def test_hmd_fixture_uses_only_report_validation_criteria(self) -> None:
        """HMD의 공유 메타데이터와 evidence_rules는 해석하지 않는다."""
        raw = json.loads((Path(__file__).parents[1] / "docs/hmd_analysis_requirement_set_v1.json").read_text())
        raw["status"] = {"shared": "metadata"}
        raw["usage"] = "ignored"
        raw["analysis_requirements"][0]["evidence_rules"] = {"unrelated": "ignored"}
        result = HmdAnalysisRequirementSetV1Interpreter().interpret(raw)
        self.assertEqual((result.status, len(result.criteria)), ("unclassified", 7))
        self.assertEqual(result.criteria[0].required_elements[-1].element_id, "p_value")

    def test_duplicate_element_is_invalid_input(self) -> None:
        """중복 required element는 범용 기준 계약 위반으로 거절된다."""
        raw = standard()
        raw["criteria"][0]["required_elements"].append({"element_id": "p_value"})
        with self.assertRaises(InputError):
            ReportValidationStandardV1Interpreter().interpret(raw)

    def test_no_candidate_is_uncertain_not_missing(self) -> None:
        """의미 분석 후보가 없으면 세부 요소를 missing으로 단정하지 않는다."""
        result = service().evaluate({"text": "서술형 보고서"}, standard()).as_dict()
        self.assertEqual(result["criteria"][0]["finding"]["status"], "uncertain")
        self.assertEqual(result["criteria"][0]["elements"][0]["status"], "uncertain")

    def test_missing_element_is_separate_from_satisfied_criterion(self) -> None:
        """기준 전체와 p-value 누락 결과가 독립적으로 반환된다."""
        text = "Endpoint change was reported."
        candidate = {"criteria": [{"criterion_id": "endpoint", "status": "satisfied", "citations": [{"start": 0, "end": len(text), "quote": text}], "elements": [{"element_id": "p_value", "status": "missing", "reason": "not reported"}]}]}
        result = service().evaluate({"text": text}, standard(), candidate).as_dict()
        self.assertEqual(result["criteria"][0]["finding"]["status"], "satisfied")
        self.assertEqual(result["criteria"][0]["elements"][0]["status"], "missing")

    def test_invalid_citation_is_normalized_to_uncertain(self) -> None:
        """조작된 quote는 성공 결과가 아니라 uncertain warning이 된다."""
        candidate = {"criteria": [{"criterion_id": "endpoint", "status": "satisfied", "citations": [{"start": 0, "end": 3, "quote": "bad"}]}]}
        finding = service().evaluate({"text": "good report"}, standard(), candidate).as_dict()["criteria"][0]["finding"]
        self.assertEqual(finding["status"], "uncertain")
        self.assertIn("citation does not match report", finding["warnings"])

    def test_evidence_search_returns_sentence_offsets_without_judgment(self) -> None:
        """검색 도구는 대소문자와 무관한 일치 문장·원문 좌표만 반환한다."""
        text = "Endpoint result was reported.\nP-VALUE was 0.01."
        result = service().find_evidence_candidates(
            {"text": text}, standard(),
            [{"criterion_id": "endpoint", "element_id": "p_value", "terms": ["p-value", "0.01"]}],
        )
        match = result["queries"][0]["matches"][0]
        self.assertEqual(match["quote"], "P-VALUE was 0.01.")
        self.assertEqual(text[match["start"]:match["end"]], match["quote"])
        self.assertEqual(match["matched_terms"], ["p-value", "0.01"])

    def test_evidence_search_rejects_unknown_element(self) -> None:
        """기준에 없는 요소를 검색 대상으로 지정할 수 없다."""
        with self.assertRaises(InputError):
            service().find_evidence_candidates(
                {"text": "report"}, standard(),
                [{"criterion_id": "endpoint", "element_id": "unknown", "terms": ["report"]}],
            )

    def test_evidence_search_normalizes_unicode_whitespace_without_losing_offsets(self) -> None:
        """공백 없는 표제어도 원문의 일반·줄바꿈·전각 공백 표기를 찾는다."""
        text = "결과 보고서\n결과\n보고서\n결과　보고서"
        matches = service().find_evidence_candidates(
            {"text": text}, standard(), [{"criterion_id": "endpoint", "terms": ["결과보고서"]}]
        )["queries"][0]["matches"]
        self.assertEqual([item["quote"] for item in matches], ["결과 보고서", "결과\n보고서", "결과　보고서"])
        self.assertTrue(all(item["match_strategy"] == "normalized_literal" for item in matches))
        self.assertTrue(all(text[item["start"]:item["end"]] == item["quote"] for item in matches))

    def test_evidence_search_uses_only_supplied_aliases_and_registered_patterns(self) -> None:
        """문서별 동의어는 caller가 주고, 기간 패턴은 allow-list ID로만 사용한다."""
        text = "3개월 동안의 모니터링. 제2장 결과분석. 제3장 결론. 임상적 유효성."
        result = service().find_evidence_candidates(
            {"text": text}, standard(),
            [{
                "criterion_id": "endpoint", "terms": ["연구기간"],
                "aliases": ["결과분석", "결론", "임상적 유효성"],
                "pattern_ids": ["korean_duration_expression/v1"],
            }],
        )["queries"][0]["matches"]
        self.assertEqual({item["match_strategy"] for item in result}, {"alias", "registered_pattern"})
        self.assertIn("3개월 동안", [item["quote"] for item in result])
        without_alias = service().find_evidence_candidates(
            {"text": text}, standard(), [{"criterion_id": "endpoint", "terms": ["연구기간"]}]
        )["queries"][0]["matches"]
        self.assertEqual(without_alias, [])

    def test_evidence_search_rejects_unapproved_patterns_and_invalid_aliases(self) -> None:
        """raw regex, 미등록 ID, 빈 값 및 정규화 후 중복 별칭은 허용하지 않는다."""
        invalid_queries = [
            {"criterion_id": "endpoint", "terms": ["report"], "pattern_ids": [".*"]},
            {"criterion_id": "endpoint", "terms": ["report"], "aliases": [""]},
            {"criterion_id": "endpoint", "terms": ["report"], "aliases": ["연구기간", "연구 기간"]},
            {"criterion_id": "endpoint", "terms": ["report"], "pattern_ids": ["korean_duration_expression/v1", "korean_duration_expression/v1"]},
        ]
        for query in invalid_queries:
            with self.subTest(query=query), self.assertRaises(InputError):
                service().find_evidence_candidates({"text": "report"}, standard(), [query])
