"""요청당 한 번 만드는 검색 색인이 이전 검색 구현과 같은 결과를 내는지 확인한다."""

import random
import re
import unittest

from main_mcp_server_document_analysis import validation
from main_mcp_server_document_analysis.validation import _REGISTERED_PATTERNS, _ReportIndex, _report_sentences


def _reference_normalized_spans(report: str, expression: str) -> tuple[tuple[int, int], ...]:
    """검색어마다 보고서 전체를 다시 casefold하던 이전 구현이다."""
    target = "".join(character.casefold() for character in expression.strip() if not character.isspace())
    if not target:
        return ()
    folded: list[str] = []
    indices: list[int] = []
    for index, character in enumerate(report):
        if character.isspace():
            continue
        for folded_character in character.casefold():
            folded.append(folded_character)
            indices.append(index)
    haystack = "".join(folded)
    spans: list[tuple[int, int]] = []
    start_at = 0
    while (found := haystack.find(target, start_at)) >= 0:
        spans.append((indices[found], indices[found + len(target) - 1] + 1))
        start_at = found + len(target)
    return tuple(spans)


def _reference_matches(report: str, terms: tuple[str, ...], aliases: tuple[str, ...], pattern_ids: tuple[str, ...]) -> list[dict]:
    """문장 목록을 선형 탐색하던 이전 `_find_evidence_matches`다."""
    sentences = _report_sentences(report)
    matches: dict = {}

    def add(start: int, end: int, matched_term: str, strategy: str, sentence_candidate: bool) -> None:
        sentence = next((item for item in sentences if item[0] <= start and end <= item[1]), None) if sentence_candidate else None
        quote_start, quote_end = sentence[:2] if sentence else (start, end)
        item = matches.setdefault((quote_start, quote_end, strategy), {
            "start": quote_start, "end": quote_end, "quote": report[quote_start:quote_end], "matched_terms": [], "match_strategy": strategy,
        })
        if matched_term not in item["matched_terms"]:
            item["matched_terms"].append(matched_term)

    for expressions, literal_strategy, normalized_strategy in ((terms, "literal", "normalized_literal"), (aliases, "alias", "normalized_alias")):
        for expression in expressions:
            exact = {(m.start(), m.end()) for m in re.finditer(re.escape(expression), report, re.IGNORECASE)}
            for start, end in exact:
                add(start, end, expression, literal_strategy, True)
            for start, end in _reference_normalized_spans(report, expression):
                if (start, end) not in exact:
                    add(start, end, expression, normalized_strategy, True)
    for pattern_id in pattern_ids:
        for match in _REGISTERED_PATTERNS[pattern_id].finditer(report):
            add(match.start(), match.end(), match.group(), "registered_pattern", False)
    return sorted(matches.values(), key=lambda item: (item["start"], item["end"], item["match_strategy"]))


class EvidenceIndexEquivalenceTest(unittest.TestCase):
    """casefold 길이가 바뀌는 문자('ß'·'İ')와 공백·문장 경계를 섞어 이전 구현과 비교한다."""

    CASES = (
        ("Straße ist groß. STRASSE\nstrasse.", ("strasse", "STRAßE", "groß"), ("ss ist",)),
        ("İstanbul i̇stanbul. ISTANBUL istanbul!", ("i̇stanbul", "İstanbul", "istanbul"), ("bul i",)),
        ("결과 보고서\n결과\n보고서\n결과　보고서. 3개월 동안 관찰.", ("결과보고서", "보고서"), ("결과 보",)),
        ("P-VALUE was 0.01. p - value 0.01 again.\n", ("p-value", "0.01", "value0.01"), ()),
        ("", ("x",), ()),
    )

    def test_fixed_cases_match_previous_implementation(self) -> None:
        for report, terms, aliases in self.CASES:
            with self.subTest(report=report):
                pattern_ids = ("korean_duration_expression/v1",)
                expected = _reference_matches(report, terms, aliases, pattern_ids)
                actual = validation._find_evidence_matches(_ReportIndex.build(report), terms, aliases, pattern_ids)
                self.assertEqual(actual, expected)

    def test_random_reports_match_previous_implementation(self) -> None:
        alphabet = ["a", "b", "ß", "s", "S", "İ", "i", "̇", " ", "\n", "　", ".", "!", "1", "결", "과"]
        generator = random.Random(20261123)
        for _ in range(300):
            report = "".join(generator.choice(alphabet) for _ in range(generator.randint(0, 40)))
            terms = tuple(dict.fromkeys(
                term for term in ("".join(generator.choice(alphabet) for _ in range(generator.randint(1, 4))) for _ in range(3)) if term.strip()
            ))
            with self.subTest(report=report, terms=terms):
                expected = _reference_matches(report, terms, (), ())
                actual = validation._find_evidence_matches(_ReportIndex.build(report), terms, (), ())
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
