"""비신뢰 semantic candidate를 파싱하고 결정적으로 결과를 정규화한다."""

from __future__ import annotations

import re
from typing import Any

from .contracts import (
    CandidateCriterion, CandidateElement, Citation, CriterionResult, EvaluationResult,
    Finding, InputError, SemanticAssessment, Status, ValidationStandard,
)

_STATUSES = {"satisfied", "missing", "insufficient", "uncertain"}
_CITATION_REQUIRED = {"satisfied", "insufficient"}
_KOREAN_DURATION_PATTERN_ID = "korean_duration_expression/v1"
_REGISTERED_PATTERNS = {
    _KOREAN_DURATION_PATTERN_ID: re.compile(
        r"(?:\d+|[일이삼사오육칠팔구십백천만영공한두세네]+)\s*(?:개월|년|주|일)(?:\s*(?:동안|간))?"
    ),
}


def parse_semantic_assessment(raw: Any) -> SemanticAssessment:
    """후보 JSON을 파싱한다.

    후보가 없으면 빈 후보를 반환한다. criterion 단위 형식 오류는 예외 대신
    invalid candidate marker로 보존해 후속 검증에서 해당 criterion만 uncertain 처리한다.
    """
    if raw is None:
        return SemanticAssessment()
    if not isinstance(raw, dict) or not isinstance(raw.get("criteria"), list):
        raise InputError("semantic_assessment must be an object with criteria")
    criteria: list[CandidateCriterion] = []
    for value in raw["criteria"]:
        if not isinstance(value, dict):
            criteria.append(CandidateCriterion("__invalid__", "uncertain", "invalid candidate criterion"))
            continue
        try:
            criterion_id = _required_text(value.get("criterion_id"), "candidate criterion_id")
            status = _status(value.get("status"))
            citations = _citations(value.get("citations", []))
            elements = tuple(_element(item) for item in value.get("elements", [])) if isinstance(value.get("elements", []), list) else ()
            criteria.append(CandidateCriterion(criterion_id, status, _optional_text(value.get("reason")), citations, elements))
        except InputError:
            criteria.append(CandidateCriterion("__invalid__", "uncertain", "invalid candidate criterion"))
    return SemanticAssessment(tuple(criteria))


def _required_text(value: Any, name: str) -> str:
    """필수 비공백 문자열을 읽는다."""
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a non-blank string")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    """선택 문자열을 반환하고 다른 타입은 입력 오류로 처리한다."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputError("reason must be a string")
    return value


def _status(value: Any) -> Status:
    """허용된 finding 상태만 반환한다."""
    if value not in _STATUSES:
        raise InputError("unsupported candidate status")
    return value


def _citations(raw: Any) -> tuple[Citation, ...]:
    """citation 목록의 구조만 검사해 불변 DTO로 만든다."""
    if not isinstance(raw, list):
        raise InputError("citations must be a list")
    citations: list[Citation] = []
    for citation in raw:
        if not isinstance(citation, dict) or not isinstance(citation.get("start"), int) or not isinstance(citation.get("end"), int):
            raise InputError("invalid citation")
        citations.append(Citation(citation["start"], citation["end"], _required_text(citation.get("quote"), "citation quote")))
    return tuple(citations)


def _element(value: Any) -> CandidateElement:
    """후보의 세부 요소 한 건을 엄격하게 파싱한다."""
    if not isinstance(value, dict):
        raise InputError("candidate element must be an object")
    return CandidateElement(
        _required_text(value.get("element_id"), "candidate element_id"),
        _status(value.get("status")),
        _optional_text(value.get("reason")),
        _citations(value.get("citations", [])),
    )


class CoverageValidator:
    """기준 목록과 citation을 신뢰 원본으로 사용하는 결정적 validator다."""

    def validate(self, standard: ValidationStandard, report: str, candidate: SemanticAssessment) -> EvaluationResult:
        """모든 기준과 요소의 결과를 만든다.

        입력 기준에 없는 ID, 중복 ID, 유효하지 않은 citation은 해당 후보만 무효화한다.
        그러므로 하나의 외부 후보 실패가 다른 criterion 결과를 중단시키지 않는다.
        """
        if not isinstance(report, str):
            raise InputError("report.text must be a string")
        candidates, duplicate_ids = self._index_candidates(candidate)
        results = tuple(self._criterion_result(spec, report, candidates.get(spec.criterion_id), spec.criterion_id in duplicate_ids) for spec in standard.criteria)
        known_ids = {spec.criterion_id for spec in standard.criteria}
        warnings = ["unknown candidate criterion: " + item_id for item_id in candidates if item_id not in known_ids and item_id != "__invalid__"]
        if "__invalid__" in candidates:
            warnings.append("invalid candidate criterion ignored")
        for criterion_id, criterion in candidates.items():
            spec = next((item for item in standard.criteria if item.criterion_id == criterion_id), None)
            if spec is not None:
                known_element_ids = {element.element_id for element in spec.required_elements}
                warnings.extend(f"unknown candidate element: {criterion_id}.{element.element_id}" for element in criterion.elements if element.element_id not in known_element_ids)
        return EvaluationResult(standard.standard_id, standard.standard_version, standard.status, standard.result_policy, standard.mutation_policy, results, warnings=tuple(warnings))

    def _index_candidates(self, candidate: SemanticAssessment) -> tuple[dict[str, CandidateCriterion], set[str]]:
        """candidate criterion ID를 index화하고 중복 ID를 별도로 기록한다."""
        indexed: dict[str, CandidateCriterion] = {}
        duplicates: set[str] = set()
        for item in candidate.criteria:
            if item.criterion_id in indexed:
                duplicates.add(item.criterion_id)
            else:
                indexed[item.criterion_id] = item
        return indexed, duplicates

    def _criterion_result(self, spec: Any, report: str, candidate: CandidateCriterion | None, duplicate: bool) -> CriterionResult:
        """criterion 후보와 각 element 후보를 독립적으로 정규화한다."""
        warning = "duplicate or invalid candidate criterion" if duplicate or (candidate is not None and candidate.criterion_id == "__invalid__") else None
        criterion = self._finding(spec.criterion_id, candidate.status if candidate else "uncertain", candidate.reason if candidate else None, candidate.citations if candidate else (), report, warning)
        elements_by_id: dict[str, CandidateElement] = {}
        duplicate_elements: set[str] = set()
        if candidate and not warning:
            for element in candidate.elements:
                if element.element_id in elements_by_id:
                    duplicate_elements.add(element.element_id)
                else:
                    elements_by_id[element.element_id] = element
        elements = tuple(
            self._finding(element.element_id, elements_by_id[element.element_id].status if element.element_id in elements_by_id else "uncertain", elements_by_id[element.element_id].reason if element.element_id in elements_by_id else None, elements_by_id[element.element_id].citations if element.element_id in elements_by_id else (), report, "duplicate candidate element" if element.element_id in duplicate_elements else None)
            for element in spec.required_elements
        )
        return CriterionResult(spec.criterion_id, spec.description, criterion, elements)

    def _finding(self, item_id: str, status: Status, reason: str | None, citations: tuple[Citation, ...], report: str, warning: str | None) -> Finding:
        """상태별 citation 규칙과 원문 offset/quote 일치를 검사해 finding을 만든다."""
        warnings = [warning] if warning else []
        if status in _CITATION_REQUIRED and not citations:
            warnings.append("citation required for satisfied or insufficient")
        if any(citation.start < 0 or citation.end < citation.start or citation.end > len(report) or report[citation.start:citation.end] != citation.quote for citation in citations):
            warnings.append("citation does not match report")
        if warnings:
            return Finding(item_id, "uncertain", None, (), tuple(warnings))
        return Finding(item_id, status, reason, citations)


class ReportValidationService:
    """기준 해석·candidate 파싱·결정적 검증을 순서대로 조합한다."""

    def __init__(self, registry: Any, validator: CoverageValidator | None = None) -> None:
        """등록된 기준 interpreter와 validator를 보관한다."""
        self._registry = registry
        self._validator = validator or CoverageValidator()

    def evaluate(self, report: dict[str, Any], standard: dict[str, Any], semantic_assessment: Any = None) -> EvaluationResult:
        """요청 본문을 받아 사람이 검토할 평가 결과를 반환한다."""
        if not isinstance(report, dict):
            raise InputError("report must be an object")
        return self._validator.validate(self._registry.interpret(standard), report.get("text"), parse_semantic_assessment(semantic_assessment))

    def guidance(self, standard: dict[str, Any]) -> dict[str, Any]:
        """LLM이 근거를 수집할 수 있는 검토 규칙을 구조화해 반환한다.

        Args:
            standard: schema_version을 포함한 검증 기준 JSON이다.

        Returns:
            기준별 설명·필수 요소와 상태·citation 정책을 담은 JSON 호환 사전이다.
        """
        interpreted = self._registry.interpret(standard)
        return {
            "standard": {
                "standard_id": interpreted.standard_id,
                "standard_version": interpreted.standard_version,
                "result_policy": interpreted.result_policy,
                "mutation_policy": interpreted.mutation_policy,
            },
            "assessment_policy": {
                "allowed_statuses": ["satisfied", "missing", "insufficient", "uncertain"],
                "citation_required_for": ["satisfied", "insufficient"],
                "missing_rule": "내용 부재를 판단한 경우에만 missing을 사용합니다.",
                "uncertain_rule": "근거나 의미 판단이 충분하지 않으면 uncertain을 사용합니다.",
            },
            "criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "category": criterion.category,
                    "description": criterion.description,
                    "required_elements": [
                        {"element_id": element.element_id, "description": element.description}
                        for element in criterion.required_elements
                    ],
                }
                for criterion in interpreted.criteria
            ],
        }

    def find_evidence_candidates(self, report: dict[str, Any], standard: dict[str, Any], queries: Any) -> dict[str, Any]:
        """검색어가 포함된 보고서 문장과 citation 좌표를 결정적으로 반환한다.

        Args:
            report: `text`를 포함하는 보고서 객체다.
            standard: 검색 대상 ID를 제한할 검증 기준 JSON이다.
            queries: criterion_id, 선택 element_id, 비공백 terms 목록을 가진 검색 요청 목록이다.

        Returns:
            요청별 원문 문장, start/end offset 및 실제로 일치한 검색어 목록이다.
        """
        if not isinstance(report, dict) or not isinstance(report.get("text"), str):
            raise InputError("report.text must be a string")
        if not isinstance(queries, list):
            raise InputError("queries must be a list")
        interpreted = self._registry.interpret(standard)
        criteria = {criterion.criterion_id: criterion for criterion in interpreted.criteria}
        sentences = _report_sentences(report["text"])
        results: list[dict[str, Any]] = []
        for index, query in enumerate(queries):
            criterion_id, element_id, terms, aliases, pattern_ids = _parse_evidence_query(query, index, criteria)
            matches = _find_evidence_matches(report["text"], sentences, terms, aliases, pattern_ids)
            results.append({"criterion_id": criterion_id, "element_id": element_id, "matches": matches})
        return {
            "standard_id": interpreted.standard_id,
            "standard_version": interpreted.standard_version,
            "queries": results,
        }


def _parse_evidence_query(
    query: Any, index: int, criteria: dict[str, Any]
) -> tuple[str, str | None, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """검색 요청의 기준 ID·선택 요소 ID·검색어를 검증해 반환한다."""
    if not isinstance(query, dict):
        raise InputError(f"queries[{index}] must be an object")
    criterion_id = _required_text(query.get("criterion_id"), f"queries[{index}].criterion_id")
    criterion = criteria.get(criterion_id)
    if criterion is None:
        raise InputError("unknown query criterion_id")
    element_id = query.get("element_id")
    if element_id is not None:
        element_id = _required_text(element_id, f"queries[{index}].element_id")
        if element_id not in {element.element_id for element in criterion.required_elements}:
            raise InputError("unknown query element_id")
    raw_terms = query.get("terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise InputError("query terms must be a non-empty list")
    terms = tuple(dict.fromkeys(_required_text(term, "query term") for term in raw_terms))
    aliases = _unique_texts(query.get("aliases", []), "aliases")
    pattern_ids = _registered_pattern_ids(query.get("pattern_ids", []))
    return criterion_id, element_id, terms, aliases, pattern_ids


def _unique_texts(value: Any, field_name: str) -> tuple[str, ...]:
    """선택 alias 배열의 빈 값과 중복을 명확한 입력 오류로 처리한다."""
    if not isinstance(value, list):
        raise InputError(f"{field_name} must be a list")
    values = tuple(_required_text(item, field_name) for item in value)
    normalized = tuple("".join(character.casefold() for character in item if not character.isspace()) for item in values)
    if len(normalized) != len(set(normalized)):
        raise InputError(f"duplicate {field_name}")
    return values


def _registered_pattern_ids(value: Any) -> tuple[str, ...]:
    """원격 regex 대신 서버 allow-list의 pattern ID만 받는다."""
    if not isinstance(value, list):
        raise InputError("pattern_ids must be a list")
    pattern_ids = tuple(_required_text(item, "pattern_id") for item in value)
    if len(pattern_ids) != len(set(pattern_ids)):
        raise InputError("duplicate pattern_ids")
    if any(pattern_id not in _REGISTERED_PATTERNS for pattern_id in pattern_ids):
        raise InputError("unknown pattern_id")
    return pattern_ids


def _find_evidence_matches(
    report: str,
    sentences: tuple[tuple[int, int, str], ...],
    terms: tuple[str, ...],
    aliases: tuple[str, ...],
    pattern_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    """문자열·승인 alias·등록 pattern의 원문 범위를 보존해 후보를 만든다."""
    matches: dict[tuple[int, int, str], dict[str, Any]] = {}

    def add(start: int, end: int, matched_term: str, strategy: str, sentence_candidate: bool) -> None:
        # 공백 정규화로 문장 경계를 넘은 경우에는 문장 quote를 만들 수 없으므로
        # 실제 원문 연속 구간 자체를 quote로 사용한다.
        sentence = _enclosing_sentence(sentences, start, end) if sentence_candidate else None
        quote_start, quote_end = sentence[:2] if sentence else (start, end)
        key = (quote_start, quote_end, strategy)
        item = matches.setdefault(key, {
            "start": quote_start,
            "end": quote_end,
            "quote": report[quote_start:quote_end],
            "matched_terms": [],
            "match_strategy": strategy,
        })
        if matched_term not in item["matched_terms"]:
            item["matched_terms"].append(matched_term)

    for expressions, literal_strategy, normalized_strategy in (
        (terms, "literal", "normalized_literal"),
        (aliases, "alias", "normalized_alias"),
    ):
        for expression in expressions:
            exact_spans = set(_literal_spans(report, expression))
            for start, end in exact_spans:
                add(start, end, expression, literal_strategy, True)
            for start, end in _whitespace_normalized_spans(report, expression):
                if (start, end) not in exact_spans:
                    add(start, end, expression, normalized_strategy, True)

    for pattern_id in pattern_ids:
        for match in _REGISTERED_PATTERNS[pattern_id].finditer(report):
            add(match.start(), match.end(), match.group(), "registered_pattern", False)

    return sorted(matches.values(), key=lambda item: (item["start"], item["end"], item["match_strategy"]))


def _literal_spans(report: str, expression: str) -> tuple[tuple[int, int], ...]:
    """Unicode 대소문자를 무시한 직접 문자열 일치의 원문 좌표를 찾는다."""
    return tuple((match.start(), match.end()) for match in re.finditer(re.escape(expression), report, re.IGNORECASE))


def _whitespace_normalized_spans(report: str, expression: str) -> tuple[tuple[int, int], ...]:
    """Unicode 공백 변형을 무시해 일치시키되, 원문 code-point 범위를 반환한다."""
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
        start = indices[found]
        end = indices[found + len(target) - 1] + 1
        spans.append((start, end))
        start_at = found + len(target)
    return tuple(spans)


def _enclosing_sentence(
    sentences: tuple[tuple[int, int, str], ...], start: int, end: int
) -> tuple[int, int, str] | None:
    """일치 범위 전체를 포함하는 문장이 있을 때만 기존 문장 단위 quote를 쓴다."""
    return next((sentence for sentence in sentences if sentence[0] <= start and end <= sentence[1]), None)


def _report_sentences(report: str) -> tuple[tuple[int, int, str], ...]:
    """문장 부호 또는 줄바꿈으로 보고서를 나누고 원문의 정확한 좌표를 보존한다.

    숫자 사이의 마침표는 소수점으로 보고 문장 경계로 사용하지 않는다.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(report):
        is_decimal_point = (
            character == "." and index > 0 and index + 1 < len(report)
            and report[index - 1].isdigit() and report[index + 1].isdigit()
        )
        is_sentence_end = (
            character in ".!?" and not is_decimal_point
            and (index + 1 == len(report) or report[index + 1].isspace())
        )
        if character == "\n" or is_sentence_end:
            spans.append((start, index + 1 if is_sentence_end else index))
            start = index + 1
    if start < len(report):
        spans.append((start, len(report)))
    sentences: list[tuple[int, int, str]] = []
    for raw_start, raw_end in spans:
        value = report[raw_start:raw_end]
        left = len(value) - len(value.lstrip())
        right = len(value.rstrip())
        if right > left:
            start = raw_start + left
            end = raw_start + right
            sentences.append((start, end, report[start:end]))
    return tuple(sentences)
