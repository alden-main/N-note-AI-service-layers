"""형태소 분석과 사전 근거를 결합해 안전한 검토 결과를 만드는 모듈이다."""

from __future__ import annotations

from dataclasses import dataclass, replace
import sqlite3

from .dictionary import DictionaryRepository
from .models import Entry
from .morphology import MorphToken, MorphologyAnalyzer
from .recommendations import RecommendationCatalog, RuleBasedCandidateGenerator


PARTICLE_TAG_PREFIXES = ("J", "E", "SF", "SP", "SS", "SE", "SO", "SW")
VERB_TAG_PREFIXES = ("VV", "VA")
QUOTE_CHARACTERS = frozenset('"\'“”‘’')


@dataclass(frozen=True)
class WordUnit:
    """공백으로 구분된 원문 어절과 이를 구성하는 Kiwi 토큰이다."""

    source_text: str
    analysis_form: str
    tags: tuple[str, ...]
    start: int
    end: int
    tokens: tuple[MorphToken, ...]
    is_quoted: bool = False


class TextReviewer:
    """사전 미확인 구간을 교정 확정 없이 구조화해 반환하는 서비스다."""

    def __init__(
        self,
        repository: DictionaryRepository,
        analyzer: MorphologyAnalyzer,
        max_candidates: int,
        catalog: RecommendationCatalog | None = None,
        generator: RuleBasedCandidateGenerator | None = None,
        max_findings: int = 1_000,
    ) -> None:
        self.repository = repository
        self.analyzer = analyzer
        self.max_candidates = max_candidates
        self.max_findings = max_findings
        self.catalog = catalog
        self.generator = generator or RuleBasedCandidateGenerator()

    def review(self, text: str) -> dict[str, object]:
        """입력 텍스트의 분석 단위마다 상태와 사전 근거 후보를 반환한다.

        Args:
            text: 검토할 UTF-8 문자열이다. 저장하거나 로그에 남기지 않는다.

        Returns:
            findings와 warnings를 가진 JSON 직렬화 가능한 검토 결과다.
        """
        try:
            tokens = self.analyzer.analyze(text)
        except Exception:
            return {"findings": [{"start": 0, "end": len(text), "source_text": text, "analysis_form": "", "status": "needs_review", "candidates": []}], "warnings": [{"code": "analysis_incomplete", "retryable": False}]}
        findings: list[dict[str, object]] = []
        truncated = False
        # 문서 검토에 필요한 모든 정확 조회가 하나의 읽기 전용 연결을 공유한다.
        # 동일 분석형은 요청 안에서 한 번만 DB에서 읽는다.
        lookup_cache: dict[tuple[str, int], list[Entry]] = {}
        with self.repository.read_session() as connection:
            for unit in self._word_units(text, tokens):
                if len(findings) >= self.max_findings:
                    truncated = True
                    break
                recommendations = self._recommend(unit, connection, lookup_cache)
                if recommendations:
                    findings.append(self._word_finding(unit, recommendations))
                    continue
                for token in unit.tokens:
                    if token.tag.startswith(PARTICLE_TAG_PREFIXES):
                        continue
                    if len(findings) >= self.max_findings:
                        truncated = True
                        break
                    findings.append(self._review_token(token, connection, lookup_cache))
        warnings: list[dict[str, object]] = []
        if truncated:
            warnings.append({"code": "findings_truncated", "retryable": False})
        return {"findings": findings, "warnings": warnings}

    @staticmethod
    def _word_units(text: str, tokens: list[MorphToken]) -> list[WordUnit]:
        """공백과 문장 부호에서 끊어 어절의 원문 범위를 결정한다."""
        units: list[WordUnit] = []
        current: list[MorphToken] = []

        def append_current() -> None:
            if not current:
                return
            start, end = current[0].start, current[-1].end
            units.append(WordUnit(
                source_text=text[start:end],
                analysis_form="+".join(token.analysis_form for token in current),
                tags=tuple(token.tag for token in current),
                start=start,
                end=end,
                tokens=tuple(current),
            ))
            current.clear()

        for token in tokens:
            gap = text[current[-1].end:token.start] if current else ""
            if token.tag.startswith(("SF", "SP", "SS", "SE", "SO", "SW")) or (current and any(char.isspace() for char in gap)):
                append_current()
                if token.tag.startswith(("SF", "SP", "SS", "SE", "SO", "SW")):
                    continue
            current.append(token)
        append_current()
        # 이전에는 각 어절에서 문서 처음부터 인용 부호를 다시 세어 O(어절 수 ×
        # 문서 길이)였다. 시작 지점만 기록하며 한 번 순회한다.
        quoted_at_start: dict[int, bool] = {}
        unit_starts = iter(unit.start for unit in units)
        next_start = next(unit_starts, None)
        quoted = False
        for position, character in enumerate(text):
            while next_start == position:
                quoted_at_start[position] = quoted
                next_start = next(unit_starts, None)
            if character in QUOTE_CHARACTERS:
                quoted = not quoted
        while next_start is not None:
            quoted_at_start[next_start] = quoted
            next_start = next(unit_starts, None)
        return [replace(unit, is_quoted=quoted_at_start.get(unit.start, False)) for unit in units]

    def _lookup(
        self, query: str, limit: int, connection: sqlite3.Connection, cache: dict[tuple[str, int], list[Entry]]
    ) -> list[Entry]:
        """요청 범위 메모이즈로 반복 형태소의 SQLite 왕복을 제거한다."""
        key = (query, limit)
        if key not in cache:
            cache[key] = self.repository.lookup(query, limit, connection=connection)
        return cache[key]

    def _review_token(
        self, token: MorphToken, connection: sqlite3.Connection, cache: dict[tuple[str, int], list[Entry]]
    ) -> dict[str, object]:
        """조사·어미가 아닌 하나의 분석 단위를 조회해 상태와 후보를 결정한다."""
        entries = self._lookup(token.analysis_form, self.max_candidates + 1, connection, cache)
        if not entries and token.tag.startswith(VERB_TAG_PREFIXES) and not token.analysis_form.endswith("다"):
            # Kiwi 동사·형용사 어간은 사전 표제어와 다를 수 있다. 이 복원은
            # 임의 유사어 생성이 아니라 정해진 기본형 규칙이며 사전 조회로 검증한다.
            entries = self._lookup(f"{token.analysis_form}다", self.max_candidates + 1, connection, cache)
        if len(entries) > self.max_candidates:
            return self._finding(token, "needs_review", [], "too_many_candidates")
        if entries:
            return self._finding(token, "matched", entries, replacement=token.source_text, confidence=1.0, reason="사전 표면형 정확 일치")
        return self._finding(token, "not_found", [])

    def _recommend(
        self, unit: WordUnit, connection: sqlite3.Connection, cache: dict[tuple[str, int], list[Entry]]
    ) -> list[tuple[Entry, str, float, str]]:
        """명시적 규범 관계와 사전 근거를 모두 만족한 후보만 순위화한다."""
        if any(tag.startswith("NNP") for tag in unit.tags) or unit.is_quoted:
            return []
        ranked: dict[tuple[str, str], tuple[Entry, str, float, str]] = {}
        rules = self.catalog.rules_for(unit.source_text, unit.analysis_form, unit.tags[0]) if self.catalog else ()
        generated = self.generator.generate(unit.source_text, unit.tags)
        for rule in rules:
            entries = self._lookup(rule.replacement, self.max_candidates + 1, connection, cache)
            if len(entries) > self.max_candidates:
                continue
            for entry in entries:
                # 동일 대체 표기는 가장 높은 명시적 규칙 하나만 유지한다.
                key = (rule.replacement, entry.entry_id)
                reason = f"{rule.rule_id}: {rule.source_reference} (규범 데이터 {self.catalog.version})"
                prior = ranked.get(key)
                if prior is None or rule.confidence > prior[2]:
                    ranked[key] = (entry, rule.replacement, rule.confidence, reason)
        for candidate in generated:
            for evidence_form in candidate.evidence_forms:
                entries = self._lookup(evidence_form, self.max_candidates + 1, connection, cache)
                if len(entries) > self.max_candidates:
                    continue
                for entry in entries:
                    key = (candidate.replacement, entry.entry_id)
                    prior = ranked.get(key)
                    if prior is None or candidate.confidence > prior[2]:
                        ranked[key] = (entry, candidate.replacement, candidate.confidence, candidate.reason)
        return sorted(ranked.values(), key=lambda candidate: (-candidate[2], candidate[0].entry_id))[: self.max_candidates]

    @staticmethod
    def _word_finding(unit: WordUnit, entries: list[tuple[Entry, str, float, str]]) -> dict[str, object]:
        """어절 규칙으로 생성한 후보의 원문 단위 결과를 만든다."""
        token = MorphToken(unit.source_text, unit.analysis_form, unit.tags[0], unit.start, unit.end)
        return TextReviewer._finding(token, "needs_review", entries)

    @staticmethod
    def _finding(
        token: MorphToken,
        status: str,
        entries: list[Entry] | list[tuple[Entry, str, float, str]],
        warning: str | None = None,
        replacement: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
    ) -> dict[str, object]:
        """검토 단위에 공통적인 위치·원문·근거 JSON 구조를 만든다."""
        candidates: list[dict[str, object]] = []
        for entry in entries:
            if isinstance(entry, tuple):
                candidate_entry, candidate_replacement, candidate_confidence, candidate_reason = entry
            else:
                candidate_entry, candidate_confidence, candidate_reason = entry, confidence, reason
                candidate_replacement = replacement or candidate_entry.surface
            candidate = candidate_entry.as_dict()
            candidate.update({"replacement": candidate_replacement, "confidence": candidate_confidence, "reason": candidate_reason})
            candidates.append(candidate)
        finding: dict[str, object] = {
            "start": token.start,
            "end": token.end,
            "source_text": token.source_text,
            "analysis_form": token.analysis_form,
            "status": status,
            "candidates": candidates,
        }
        if warning:
            finding["reason"] = warning
        return finding
