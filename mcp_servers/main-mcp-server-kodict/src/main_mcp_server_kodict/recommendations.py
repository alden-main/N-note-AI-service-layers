"""승인된 표기 관계를 읽고 후보를 만들기 위한 읽기 전용 모델이다.

이 모듈의 관계는 유사도나 언어 모델 생성 결과가 아니다. 각 관계는 배포 전에
검토되는 작은 버전 관리 목록이며, 대상 표기가 사전에서 다시 확인될 때만 후보가
된다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class RecommendationRule:
    """한 비표준 표기와 권장 표기의 명시적 관계다."""

    source: str
    replacement: str
    rule_id: str
    source_reference: str
    tags: tuple[str, ...] = ()
    confidence: float = 0.9

    def applies_to(self, analysis_form: str, tag: str) -> bool:
        """규칙의 품사 조건을 결정적으로 검사한다."""
        return (not self.tags or any(tag.startswith(prefix) for prefix in self.tags)) and bool(analysis_form)


class RecommendationCatalog:
    """버전된 규범 관계만 조회하는 불변 카탈로그다."""

    def __init__(self, version: str, rules: tuple[RecommendationRule, ...]) -> None:
        self.version = version
        self._by_source: dict[str, tuple[RecommendationRule, ...]] = {}
        for rule in rules:
            self._by_source[rule.source] = (*self._by_source.get(rule.source, ()), rule)

    @classmethod
    def from_path(cls, path: Path) -> "RecommendationCatalog":
        """배포 산출물의 형식을 검증해 읽는다."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("version")
        rules = payload.get("rules")
        if not isinstance(version, str) or not isinstance(rules, list):
            raise ValueError("추천 규범 데이터 형식이 올바르지 않습니다.")
        parsed: list[RecommendationRule] = []
        for item in rules:
            if not isinstance(item, dict):
                raise ValueError("추천 규범 규칙 형식이 올바르지 않습니다.")
            source, replacement, rule_id, source_reference = (
                item.get("source"), item.get("replacement"), item.get("rule_id"), item.get("source_reference")
            )
            tags = item.get("tags", [])
            confidence = item.get("confidence", 0.9)
            if not all(isinstance(value, str) and value for value in (source, replacement, rule_id, source_reference)):
                raise ValueError("추천 규범 규칙의 필수 값이 없습니다.")
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("추천 규범 규칙의 품사 조건이 올바르지 않습니다.")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError("추천 규범 규칙의 confidence가 올바르지 않습니다.")
            parsed.append(RecommendationRule(source, replacement, rule_id, source_reference, tuple(tags), float(confidence)))
        return cls(version, tuple(parsed))

    def rules_for(self, source_text: str, analysis_form: str, tag: str) -> tuple[RecommendationRule, ...]:
        """원문 표기와 품사에 명시적으로 일치하는 규칙만 돌려준다."""
        return tuple(rule for rule in self._by_source.get(source_text, ()) if rule.applies_to(analysis_form, tag))


@dataclass(frozen=True)
class GeneratedCandidate:
    """규범 철자 규칙으로 생성했지만 아직 사전 검증 전인 후보다."""

    replacement: str
    evidence_forms: tuple[str, ...]
    confidence: float
    reason: str


class RuleBasedCandidateGenerator:
    """사전 매핑 없이도 적용 가능한 좁은 규범 철자 규칙 집합이다.

    이 클래스는 편집거리나 언어 모델을 사용하지 않는다. 새 규칙은 규칙 ID,
    적용 품사, 검증 사례를 함께 추가해야 하며, 생성 결과는 사전 정확 조회를
    통과하기 전에는 외부 후보가 될 수 없다.
    """

    _doe_yo = re.compile(r"^(?P<stem>.*)되요$")
    _dwaet = re.compile(r"^(?P<stem>.*)됬(?P<ending>다|어요|습니다|지만)?$")

    def generate(self, source_text: str, tags: tuple[str, ...]) -> tuple[GeneratedCandidate, ...]:
        """동사·형용사 활용에 한정한 결정적 후보를 만든다."""
        if not any(tag.startswith(("VV", "VA")) for tag in tags):
            return ()
        candidates: list[GeneratedCandidate] = []
        doe_yo = self._doe_yo.fullmatch(source_text)
        if doe_yo:
            candidates.append(GeneratedCandidate(
                f"{doe_yo.group('stem')}돼요", (f"{doe_yo.group('stem')}되다",), 0.92,
                "ORTH-RULE-001: '되어요'의 준말 표기 규칙",
            ))
        dwaet = self._dwaet.fullmatch(source_text)
        if dwaet:
            candidates.append(GeneratedCandidate(
                f"{dwaet.group('stem')}됐{dwaet.group('ending') or ''}", (f"{dwaet.group('stem')}되다",), 0.92,
                "ORTH-RULE-002: '되었다'의 준말 표기 규칙",
            ))
        return tuple(candidates)
