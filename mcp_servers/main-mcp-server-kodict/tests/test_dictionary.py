"""적재·조회·검토 서비스의 핵심 계약을 확인한다."""

import json
from pathlib import Path
import sqlite3

from main_mcp_server_kodict.dictionary import DictionaryRepository
from main_mcp_server_kodict.morphology import MorphToken, MorphologyAnalyzer
from main_mcp_server_kodict.recommendations import RecommendationCatalog
from main_mcp_server_kodict.review import TextReviewer


def make_database(path: Path) -> None:
    """외부 XLS 없이 조회 계약을 검증할 수 있는 최소 읽기 전용 DB를 만든다."""
    connection = sqlite3.connect(path)
    connection.executescript("""
    CREATE TABLE source_file (id INTEGER PRIMARY KEY, file_name TEXT, source_hash TEXT);
    CREATE TABLE dictionary_entry (entry_id TEXT PRIMARY KEY, source_file_id INTEGER, row_number INTEGER, payload_json TEXT);
    CREATE TABLE surface_form (id INTEGER PRIMARY KEY, entry_id TEXT, source_column TEXT, surface TEXT, normalized_form TEXT);
    CREATE INDEX idx_surface_form_normalized ON surface_form(normalized_form);
    """)
    payload = {"어휘": "가다", "발음": "[가다]", "품사": "동사", "뜻풀이": "어떤 곳으로 움직이다."}
    connection.execute("INSERT INTO source_file VALUES (1, 'sample.xls', 'hash')")
    connection.execute("INSERT INTO dictionary_entry VALUES ('sample.xls:2', 1, 2, ?)", (json.dumps(payload, ensure_ascii=False),))
    connection.execute("INSERT INTO surface_form(entry_id, source_column, surface, normalized_form) VALUES ('sample.xls:2', '어휘', '가다', '가다')")
    connection.commit()
    connection.close()


def add_entry(path: Path, entry_id: str, surface: str, part_of_speech: str = "동사") -> None:
    """추천 대상이 실제 사전 근거를 가져야 하는 테스트용 항목을 추가한다."""
    connection = sqlite3.connect(path)
    payload = {"어휘": surface, "발음": f"[{surface}]", "품사": part_of_speech, "뜻풀이": "테스트 뜻풀이"}
    connection.execute("INSERT INTO dictionary_entry VALUES (?, 1, 3, ?)", (entry_id, json.dumps(payload, ensure_ascii=False)))
    connection.execute("INSERT INTO surface_form(entry_id, source_column, surface, normalized_form) VALUES (?, '어휘', ?, ?)", (entry_id, surface, surface))
    connection.commit()
    connection.close()


def test_lookup_returns_evidence_and_source(tmp_path: Path) -> None:
    """표면형 조회는 원본 행 추적 정보와 사전 근거를 함께 반환한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    entries = DictionaryRepository(database).lookup("가다")
    assert entries[0].entry_id == "sample.xls:2"
    assert entries[0].evidence.source.row_number == 2
    assert entries[0].evidence.definition == "어떤 곳으로 움직이다."


def test_readiness_checks_required_schema_without_full_integrity_scan(tmp_path: Path) -> None:
    """Readiness는 healthcheck마다 전체 DB를 검사하지 않고 필수 스키마만 확인한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    repository = DictionaryRepository(database)

    assert repository.is_ready() is True

    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE dictionary_entry")
    connection.commit()
    connection.close()

    assert repository.is_ready() is False


def test_lookup_query_uses_surface_index(tmp_path: Path) -> None:
    """운영 조회 경로가 표면형 정규화 색인을 사용한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    connection = sqlite3.connect(database)
    plan = connection.execute("EXPLAIN QUERY PLAN SELECT entry_id FROM surface_form WHERE normalized_form = ?", ("가다",)).fetchall()
    connection.close()
    assert "idx_surface_form_normalized" in " ".join(str(item) for item in plan)


class Analyzer:
    """결정적인 검토 서비스 테스트용 형태소 분석기다."""

    def analyze(self, text: str) -> list[MorphToken]:
        return [MorphToken("가다", "가다", "VV", 0, 2), MorphToken("미등재", "미등재", "NNG", 3, 6)]


def test_review_reports_matched_and_not_found(tmp_path: Path) -> None:
    """사전 등재·미등재 단위가 서로 다른 상태로 반환된다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    result = TextReviewer(DictionaryRepository(database), Analyzer(), 10).review("가다 미등재")
    assert [finding["status"] for finding in result["findings"]] == ["matched", "not_found"]
    assert result["findings"][0]["start"] == 0
    assert result["findings"][1]["end"] == 6
    assert result["findings"][0]["source_text"] == "가다"
    assert result["findings"][0]["analysis_form"] == "가다"


def test_review_reuses_one_connection_and_memoizes_repeated_forms(tmp_path: Path) -> None:
    """반복 형태소는 한 요청 안에서 한 번만 DB를 조회한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)

    class RepeatedAnalyzer:
        def analyze(self, text: str) -> list[MorphToken]:
            return [
                MorphToken("가다", "가다", "VV", 0, 2),
                MorphToken("가다", "가다", "VV", 3, 5),
            ]

    repository = DictionaryRepository(database)
    original_lookup = repository.lookup
    calls: list[object] = []

    def tracking_lookup(*args: object, **kwargs: object) -> list[object]:
        calls.append(kwargs.get("connection"))
        return original_lookup(*args, **kwargs)

    repository.lookup = tracking_lookup  # type: ignore[method-assign]
    result = TextReviewer(repository, RepeatedAnalyzer(), 10).review("가다 가다")

    assert len(result["findings"]) == 2
    assert len(calls) == 1
    assert calls[0] is not None


def test_review_limits_findings_and_marks_truncation(tmp_path: Path) -> None:
    """긴 문서는 결과와 후속 사전 조회를 무한히 늘리지 않는다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)

    class ManyTokensAnalyzer:
        def analyze(self, text: str) -> list[MorphToken]:
            return [MorphToken("가다", "가다", "VV", index * 3, index * 3 + 2) for index in range(3)]

    result = TextReviewer(DictionaryRepository(database), ManyTokensAnalyzer(), 10, max_findings=2).review("가다 가다 가다")

    assert len(result["findings"]) == 2
    assert result["warnings"] == [{"code": "findings_truncated", "retryable": False}]


def test_word_units_compute_quoted_state_without_rescanning_prefixes() -> None:
    """인용문 상태는 어절 시작점마다 정확히 보존한다."""
    tokens = [
        MorphToken("가다", "가다", "VV", 0, 2),
        MorphToken("되요", "되요", "VV", 4, 6),
        MorphToken("가다", "가다", "VV", 8, 10),
    ]

    units = TextReviewer._word_units('가다 "되요" 가다', tokens)

    assert [unit.is_quoted for unit in units] == [False, True, False]


def test_review_returns_only_catalog_candidates_with_dictionary_evidence(tmp_path: Path) -> None:
    """명시적 규범 관계와 사전 근거가 모두 있을 때만 추천을 반환한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    catalog_path = tmp_path / "recommendations.json"
    catalog_path.write_text(json.dumps({
        "version": "test-v1",
        "rules": [{
            "rule_id": "TEST-001", "source": "간다아", "replacement": "가다",
            "source_reference": "approved-test-rule", "tags": ["VV"], "confidence": 0.9,
        }],
    }), encoding="utf-8")

    class RecommendationAnalyzer:
        def analyze(self, text: str) -> list[MorphToken]:
            return [MorphToken("간다아", "간다아", "VV", 0, 3)]

    result = TextReviewer(
        DictionaryRepository(database), RecommendationAnalyzer(), 10, RecommendationCatalog.from_path(catalog_path)
    ).review("간다아")
    finding = result["findings"][0]
    candidate = finding["candidates"][0]
    assert finding["status"] == "needs_review"
    assert finding["source_text"] == "간다아"
    assert finding["analysis_form"] == "간다아"
    assert candidate["replacement"] == "가다"
    assert candidate["confidence"] == 0.9
    assert candidate["entry_id"] == "sample.xls:2"
    assert "TEST-001" in candidate["reason"]


def test_review_does_not_suggest_without_dictionary_evidence(tmp_path: Path) -> None:
    """규범 관계의 대상이 사전에 없으면 근거 없는 후보를 만들지 않는다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    catalog_path = tmp_path / "recommendations.json"
    catalog_path.write_text(json.dumps({
        "version": "test-v1",
        "rules": [{
            "rule_id": "TEST-002", "source": "미등재", "replacement": "없는표제어",
            "source_reference": "approved-test-rule", "confidence": 0.9,
        }],
    }), encoding="utf-8")
    result = TextReviewer(DictionaryRepository(database), Analyzer(), 10, RecommendationCatalog.from_path(catalog_path)).review("가다 미등재")
    assert result["findings"][1]["status"] == "not_found"
    assert result["findings"][1]["candidates"] == []


def test_morphology_keeps_source_span_when_analysis_form_differs() -> None:
    """활용형 분석형이 원문과 달라도 유효한 좌표라면 분석을 중단하지 않는다."""
    class KiwiToken:
        form = "하"
        tag = "VV"
        start = 0
        len = 1

    class Kiwi:
        def tokenize(self, text: str) -> list[KiwiToken]:
            return [KiwiToken()]

    analyzer = MorphologyAnalyzer.__new__(MorphologyAnalyzer)
    analyzer._kiwi = Kiwi()
    tokens = analyzer.analyze("합")
    assert tokens == [MorphToken("합", "하", "VV", 0, 1)]


def test_review_looks_up_verb_lemma_when_kiwi_returns_a_stem(tmp_path: Path) -> None:
    """Kiwi의 동사 어간은 원문과 분리한 채 기본형 사전 근거에 연결한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)

    class StemAnalyzer:
        def analyze(self, text: str) -> list[MorphToken]:
            return [MorphToken("간", "가", "VV", 0, 1)]

    result = TextReviewer(DictionaryRepository(database), StemAnalyzer(), 10).review("간")
    assert result["findings"][0]["status"] == "matched"
    assert result["findings"][0]["source_text"] == "간"
    assert result["findings"][0]["candidates"][0]["entry_id"] == "sample.xls:2"


def test_rule_generator_suggests_dictionary_backed_candidate_without_catalog(tmp_path: Path) -> None:
    """매핑 파일 없이도 좁은 활용 규칙과 사전 근거로만 후보를 반환한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    add_entry(database, "sample.xls:3", "되다")

    class TypoAnalyzer:
        def analyze(self, text: str) -> list[MorphToken]:
            return [MorphToken("되요", "되요", "VV", 0, 2)]

    result = TextReviewer(DictionaryRepository(database), TypoAnalyzer(), 10).review("되요")
    finding = result["findings"][0]
    assert finding["status"] == "needs_review"
    assert finding["candidates"][0]["replacement"] == "돼요"
    assert finding["candidates"][0]["entry_id"] == "sample.xls:3"
    assert "ORTH-RULE-001" in finding["candidates"][0]["reason"]


def test_rule_generator_does_not_return_unverified_candidate(tmp_path: Path) -> None:
    """규칙이 후보를 만들어도 사전 근거가 없으면 외부에 노출하지 않는다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)

    class TypoAnalyzer:
        def analyze(self, text: str) -> list[MorphToken]:
            return [MorphToken("되요", "되요", "VV", 0, 2)]

    result = TextReviewer(DictionaryRepository(database), TypoAnalyzer(), 10).review("되요")
    assert result["findings"][0]["status"] == "not_found"
    assert result["findings"][0]["candidates"] == []


def test_real_kiwi_word_rule_runs_before_homonym_lookup(tmp_path: Path) -> None:
    """실제 Kiwi가 분리한 `되요`도 동형 명사 조회보다 먼저 어절 규칙을 적용한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    add_entry(database, "sample.xls:3", "되다")
    result = TextReviewer(DictionaryRepository(database), MorphologyAnalyzer(), 10).review("오늘 되요.")
    finding = next(finding for finding in result["findings"] if finding["source_text"] == "되요")
    assert finding["start"] == 3
    assert finding["end"] == 5
    assert finding["status"] == "needs_review"
    assert finding["candidates"][0]["replacement"] == "돼요"
    assert finding["candidates"][0]["evidence"]["headword"] == "되다"
