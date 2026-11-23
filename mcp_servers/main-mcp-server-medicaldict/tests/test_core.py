"""사전 조회, 분석, 승인 이력의 핵심 계약을 검증한다."""

from __future__ import annotations

import sqlite3
import threading
import unittest
import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from mcp import types
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from main_mcp_server_medicaldict.analysis import DocumentAnalyzer, SuggestionValidator
from main_mcp_server_medicaldict.dictionary import DictionaryRepository
from main_mcp_server_medicaldict.models import AnalysisStatus, CandidateType, DecisionStatus, IssueType, MatchType
from main_mcp_server_medicaldict.store import AnalysisStore, StoreError
from main_mcp_server_medicaldict.server import SERVICE_VERSION, create_server


ROOT = Path(__file__).parents[1]


class MedicalDictionaryTests(unittest.TestCase):
    """외부 LLM 없이도 결정적으로 검증 가능한 동작을 모은다."""

    def setUp(self) -> None:
        """현재 활성 사전 저장소를 각 테스트에 제공한다."""
        self.repository = DictionaryRepository(ROOT / "data/dictionary-snapshots")

    def test_exact_korean_and_english_lookup(self) -> None:
        """한글과 영문 정규화 정확 일치 조회가 사전 항목을 반환한다."""
        self.assertTrue(self.repository.find_exact("당 뇨 병"))
        self.assertTrue(self.repository.find_exact("DIABETES MELLITUS"))
        self.assertTrue(self.repository.find_exact("Diabetes-Mellitus"))

    def test_analysis_preserves_source_range(self) -> None:
        """분석 응답의 범위가 호출 텍스트에서 같은 원문을 가리킨다."""
        text = "진단명은 당뇨병입니다."
        result = DocumentAnalyzer(self.repository).analyze(text)
        matches = [item for item in result.suggestions if item.source_text == "당뇨병"]
        self.assertTrue(matches)
        self.assertEqual(text[matches[0].start_offset:matches[0].end_offset], "당뇨병")

    def test_repeated_korean_prefixes_reuse_bounded_dictionary_cache(self) -> None:
        """반복 한글 구간은 같은 prefix마다 SQLite를 다시 열지 않는다."""
        text = "당뇨병" * 100
        analyzer = DocumentAnalyzer(self.repository)
        with patch.object(self.repository, "_connect", wraps=self.repository._connect) as connect:
            first = analyzer.analyze(text)
            first_connection_count = connect.call_count
            second = analyzer.analyze(text)
            second_connection_count = connect.call_count - first_connection_count
        self.assertEqual(
            [(item.start_offset, item.end_offset, item.recommended_term.term_code) for item in first.suggestions],
            [(item.start_offset, item.end_offset, item.recommended_term.term_code) for item in second.suggestions],
        )
        # 첫 분석은 전체 299개 위치가 아닌 고유 두 글자 prefix 3개와 exact
        # lookup만 수행하며, 두 번째 분석은 prefix 결과를 모두 재사용한다.
        self.assertLessEqual(first_connection_count, 4)
        self.assertEqual(second_connection_count, 1)

    def test_analysis_limits_result_size_before_serialization(self) -> None:
        """반복 용어가 있어도 MCP 응답이 무제한으로 커지지 않는다."""
        analyzer = DocumentAnalyzer(self.repository, max_suggestions=10)
        result = analyzer.analyze("당뇨병" * 20)
        self.assertEqual(len(result.suggestions), 10)
        self.assertEqual(result.status, AnalysisStatus.PARTIALLY_COMPLETED)
        self.assertEqual(result.error_code, "result_limit_exceeded")

    def test_validator_rejects_invalid_term_code_and_range(self) -> None:
        """환각 사전 코드와 문서 밖 범위의 LLM 응답은 공개하지 않는다."""
        validator = SuggestionValidator(self.repository)
        invalid_code = {"start_offset": 0, "end_offset": 3, "review_status": "review_recommended", "confidence": "high", "reason": "x", "recommended_term": {"source_id": "x", "term_code": "x", "concept_code": "x"}}
        invalid_range = {"start_offset": 0, "end_offset": 99, "review_status": "review_recommended", "confidence": "high", "reason": "x", "recommended_term": {"source_id": "x", "term_code": "x", "concept_code": "x"}}
        self.assertIsNone(validator.validate(invalid_code, "당뇨병"))
        self.assertIsNone(validator.validate(invalid_range, "당뇨병"))

    def test_validated_host_suggestion_can_be_stored(self) -> None:
        """접속한 LLM 호스트의 제안은 서버 검증 후에만 분석 이력에 추가된다."""
        term = self.repository.find_exact("당뇨병")[0]
        payload = {"start_offset": 0, "end_offset": 3, "review_status": "review_recommended", "confidence": "medium", "reason": "테스트", "recommended_term": {"source_id": term.source_id, "term_code": term.term_code, "concept_code": term.concept_code}}
        suggestion = SuggestionValidator(self.repository).validate(payload, "당뇨병")
        result = DocumentAnalyzer(self.repository).analyze("당뇨병")
        with TemporaryDirectory() as directory:
            store = AnalysisStore(Path(directory) / "history.sqlite")
            store.save(result)
            store.append_suggestions(result.analysis_id, [suggestion])
            self.assertEqual(len(store.get(result.analysis_id).suggestions), len(result.suggestions) + 1)

    def test_store_has_no_document_source_and_decision_is_idempotent(self) -> None:
        """이력 DB에는 원문 열이 없고 같은 승인 재시도는 같은 결과를 반환한다."""
        result = DocumentAnalyzer(self.repository).analyze("당뇨병")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite"
            store = AnalysisStore(path)
            store.save(result)
            columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(suggestions)")}
            self.assertNotIn("source_text", columns)
            suggestion = store.get(result.analysis_id).suggestions[0]
            self.assertIsNone(suggestion.source_text)
            first = store.decide(result.analysis_id, suggestion.suggestion_id, DecisionStatus.APPROVED, suggestion.recommended_term.term_code, "reviewer")
            second = store.decide(result.analysis_id, suggestion.suggestion_id, DecisionStatus.APPROVED, suggestion.recommended_term.term_code, "reviewer")
            self.assertEqual(first.decision, second.decision)
            with self.assertRaises(StoreError):
                store.decide(result.analysis_id, suggestion.suggestion_id, DecisionStatus.REJECTED, None, "reviewer")

    def test_document_candidates_cover_particles_typo_and_chunk_offsets(self) -> None:
        """조사 결합·오탈자 후보는 전체 문서 기준 Unicode code point 범위를 유지한다."""
        text = "당뇨병은 Diabetes Mellitus를 앓고 당뇨뼝 의심"
        offset = 17
        result = DocumentAnalyzer(self.repository).analyze_document(text, offset)
        observed = {(item.candidate_type, text[item.start_offset - offset:item.end_offset - offset]) for item in result.candidates}
        self.assertIn((CandidateType.VARIANT_MATCH, "당뇨병은"), observed)
        self.assertIn((CandidateType.VARIANT_MATCH, "Diabetes Mellitus를"), observed)
        self.assertIn((CandidateType.UNKNOWN_OR_POSSIBLE_TYPO, "당뇨뼝"), observed)
        for item in result.candidates:
            self.assertGreaterEqual(item.start_offset, offset)
            self.assertLessEqual(item.end_offset, offset + len(text))

    def test_search_is_limited_and_typo_returns_similar_result(self) -> None:
        """유사 검색은 제한된 결과만 반환하고 오탈자 후보에 사전 코드를 제공한다."""
        result = self.repository.search("당뇨뼝", limit=3)
        self.assertLessEqual(len(result.results), 3)
        self.assertTrue(result.results)
        self.assertEqual(result.dictionary_version, self.repository.version)
        self.assertIs(self.repository.get_search_result(result.search_result_id), result)
        aliases = self.repository.search("DM", limit=10)
        self.assertTrue(any(hit.match_type is MatchType.SYNONYM_OR_ABBREVIATION for hit in aliases.results))

    def test_document_analysis_does_not_evict_issued_search_results(self) -> None:
        """문서 분석의 내부 오타 검색은 사용자가 받은 search_result_id를 밀어내지 않는다."""
        issued = self.repository.search("당뇨뼝", limit=3)
        with self.repository._search_results_lock:
            stored_before = len(self.repository._search_results)
        DocumentAnalyzer(self.repository).analyze_document("당뇨뼝 강경변증 " * 600, 0)
        self.assertIs(self.repository.get_search_result(issued.search_result_id), issued)
        with self.repository._search_results_lock:
            self.assertEqual(len(self.repository._search_results), stored_before)

    def test_extended_mcp_tools_are_registered(self) -> None:
        """Gateway catalog의 모든 도구가 이름·설명·객체 입력 스키마를 제공한다."""
        with TemporaryDirectory() as directory:
            server = create_server(ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite")
            tools = asyncio.run(server.list_tools())
            names = {tool.name for tool in tools}
        self.assertTrue({"analyze_document_terms", "search_medical_terms", "submit_review_suggestions"}.issubset(names))
        self.assertTrue(all(tool.name and isinstance(tool.description, str) and tool.description and isinstance(tool.inputSchema, dict) for tool in tools))

    def test_server_accepts_network_listener_configuration(self) -> None:
        """원격 MCP 배포는 FastMCP의 Streamable HTTP 리스너 설정을 사용한다."""
        with TemporaryDirectory() as directory:
            server = create_server(
                ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite",
                host="0.0.0.0", port=9100,
            )
        self.assertEqual(server.settings.host, "0.0.0.0")
        self.assertEqual(server.settings.port, 9100)

    def test_healthz_returns_non_sensitive_ready_status(self) -> None:
        """Gateway의 상태 확인은 MCP 도구나 사전 내용을 노출하지 않는다."""
        with TemporaryDirectory() as directory:
            server = create_server(ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite")

            async def request_health() -> httpx.Response:
                transport = httpx.ASGITransport(app=server.streamable_http_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    return await client.get("/healthz")

            response = asyncio.run(request_health())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service_id": "medic"})

    def test_gateway_tool_contract_returns_structured_success_and_safe_errors(self) -> None:
        """Gateway는 성공·입력 오류·과대 요청을 MCP tool result로 구분할 수 있다."""
        with TemporaryDirectory() as directory:
            server = create_server(
                ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite", max_text_chars=3,
            )
            handler = server._mcp_server.request_handlers[types.CallToolRequest]

            async def call(arguments: dict) -> types.CallToolResult:
                response = await handler(types.CallToolRequest(
                    params=types.CallToolRequestParams(name="analyze_document_terms", arguments=arguments),
                ))
                self.assertIsInstance(response.root, types.CallToolResult)
                return response.root

            success = asyncio.run(call({"text": "당뇨병"}))
            self.assertFalse(success.isError)
            self.assertNotIn("result", success.structuredContent or {})  # {"result": ...}로 감싸지 않는다
            candidate = success.structuredContent["candidates"][0]
            self.assertTrue({"start_offset", "end_offset", "candidate_type"}.issubset(candidate))

            too_large = asyncio.run(call({"text": "가나다라마바사"}))
            self.assertTrue(too_large.isError)
            self.assertEqual(too_large.structuredContent["error"], {
                "code": "request_too_large", "message": "text는 3자 이하여야 합니다.", "retryable": False,
            })

            invalid = asyncio.run(call({"text": 1}))
            self.assertTrue(invalid.isError)
            self.assertEqual(invalid.structuredContent["error"]["code"], "invalid_input")

    def test_unready_dictionary_returns_503_health_and_tool_error(self) -> None:
        """활성 사전이 없으면 health와 도구가 준비되지 않음만 노출한다."""
        with TemporaryDirectory() as directory:
            root = Path(directory) / "missing-snapshot"
            server = create_server(root, Path(directory) / "history.sqlite")

            async def request_health() -> httpx.Response:
                transport = httpx.ASGITransport(app=server.streamable_http_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    return await client.get("/healthz")

            health = asyncio.run(request_health())
            self.assertEqual(health.status_code, 503)
            self.assertEqual(health.json(), {"status": "unavailable", "service_id": "medic"})
            handler = server._mcp_server.request_handlers[types.CallToolRequest]
            result = asyncio.run(handler(types.CallToolRequest(
                params=types.CallToolRequestParams(name="analyze_document_terms", arguments={"text": "당뇨병"}),
            ))).root
            self.assertTrue(result.isError)
            self.assertEqual(result.structuredContent["error"]["code"], "dictionary_unavailable")

    def test_installed_mcp_supports_gateway_protocol_version(self) -> None:
        """사용 중인 MCP 런타임이 Gateway의 2025-06-18 세션 협상을 지원한다."""
        self.assertIn("2025-06-18", SUPPORTED_PROTOCOL_VERSIONS)

    def test_first_syllable_error_is_found_by_phonetic_search(self) -> None:
        """접두어가 다른 오인식도 받침·유사 자모가 같으면 올바른 용어를 찾는다."""
        names = [hit.term.kor_name for hit in self.repository.search("강경변증").results]
        self.assertIn("간경변증", names)

    def test_typo_candidate_drops_particle_and_keeps_source_range(self) -> None:
        """오타 후보 범위는 조사·어미를 뺀 원문을 가리키고 사전 용어를 제안한다."""
        text = "과거력은 강경변증으로 진단"
        result = DocumentAnalyzer(self.repository).analyze_document(text)
        typos = {text[item.start_offset:item.end_offset]: [term.kor_name for term in item.terms] for item in result.candidates if item.candidate_type is CandidateType.UNKNOWN_OR_POSSIBLE_TYPO}
        self.assertIn("간경변증", typos.get("강경변증", []))

    def test_loose_lookalikes_are_not_typo_candidates(self) -> None:
        """글자 일부만 닮은 일반 표현은 의학용어 오타로 제안하지 않는다."""
        for text in ("이력이 있습니다", "간도 기록", "시스토이며", "상부프 통증", "상공묘 농거"):
            result = DocumentAnalyzer(self.repository).analyze_document(text)
            self.assertEqual([], [item for item in result.candidates if item.candidate_type is CandidateType.UNKNOWN_OR_POSSIBLE_TYPO], text)

    def test_registered_term_with_ending_is_not_a_typo(self) -> None:
        """사전에 있는 용어에 어미가 붙었을 뿐이면 비슷한 다른 용어를 제안하지 않는다."""
        result = DocumentAnalyzer(self.repository).analyze_document("진단은 당뇨병입니다")
        self.assertEqual([], [item for item in result.candidates if item.candidate_type is CandidateType.UNKNOWN_OR_POSSIBLE_TYPO])

    def test_review_suggestion_requires_owned_candidate_and_matching_code(self) -> None:
        """확장 제안은 같은 분석 후보 범위와 그 후보가 허용한 코드에만 연결된다."""
        text = "당뇨뼝"
        result = DocumentAnalyzer(self.repository).analyze_document(text)
        candidate = next(item for item in result.candidates if item.candidate_type is CandidateType.UNKNOWN_OR_POSSIBLE_TYPO)
        with TemporaryDirectory() as directory:
            store = AnalysisStore(Path(directory) / "history.sqlite")
            store.save(result)
            restored = store.get_candidate(result.analysis_id, candidate.candidate_id)
            self.assertIsNotNone(restored)
            payload = {
                "start_offset": 0, "end_offset": 3, "review_status": "review_recommended",
                "confidence": "medium", "reason": "표기 변형 검토", "issue_type": IssueType.POSSIBLE_TYPO.value,
                "candidate_id": candidate.candidate_id,
                "recommended_term": restored.terms[0].to_dict(),
            }
            accepted = SuggestionValidator(self.repository).validate_review(payload, text, restored.terms, restored)
            self.assertIsNotNone(accepted)
            payload["end_offset"] = 2
            self.assertIsNone(SuggestionValidator(self.repository).validate_review(payload, text, restored.terms, restored))

    def test_document_analysis_reuses_one_read_connection_per_thread(self) -> None:
        """문서 분석은 조회마다 SQLite 연결을 열지 않고 thread의 읽기 전용 연결을 재사용한다."""
        repository = DictionaryRepository(ROOT / "data/dictionary-snapshots")
        repository.load_phonetic_index()
        text = "환자 당뇨병 있고 고혈압으로 약 복용 중입니다. 강경변증 의심, 수술 후 통증 호소함."
        with patch("main_mcp_server_medicaldict.dictionary.sqlite3.connect", wraps=sqlite3.connect) as connect:
            first = DocumentAnalyzer(repository).analyze_document(text)
            second = DocumentAnalyzer(repository).analyze_document(text)
        self.assertEqual(connect.call_count, 0)  # 색인을 만들 때 연 연결을 그대로 쓴다.
        self.assertEqual(
            [(item.start_offset, item.end_offset, item.candidate_type) for item in first.candidates],
            [(item.start_offset, item.end_offset, item.candidate_type) for item in second.candidates],
        )

    def test_health_responds_while_a_tool_call_is_running(self) -> None:
        """도구는 worker thread에서 실행되므로 긴 분석이 /healthz와 다른 호출을 막지 않는다."""
        started, release = threading.Event(), threading.Event()
        original = DocumentAnalyzer.analyze_document

        def slow_analyze(analyzer: DocumentAnalyzer, text: str, document_offset: int = 0):
            started.set()
            release.wait(timeout=10)
            return original(analyzer, text, document_offset)

        with TemporaryDirectory() as directory, patch.object(DocumentAnalyzer, "analyze_document", slow_analyze):
            server = create_server(ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite")
            handler = server._mcp_server.request_handlers[types.CallToolRequest]

            async def scenario() -> tuple[int, bool]:
                call = asyncio.ensure_future(handler(types.CallToolRequest(
                    params=types.CallToolRequestParams(name="analyze_document_terms", arguments={"text": "당뇨병"}),
                )))
                while not started.is_set():
                    await asyncio.sleep(0.01)
                transport = httpx.ASGITransport(app=server.streamable_http_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    health = await asyncio.wait_for(client.get("/healthz"), timeout=5)
                still_running = not call.done()
                release.set()
                result = (await call).root
                self.assertFalse(result.isError)
                return health.status_code, still_running

            status, still_running = asyncio.run(scenario())
        self.assertEqual(status, 200)
        self.assertTrue(still_running)

    def test_exact_analysis_looks_up_each_expression_once(self) -> None:
        """정확 일치 분석도 겹치는 n-gram의 같은 표현을 한 번만 조회한다."""
        with patch.object(self.repository, "find_exact", wraps=self.repository.find_exact) as find_exact:
            DocumentAnalyzer(self.repository).analyze("당뇨병 당뇨병 당뇨병 고혈압 당뇨병")
        counts = Counter(call.args[0] for call in find_exact.call_args_list)
        self.assertTrue(counts)
        self.assertEqual(max(counts.values()), 1)

    def test_store_reuses_one_connection_per_thread_and_rolls_back_failures(self) -> None:
        """이력 저장소는 thread별 연결을 재사용하고, 실패한 쓰기는 트랜잭션 단위로 되돌린다."""
        result = DocumentAnalyzer(self.repository).analyze("당뇨병")
        with TemporaryDirectory() as directory:
            store = AnalysisStore(Path(directory) / "history.sqlite")
            with patch("main_mcp_server_medicaldict.store.sqlite3.connect", wraps=sqlite3.connect) as connect:
                store.save(result)
                store.get(result.analysis_id)
                store.check_ready()
            self.assertEqual(connect.call_count, 0)
            with self.assertRaises(sqlite3.IntegrityError):
                store.save(result)  # 같은 analysis_id 재저장은 전체가 롤백된다.
            self.assertEqual(len(store.get(result.analysis_id).suggestions), len(result.suggestions))
            other: list[int] = []
            thread = threading.Thread(target=lambda: other.append(id(store._connect())))
            thread.start()
            thread.join()
            self.assertNotEqual(other[0], id(store._connect()))
            # 다른 thread의 연결에서도 커밋된 쓰기가 보인다.
            seen: list[bool] = []
            thread = threading.Thread(target=lambda: seen.append(store.get(result.analysis_id) is not None))
            thread.start()
            thread.join()
            self.assertEqual(seen, [True])

    def test_tool_calls_do_not_recheck_readiness_after_initialization(self) -> None:
        """준비 확인(DB 질의)은 /healthz에서만 하고 도구 호출은 초기화된 구성 요소를 재사용한다."""
        with TemporaryDirectory() as directory, \
                patch.object(DictionaryRepository, "check_ready") as dictionary_ready, \
                patch.object(AnalysisStore, "check_ready") as store_ready, \
                patch.object(AnalysisStore, "__init__", side_effect=AnalysisStore.__init__, autospec=True) as store_init:
            server = create_server(ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite")
            handler = server._mcp_server.request_handlers[types.CallToolRequest]

            async def calls() -> list[types.CallToolResult]:
                return [
                    (await handler(types.CallToolRequest(params=types.CallToolRequestParams(name=name, arguments=arguments)))).root
                    for name, arguments in (
                        ("analyze_document_terms", {"text": "당뇨병"}),
                        ("search_medical_terms", {"query": "당뇨병"}),
                        ("get_analysis", {"analysis_id": "missing"}),
                    )
                ]

            results = asyncio.run(calls())
        self.assertFalse(results[0].isError)
        self.assertFalse(results[1].isError)
        self.assertEqual(results[2].structuredContent["error"]["code"], "analysis_not_found")
        self.assertEqual((dictionary_ready.call_count, store_ready.call_count, store_init.call_count), (0, 0, 1))

    def test_error_envelope_matches_gateway_contract(self) -> None:
        """오류는 kodict·document-analysis와 같이 structuredContent와 content text 모두 최상위 ``error``다."""
        with TemporaryDirectory() as directory:
            server = create_server(ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite")
            handler = server._mcp_server.request_handlers[types.CallToolRequest]
            result = asyncio.run(handler(types.CallToolRequest(
                params=types.CallToolRequestParams(name="get_analysis", arguments={"analysis_id": "missing"}),
            ))).root
        self.assertTrue(result.isError)
        expected = {"code": "analysis_not_found", "message": "분석을 찾을 수 없습니다.", "retryable": False}
        self.assertEqual(result.structuredContent, {"error": expected})
        self.assertEqual(json.loads(result.content[0].text), {"error": expected})

    def test_tools_publish_no_output_schema_and_success_is_not_wrapped(self) -> None:
        """도구는 출력 스키마를 공개하지 않고(structured_output=False), 성공 결과를 그대로 structuredContent에 담는다."""
        with TemporaryDirectory() as directory:
            server = create_server(ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite")
            tools = asyncio.run(server.list_tools())
            handler = server._mcp_server.request_handlers[types.CallToolRequest]
            result = asyncio.run(handler(types.CallToolRequest(
                params=types.CallToolRequestParams(name="search_medical_terms", arguments={"query": "당뇨", "limit": 2}),
            ))).root
        self.assertEqual({tool.name: tool.outputSchema for tool in tools if tool.outputSchema is not None}, {})
        self.assertFalse(result.isError)
        self.assertTrue({"search_result_id", "dictionary_version", "results"}.issubset(result.structuredContent))
        self.assertEqual(json.loads(result.content[0].text), result.structuredContent)

    def test_audit_log_uses_gateway_request_id_header(self) -> None:
        """감사 로그는 Gateway의 X-Request-ID를 JSON-RPC id보다 우선하고 패키지 버전을 남긴다."""
        with TemporaryDirectory() as directory:
            server = create_server(ROOT / "data/dictionary-snapshots", Path(directory) / "history.sqlite")

            async def call() -> httpx.Response:
                app = server.streamable_http_app()
                async with server.session_manager.run():
                    transport = httpx.ASGITransport(app=app)
                    # 기본 DNS rebinding 보호가 localhost Host만 허용한다.
                    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
                        return await client.post(
                            "/mcp",
                            json={"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "get_analysis", "arguments": {"analysis_id": "missing"}}},
                            headers={"Accept": "application/json, text/event-stream", "X-Request-ID": "gw-123\nforged", "MCP-Protocol-Version": "2025-06-18"},
                        )

            with self.assertLogs("medical_term_review.audit", level=logging.INFO) as logs:
                response = asyncio.run(call())
        self.assertEqual(response.status_code, 200)
        self.assertIn("request_id=gw-123forged tool=get_analysis result_code=analysis_not_found", logs.output[0])
        self.assertIn(f"service_version={SERVICE_VERSION}", logs.output[0])
        self.assertNotEqual(SERVICE_VERSION, "unknown")
