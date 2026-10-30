"""Gateway 전용 network MCP 도구와 감사 경계를 확인한다."""

import io
import json
import unittest

import threading
import urllib.request

from main_mcp_server_document_analysis.mcp_server import McpApplication, create_server
from main_mcp_server_document_analysis.observability import SafeAuditLog


def arguments() -> dict:
    """MCP 도구 호출에 사용할 최소 유효 인자를 반환한다."""
    return {
        "report": {"text": "report", "tenant_id": "tenant-a"},
        "validation_standard": {
            "schema_version": "report-validation-standard/v1", "standard_id": "s", "standard_version": "v1", "status": "approved",
            "usage": {"result_policy": "candidate_only_human_review_required", "mutation_policy": "no_automatic_report_modification"},
            "criteria": [{"criterion_id": "c", "description": "description", "required_elements": []}],
        },
    }


class McpApplicationTest(unittest.TestCase):
    """도구 권한과 민감정보 없는 감사 기록을 검증한다."""

    def test_private_network_mode_does_not_require_claims(self) -> None:
        """표준의 network 신뢰 모드에서는 claim 없이 도구를 호출한다."""
        app = McpApplication()
        self.assertTrue(app.evaluate(arguments())["review_required"])

    def test_audit_does_not_retain_report_text(self) -> None:
        """감사 이벤트 직렬화에 보고서 원문이 포함되지 않는다."""
        app = McpApplication()
        response = app.evaluate(arguments())
        self.assertTrue(response["review_required"])
        self.assertNotIn("report", app._audit.events[-1].__dict__.values())

    def test_guidance_exposes_policy_and_required_elements(self) -> None:
        """LLM이 기준별 근거를 수집할 정책과 요소 설명을 받을 수 있다."""
        app = McpApplication()
        response = app.guidance({"validation_standard": arguments()["validation_standard"]})
        self.assertEqual(response["assessment_policy"]["citation_required_for"], ["satisfied", "insufficient"])
        self.assertEqual(response["criteria"][0]["criterion_id"], "c")

    def test_tools_list_includes_guidance_and_evaluation(self) -> None:
        """MCP discovery는 정책 안내·근거 검색·후보 검증 도구를 모두 노출한다."""
        response = McpApplication()._handle_rpc({"id": 1, "method": "tools/list"})
        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertEqual(names, {"get_validation_guidance", "find_evidence_candidates", "evaluate_report_against_standard"})
        evidence_tool = next(tool for tool in response["result"]["tools"] if tool["name"] == "find_evidence_candidates")
        self.assertIn("1048576 bytes", evidence_tool["description"])
        self.assertEqual(evidence_tool["inputSchema"]["properties"]["report"]["properties"]["text"]["maxLength"], 100_000)

    def test_tools_list_exposes_the_nested_arguments_used_by_each_handler(self) -> None:
        """Gateway는 catalog만으로 원본 standard와 검색·후보 인자를 구성할 수 있다."""
        tools = {tool["name"]: tool["inputSchema"] for tool in McpApplication()._handle_rpc({"id": 1, "method": "tools/list"})["result"]["tools"]}
        self.assertEqual(tools["get_validation_guidance"]["required"], ["validation_standard"])
        standard = tools["get_validation_guidance"]["properties"]["validation_standard"]
        self.assertEqual(standard["required"], ["schema_version"])
        self.assertIn("hmd-analysis-requirement-set/v1", standard["properties"]["schema_version"]["enum"])
        query = tools["find_evidence_candidates"]["properties"]["queries"]["items"]
        self.assertEqual(query["required"], ["criterion_id", "terms"])
        self.assertEqual(query["properties"]["terms"]["minItems"], 1)
        self.assertIn("aliases", query["properties"])
        self.assertEqual(query["properties"]["pattern_ids"]["items"]["enum"], ["korean_duration_expression/v1"])
        candidate = tools["evaluate_report_against_standard"]["properties"]["semantic_assessment"]
        self.assertEqual(candidate["required"], ["criteria"])

    def test_initialize_advertises_streamable_http_tool_capability(self) -> None:
        """Gateway는 MCP initialize에서 서버와 도구 capability를 발견한다."""
        response = McpApplication()._handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(response["result"]["serverInfo"]["name"], "report-validation")
        self.assertEqual(response["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", response["result"]["capabilities"])

    def test_initialize_rejects_unsupported_protocol_version(self) -> None:
        """지원 목록 밖 버전은 안전한 JSON-RPC 오류로 거절한다."""
        response = McpApplication()._handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-01-01"}})
        self.assertEqual(response["error"]["code"], -32602)
        self.assertEqual(response["error"]["message"], "unsupported protocol version")

    def test_initialized_notification_is_accepted_without_response(self) -> None:
        """MCP notification은 수신하되 JSON-RPC 응답을 만들지 않는다."""
        self.assertIsNone(McpApplication()._handle_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_evidence_search_returns_literal_candidates(self) -> None:
        """도구는 외부 AI 검색어에 맞는 문장과 offset을 반환한다."""
        app = McpApplication()
        request = arguments()
        request["report"]["text"] = "First sentence. Target evidence appears here."
        request["queries"] = [{"criterion_id": "c", "terms": ["target"]}]
        response = app.find_evidence_candidates(request)
        self.assertEqual(response["queries"][0]["matches"][0]["quote"], "Target evidence appears here.")

    def test_gateway_can_call_tool_without_authorization_header(self) -> None:
        """격리 network의 Gateway HTTP 호출은 Authorization 없이 성공한다."""
        app = McpApplication()
        request = {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_validation_guidance", "arguments": {"validation_standard": arguments()["validation_standard"]}},
        }
        captured: list[str] = []
        body = json.dumps(request).encode()
        response = app(
            {"PATH_INFO": "/mcp", "REQUEST_METHOD": "POST", "CONTENT_LENGTH": str(len(body)), "wsgi.input": io.BytesIO(body)},
            lambda status, _headers: captured.append(status),
        )
        body = json.loads(b"".join(response))
        self.assertEqual(captured, ["200 OK"])
        self.assertIn("assessment_policy", body["result"]["structuredContent"])
        self.assertFalse(body["result"]["isError"])

    def test_gateway_prefixed_catalog_name_is_routed_to_the_actual_tool(self) -> None:
        """Gateway가 충돌 방지를 위해 붙인 service prefix도 호출 대상에 사용할 수 있다."""
        response = McpApplication()._handle_rpc({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "report-validation__get_validation_guidance",
                "arguments": {"validation_standard": arguments()["validation_standard"]},
            },
        })
        self.assertFalse(response["result"]["isError"])
        self.assertIn("assessment_policy", response["result"]["structuredContent"])

    def test_gateway_tools_call_accepts_extended_evidence_query_and_propagates_request_id(self) -> None:
        """확장 aliases/pattern_ids도 Gateway MCP envelope에서 실제 handler까지 전달된다."""
        app = McpApplication()
        request = {
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {
                "name": "report-validation__find_evidence_candidates",
                "arguments": {
                    **arguments(),
                    "report": {"text": "3개월 동안의 결과분석"},
                    "queries": [{
                        "criterion_id": "c", "terms": ["연구기간"], "aliases": ["결과분석"],
                        "pattern_ids": ["korean_duration_expression/v1"],
                    }],
                },
            },
        }
        body = json.dumps(request).encode()
        response = app(
            {"PATH_INFO": "/mcp", "REQUEST_METHOD": "POST", "CONTENT_LENGTH": str(len(body)), "HTTP_X_REQUEST_ID": "evidence-8", "wsgi.input": io.BytesIO(body)},
            lambda _status, _headers: None,
        )
        result = json.loads(b"".join(response))["result"]
        self.assertFalse(result["isError"])
        self.assertEqual({match["match_strategy"] for match in result["structuredContent"]["queries"][0]["matches"]}, {"alias", "registered_pattern"})
        self.assertEqual(app._audit.events[-1].request_id, "evidence-8")

    def test_tool_error_uses_tool_result_envelope(self) -> None:
        """잘못된 도구 인자는 Gateway가 해석할 isError tool result로 반환한다."""
        response = McpApplication()._handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "missing", "arguments": {}}})
        self.assertTrue(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["error"]["code"], "invalid_input")

    def test_request_id_is_propagated_to_safe_audit_event(self) -> None:
        """Gateway request ID 외에는 감사 이벤트에 원문이 저장되지 않는다."""
        app = McpApplication()
        app._handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_validation_guidance", "arguments": {"validation_standard": arguments()["validation_standard"]}}}, "gateway-request-7")
        event = app._audit.events[-1]
        self.assertEqual(event.request_id, "gateway-request-7")
        self.assertEqual(set(event.__dict__), {"request_id", "service_id", "service_version", "tool_name", "result_code", "standard_version", "duration_ms"})

    def test_health_returns_503_when_required_configuration_is_not_ready(self) -> None:
        """준비되지 않은 필수 설정은 비민감 503 상태로 노출한다."""
        app = McpApplication()
        app._configuration_ready = False
        captured: list[str] = []
        response = app({"PATH_INFO": "/healthz", "REQUEST_METHOD": "GET", "wsgi.input": io.BytesIO()}, lambda status, _headers: captured.append(status))
        self.assertEqual(captured, ["503 Service Unavailable"])
        self.assertEqual(json.loads(b"".join(response)), {"status": "unavailable"})

    def test_not_ready_tool_call_is_retryable_error(self) -> None:
        """준비되지 않은 서비스의 도구 호출은 재시도 가능한 tool result다."""
        app = McpApplication()
        app._configuration_ready = False
        response = app._handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_validation_guidance", "arguments": {"validation_standard": arguments()["validation_standard"]}}})
        self.assertTrue(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["error"]["code"], "upstream_unavailable")
        self.assertTrue(response["result"]["structuredContent"]["error"]["retryable"])

    def test_text_over_limit_is_a_non_retryable_tool_error(self) -> None:
        """보고서 text 제한을 넘으면 retry하지 않는 도구 오류를 반환한다."""
        app = McpApplication()
        app._max_text_chars = 3
        request = arguments()
        request["report"]["text"] = "four"
        response = app._handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "evaluate_report_against_standard", "arguments": request}})
        error = response["result"]["structuredContent"]["error"]
        self.assertTrue(response["result"]["isError"])
        self.assertEqual(error, {"code": "request_too_large", "message": "text input exceeds configured size limit", "retryable": False})

    def test_request_over_limit_is_rejected(self) -> None:
        """무제한 입력을 허용하지 않고 표준 오류를 반환한다."""
        app = McpApplication()
        app._max_request_bytes = 4
        captured: list[str] = []
        response = app({"PATH_INFO": "/mcp", "REQUEST_METHOD": "POST", "CONTENT_LENGTH": "5", "wsgi.input": io.BytesIO(b"12345")}, lambda status, _headers: captured.append(status))
        self.assertEqual(captured, ["413 Payload Too Large"])
        body = json.loads(b"".join(response))
        self.assertTrue(body["result"]["isError"])
        self.assertEqual(body["result"]["structuredContent"]["error"]["code"], "request_too_large")

    def test_missing_content_length_is_rejected_without_reading_input(self) -> None:
        """길이 없는 keep-alive 요청을 최대 본문 크기만큼 대기하며 읽지 않는다."""
        app = McpApplication()
        captured: list[str] = []
        response = app(
            {"PATH_INFO": "/mcp", "REQUEST_METHOD": "POST", "wsgi.input": io.BytesIO(b"")},
            lambda status, _headers: captured.append(status),
        )
        self.assertEqual(captured, ["411 Length Required"])
        self.assertEqual(json.loads(b"".join(response))["error"]["code"], "invalid_input")


class ThreadedServerTest(unittest.TestCase):
    """긴 요청이 다른 요청을 막지 않는지 실제 HTTP로 확인한다."""

    def test_requests_are_served_concurrently(self) -> None:
        barrier = threading.Barrier(2, timeout=3)

        def app(environ, start_response):
            barrier.wait()  # 직렬 서버라면 두 번째 요청이 들어오지 못해 시간 초과가 난다.
            start_response("200 OK", [("Content-Type", "text/plain"), ("Content-Length", "2")])
            return [b"ok"]

        server = create_server("127.0.0.1", 0, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/healthz"
            results: list[bytes] = []
            clients = [threading.Thread(target=lambda: results.append(urllib.request.urlopen(url, timeout=5).read())) for _ in range(2)]
            for client in clients:
                client.start()
            for client in clients:
                client.join()
            self.assertEqual(results, [b"ok", b"ok"])
        finally:
            server.shutdown()
            server.server_close()


class AuditRetentionTest(unittest.TestCase):
    def test_audit_keeps_only_recent_events(self) -> None:
        audit = SafeAuditLog(max_events=2)
        for request_id in ("a", "b", "c"):
            audit.record(request_id=request_id, tool_name="t", result_code="ok", standard_version=None, started_at=0.0)
        self.assertEqual([event.request_id for event in audit.events], ["b", "c"])
