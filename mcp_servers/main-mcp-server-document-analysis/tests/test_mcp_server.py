"""Gateway 전용 network MCP 도구와 감사 경계를 확인한다."""

import importlib.metadata
import json
import os
import socket
import threading
import time
import unittest
import unittest.mock
import urllib.request

import uvicorn
from starlette.testclient import TestClient

from main_mcp_server_document_analysis.mcp_server import McpApplication, create_asgi_app
from main_mcp_server_document_analysis.observability import SERVICE_VERSION, SafeAuditLog


def arguments() -> dict:
    """MCP 도구 호출에 사용할 최소 유효 인자를 반환한다."""
    return {
        "report": {"text": "confidential report text", "tenant_id": "tenant-a"},
        "validation_standard": {
            "schema_version": "report-validation-standard/v1", "standard_id": "s", "standard_version": "v1", "status": "approved",
            "usage": {"result_policy": "candidate_only_human_review_required", "mutation_policy": "no_automatic_report_modification"},
            "criteria": [{"criterion_id": "c", "description": "description", "required_elements": []}],
        },
    }


ACCEPT = {"Accept": "application/json, text/event-stream"}


def rpc(app: McpApplication, payload: dict, headers: dict | None = None):
    """SDK Streamable HTTP 서버(/mcp)로 JSON-RPC를 보낸다. lifespan(세션 관리자)을 띄우려고 with로 연다."""
    with TestClient(create_asgi_app(app)) as client:
        return client.post("/mcp", json=payload, headers={**ACCEPT, **(headers or {})})


def tool_call(name: str, arguments: dict, message_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


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
        self.assertNotIn("confidential report text", app._audit.events[-1].__dict__.values())

    def test_guidance_exposes_policy_and_required_elements(self) -> None:
        """LLM이 기준별 근거를 수집할 정책과 요소 설명을 받을 수 있다."""
        app = McpApplication()
        response = app.guidance({"validation_standard": arguments()["validation_standard"]})
        self.assertEqual(response["assessment_policy"]["citation_required_for"], ["satisfied", "insufficient"])
        self.assertEqual(response["criteria"][0]["criterion_id"], "c")

    def test_tools_list_includes_guidance_and_evaluation(self) -> None:
        """MCP discovery는 정책 안내·근거 검색·후보 검증 도구를 모두 노출한다."""
        listed = rpc(McpApplication(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()["result"]["tools"]
        names = {tool["name"] for tool in listed}
        self.assertEqual(names, {"get_validation_guidance", "find_evidence_candidates", "evaluate_report_against_standard"})
        evidence_tool = next(tool for tool in listed if tool["name"] == "find_evidence_candidates")
        self.assertIn("1048576 bytes", evidence_tool["description"])
        self.assertEqual(evidence_tool["inputSchema"]["properties"]["report"]["properties"]["text"]["maxLength"], 100_000)

    def test_tools_list_exposes_the_nested_arguments_used_by_each_handler(self) -> None:
        """Gateway는 catalog만으로 원본 standard와 검색·후보 인자를 구성할 수 있다."""
        tools = {tool["name"]: tool["inputSchema"] for tool in McpApplication().tools()}
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
        params = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "gateway", "version": "0"}}
        response = rpc(McpApplication(), {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params}).json()
        self.assertEqual(response["result"]["serverInfo"]["name"], "report")
        self.assertEqual(response["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", response["result"]["capabilities"])

    def test_initialize_negotiates_away_from_an_unknown_protocol_version(self) -> None:
        """모르는 버전을 요청하면 MCP 버전 협상대로 서버가 지원하는 버전으로 답한다(Gateway가 불일치를 거부한다)."""
        params = {"protocolVersion": "2024-01-01", "capabilities": {}, "clientInfo": {"name": "gateway", "version": "0"}}
        response = rpc(McpApplication(), {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params}).json()
        self.assertNotEqual(response["result"]["protocolVersion"], "2024-01-01")

    def test_initialized_notification_is_accepted_without_response(self) -> None:
        """MCP notification은 수신하되 JSON-RPC 응답을 만들지 않는다(202, 본문 없음)."""
        response = rpc(McpApplication(), {"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual((response.status_code, response.content), (202, b""))

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
        response = rpc(app, request)
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(body["result"]["content"][0]["text"]), body["result"]["structuredContent"])
        self.assertIn("assessment_policy", body["result"]["structuredContent"])
        self.assertFalse(body["result"]["isError"])

    def test_prefixed_catalog_name_is_unknown_tool(self) -> None:
        """Gateway가 service prefix를 떼고 호출하므로 접두사 붙은 이름은 알 수 없는 도구다."""
        response = McpApplication().call_tool("report__get_validation_guidance", {"validation_standard": arguments()["validation_standard"]})
        self.assertTrue(response["isError"])
        self.assertEqual(response["structuredContent"]["error"]["code"], "invalid_input")

    def test_gateway_tools_call_accepts_extended_evidence_query_and_propagates_request_id(self) -> None:
        """확장 aliases/pattern_ids도 Gateway MCP envelope에서 실제 handler까지 전달된다."""
        app = McpApplication()
        request = {
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {
                "name": "find_evidence_candidates",
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
        result = rpc(app, request, {"X-Request-ID": "evidence-8"}).json()["result"]
        self.assertFalse(result["isError"])
        self.assertEqual({match["match_strategy"] for match in result["structuredContent"]["queries"][0]["matches"]}, {"alias", "registered_pattern"})
        self.assertEqual(app._audit.events[-1].request_id, "evidence-8")

    def test_tool_error_uses_tool_result_envelope(self) -> None:
        """잘못된 도구 인자는 Gateway가 해석할 isError tool result로 반환한다."""
        result = rpc(McpApplication(), tool_call("missing", {})).json()["result"]
        self.assertTrue(result["isError"])
        self.assertEqual(result["structuredContent"]["error"]["code"], "invalid_input")
        # SDK 입력 검증을 끄고 도구가 직접 검증하므로, 계약 위반 인자도 같은 오류 형식으로 돌아온다.
        result = rpc(McpApplication(), tool_call("evaluate_report_against_standard", {"report": "not an object"})).json()["result"]
        self.assertEqual((result["isError"], result["structuredContent"]["error"]["code"]), (True, "invalid_input"))

    def test_request_id_is_propagated_to_safe_audit_event(self) -> None:
        """Gateway request ID 외에는 감사 이벤트에 원문이 저장되지 않는다."""
        app = McpApplication()
        rpc(app, tool_call("get_validation_guidance", {"validation_standard": arguments()["validation_standard"]}), {"X-Request-ID": "gateway-request-7"})
        event = app._audit.events[-1]
        self.assertEqual(event.request_id, "gateway-request-7")
        self.assertEqual(set(event.__dict__), {"request_id", "service_id", "service_version", "tool_name", "result_code", "standard_version", "duration_ms"})

    def test_health_returns_503_when_required_configuration_is_not_ready(self) -> None:
        """준비되지 않은 필수 설정은 비민감 503 상태로 노출한다."""
        app = McpApplication()
        app._configuration_ready = False
        with TestClient(create_asgi_app(app)) as client:
            response = client.get("/healthz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unavailable"})

    def test_not_ready_tool_call_is_retryable_error(self) -> None:
        """준비되지 않은 서비스의 도구 호출은 재시도 가능한 tool result다."""
        app = McpApplication()
        app._configuration_ready = False
        response = app.call_tool("get_validation_guidance", {"validation_standard": arguments()["validation_standard"]})
        self.assertTrue(response["isError"])
        self.assertEqual(response["structuredContent"]["error"]["code"], "upstream_unavailable")
        self.assertTrue(response["structuredContent"]["error"]["retryable"])

    def test_text_over_limit_is_a_non_retryable_tool_error(self) -> None:
        """보고서 text 제한을 넘으면 retry하지 않는 도구 오류를 반환한다."""
        app = McpApplication()
        app._max_text_chars = 3
        request = arguments()
        request["report"]["text"] = "four"
        response = app.call_tool("evaluate_report_against_standard", request)
        error = response["structuredContent"]["error"]
        self.assertTrue(response["isError"])
        self.assertEqual(error, {"code": "request_too_large", "message": "text input exceeds configured size limit", "retryable": False})

    def test_request_over_limit_is_rejected(self) -> None:
        """무제한 입력을 허용하지 않고 표준 오류를 반환한다."""
        app = McpApplication()
        app._max_request_bytes = 4
        with TestClient(create_asgi_app(app)) as client:
            response = client.post("/mcp", content=b'{"jsonrpc": "2.0"}', headers={**ACCEPT, "Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_unknown_path_and_malformed_json_are_rejected(self) -> None:
        with TestClient(create_asgi_app(McpApplication())) as client:
            self.assertEqual(client.get("/other").status_code, 404)
            response = client.post("/mcp", content=b"{not json", headers={**ACCEPT, "Content-Type": "application/json"})
            self.assertEqual(response.status_code, 400)


class UvicornServerTest(unittest.TestCase):
    """긴 도구 실행이 다른 요청을 막지 않는지 실제 uvicorn 서버로 확인한다."""

    def test_requests_are_served_concurrently(self) -> None:
        barrier = threading.Barrier(2, timeout=3)

        class Blocking(McpApplication):
            def call_tool(self, name, arguments, request_id=None):
                barrier.wait()  # 요청을 하나씩 처리한다면 두 번째 요청이 들어오지 못해 시간 초과가 난다.
                return super().call_tool(name, arguments, request_id)

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_asgi_app(Blocking()), host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.02)
            body = json.dumps(tool_call("get_validation_guidance", {"validation_standard": arguments()["validation_standard"]})).encode()

            def call() -> None:
                request = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=body, headers={"Content-Type": "application/json", **ACCEPT})
                results.append(json.loads(urllib.request.urlopen(request, timeout=5).read())["id"])

            results: list[int] = []
            clients = [threading.Thread(target=call) for _ in range(2)]
            for client in clients:
                client.start()
            for client in clients:
                client.join()
            self.assertEqual(results, [1, 1])
        finally:
            server.should_exit = True
            thread.join(timeout=5)


class AuditRetentionTest(unittest.TestCase):
    def test_audit_keeps_only_recent_events(self) -> None:
        audit = SafeAuditLog(max_events=2)
        for request_id in ("a", "b", "c"):
            audit.record(request_id=request_id, tool_name="t", result_code="ok", standard_version=None, started_at=0.0)
        self.assertEqual([event.request_id for event in audit.events], ["b", "c"])

    def test_audit_event_is_also_emitted_as_log_without_report_text(self) -> None:
        """감사 이벤트는 같은 허용 필드만 logging으로도 내보낸다."""
        audit = SafeAuditLog()
        with self.assertLogs("main_mcp_server_document_analysis.audit", level="INFO") as captured:
            audit.record(request_id="r-1", tool_name="t", result_code="ok", standard_version="v1", started_at=0.0)
        logged = json.loads(captured.records[0].getMessage())
        self.assertEqual(logged, audit.events[-1].__dict__)


class ConfigurationTest(unittest.TestCase):
    """입력 한도 환경 설정과 서비스 버전을 확인한다."""

    def test_invalid_limit_environment_makes_service_not_ready(self) -> None:
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value), unittest.mock.patch.dict(os.environ, {"MCP_MAX_TEXT_CHARS": value}):
                self.assertFalse(McpApplication().is_ready())
        with unittest.mock.patch.dict(os.environ, {"MCP_MAX_TEXT_CHARS": "7"}):
            app = McpApplication()
        self.assertTrue(app.is_ready())
        self.assertEqual(app._max_text_chars, 7)

    def test_each_tool_records_its_own_audit_name(self) -> None:
        app = McpApplication()
        app.call_tool("get_validation_guidance", {"validation_standard": arguments()["validation_standard"]})
        app.call_tool("find_evidence_candidates", {**arguments(), "queries": "invalid"})
        app.call_tool("evaluate_report_against_standard", arguments())
        self.assertEqual(
            [(event.tool_name, event.result_code, event.standard_version) for event in app._audit.events],
            [("get_validation_guidance", "ok", "v1"), ("find_evidence_candidates", "invalid_input", "v1"), ("evaluate_report_against_standard", "ok", "v1")],
        )

    def test_service_version_comes_from_package_metadata(self) -> None:
        self.assertEqual(SERVICE_VERSION, importlib.metadata.version("main-mcp-server-document-analysis"))
        app = McpApplication()
        app.evaluate(arguments())
        self.assertEqual(app._audit.events[-1].service_version, SERVICE_VERSION)
