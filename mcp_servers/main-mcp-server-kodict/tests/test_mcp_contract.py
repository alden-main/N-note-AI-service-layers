"""Gateway가 사용하는 MCP 세션과 도구 결과 계약을 검증한다."""

import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

import anyio
import httpx
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from main_mcp_server_kodict.config import Settings
from main_mcp_server_kodict import server as server_module
from main_mcp_server_kodict.server import create_server


def make_database(path: Path) -> None:
    """외부 XLS 없이 MCP 조회를 확인할 최소 읽기 전용 DB를 만든다."""
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


async def run_gateway_contract(server: object) -> tuple[types.InitializeResult, types.ListToolsResult, types.CallToolResult, types.CallToolResult]:
    """Gateway와 같은 순서로 initialize, initialized, catalog, tools/call을 수행한다."""
    mcp_server = getattr(server, "_mcp_server")
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                lambda: mcp_server.run(server_read, server_write, mcp_server.create_initialization_options())
            )
            try:
                async with ClientSession(client_read, client_write) as client:
                    initialized = await client.send_request(
                        types.ClientRequest(
                            types.InitializeRequest(
                                params=types.InitializeRequestParams(
                                    protocolVersion="2025-06-18",
                                    capabilities=types.ClientCapabilities(),
                                    clientInfo=types.Implementation(name="gateway-contract-test", version="1.0"),
                                )
                            )
                        ),
                        types.InitializeResult,
                    )
                    await client.send_notification(types.ClientNotification(types.InitializedNotification()))
                    tools = await client.list_tools()
                    success = await client.call_tool("lookup_standard_dictionary", {"query": "가다"})
                    invalid_input = await client.call_tool("review_standard_korean_text", {"text": ""})
                    return initialized, tools, success, invalid_input
            finally:
                task_group.cancel_scope.cancel()


def test_gateway_initialize_catalog_and_tool_result_contract(tmp_path: Path) -> None:
    """MCP 2025-06-18과 성공·실패 tool result의 안정 계약을 확인한다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    server = create_server(Settings(database_path=database, dictionary_version="test-dictionary"))

    with patch.object(server_module.logger, "info") as audit_log:
        initialized, tools, success, invalid_input = anyio.run(run_gateway_contract, server)

    assert initialized.protocolVersion == "2025-06-18"
    catalog = {tool.name: tool for tool in tools.tools}
    assert {"lookup_standard_dictionary", "review_standard_korean_text"} <= set(catalog)
    assert all(tool.description and isinstance(tool.inputSchema, dict) for tool in catalog.values())
    assert "기본 20개" in catalog["lookup_standard_dictionary"].description
    assert "findings_truncated" in catalog["review_standard_korean_text"].description
    assert success.isError is False
    assert success.structuredContent is not None
    assert success.structuredContent["entries"][0]["entry_id"] == "sample.xls:2"
    assert invalid_input.isError is True
    assert invalid_input.structuredContent == {
        "error": {
            "code": "invalid_input",
            "message": "text는 비어 있지 않은 문자열이어야 합니다.",
            "retryable": False,
        }
    }
    assert any(str(call.args[0]).startswith("audit service_id=%s") for call in audit_log.call_args_list)
    assert any("test-dictionary" in call.args for call in audit_log.call_args_list)
    assert any("request_id=%s" in str(call.args[0]) and "duration_ms=%s" in str(call.args[0]) for call in audit_log.call_args_list)
    assert all("가다" not in call.args and "text는 비어 있지 않은 문자열이어야 합니다." not in call.args for call in audit_log.call_args_list)


def test_review_returns_retryable_unavailable_when_dictionary_is_not_ready(tmp_path: Path) -> None:
    """사전 준비 실패는 HTTP transport 오류가 아닌 재시도 가능한 도구 오류다."""
    server = create_server(Settings(database_path=tmp_path / "missing.sqlite3"))

    async def call_review() -> types.CallToolResult:
        result = await server.call_tool("review_standard_korean_text", {"text": "검토할 문장"})
        assert isinstance(result, types.CallToolResult)
        return result

    result = anyio.run(call_review)

    assert result.isError is True
    assert result.structuredContent == {
        "error": {
            "code": "upstream_unavailable",
            "message": "분석기 또는 사전 DB를 사용할 수 없습니다.",
            "retryable": True,
        }
    }


def test_healthz_returns_503_when_dictionary_is_not_ready(tmp_path: Path) -> None:
    """준비되지 않은 사전은 민감한 상세 정보 없이 readiness에서 제외한다."""
    server = create_server(Settings(database_path=tmp_path / "missing.sqlite3"))

    async def get_healthz() -> httpx.Response:
        transport = httpx.ASGITransport(app=server.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/healthz")

    response = anyio.run(get_healthz)

    assert response.status_code == 503
    assert response.text == "unavailable"


def test_request_body_limit_is_independent_from_text_limit(tmp_path: Path) -> None:
    """HTTP 본문 한도는 text 코드 포인트 한도와 별도의 배포 설정이다."""
    database = tmp_path / "dictionary.sqlite3"
    make_database(database)
    settings = Settings(database_path=database, max_request_body_size=1234, max_text_length=10)
    server = create_server(settings)

    assert server.settings.max_request_body_size == 1234
    assert settings.max_text_length == 10

    async def call_oversized_text() -> types.CallToolResult:
        result = await server.call_tool("review_standard_korean_text", {"text": "가" * 11})
        assert isinstance(result, types.CallToolResult)
        return result

    result = anyio.run(call_oversized_text)
    assert result.isError is True
    assert result.structuredContent == {
        "error": {
            "code": "request_too_large",
            "message": "text가 허용 길이를 초과했습니다.",
            "retryable": False,
        }
    }
