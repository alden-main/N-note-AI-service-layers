# MCP Gateway

Docker 내부망에서 AI 서비스의 MCP 요청을 서비스별 upstream MCP 서버로 중계하는 최소 Gateway 구현입니다.

## 실행 및 검증

```bash
uv sync
uv run python -m unittest discover -s tests -v
```

Gateway는 저장소 root의 `docker-compose.yml`로 ai-service, MCP 서버 3종과 함께 실행합니다. 이 폴더에는 별도 Compose 파일이 없습니다. Gateway만 다시 빌드해 교체하려면 root에서 다음을 실행합니다.

```bash
docker compose up -d --build --no-deps gateway
```

Gateway endpoint는 Compose 내부의 `http://gateway:8080/mcp`입니다. upstream MCP 서버는 같은 root Compose에서 `mcp-gateway-network`에 연결됩니다.

## 호출 프로토콜

`/mcp`는 MCP Python SDK의 Streamable HTTP 서버(uvicorn)입니다. 표준 MCP 클라이언트(SDK `ClientSession` 등)가 그대로 붙습니다.

- 상태 없는(stateless) JSON 응답 모드입니다. Gateway는 세션을 보관하지 않으므로 `Mcp-Session-Id`를 주지 않고, `initialize` 없이 보낸 `tools/call`도 처리합니다.
- 요청의 `Accept`에는 `application/json`이 있어야 합니다(없으면 406). 본문이 `GATEWAY_MAX_REQUEST_BODY_BYTES`를 넘으면 413입니다.
- 모르는 `protocolVersion`으로 `initialize`하면 오류 대신 Gateway가 지원하는 버전으로 답합니다(MCP 버전 협상). 버전이 맞는지는 클라이언트가 확인합니다.
- `X-Request-ID`를 보내면 upstream 호출과 감사 로그(`gateway_tool ... request_id=`)에 이어지고 응답 헤더로 돌아옵니다. 없으면 Gateway가 만듭니다.
- `/v1/tools/batch-call`(`{"calls": [{"name", "arguments"}, ...]}` → `{"results": [...]}`)은 MCP 밖의 Gateway 확장 경로입니다. 한 번에 `GATEWAY_BATCH_MAX_CALLS`(기본 32)개까지 받습니다.

upstream에는 요청마다 다른 JSON-RPC id와 `MCP-Protocol-Version` 헤더를 보냅니다. 세션을 쓰는 upstream은 응답을 id로 짝지으므로, 같은 세션에서 동시에 호출해도 응답이 섞이지 않습니다.

## Upstream 등록

Gateway는 [config/upstreams.yaml](config/upstreams.yaml)에서 upstream을 읽습니다. Compose는 이 파일을 `/app/config/upstreams.yaml`에 read-only bind mount하므로 새 MCP 서버를 추가할 때 이미지를 다시 build할 필요가 없습니다.

```yaml
upstreams:
  - service_id: new-mcp-server
    mcp_url: http://new-mcp-server:8000/mcp
    health_url: http://new-mcp-server:8000/healthz
```

각 서버는 root `docker-compose.yml`에 URL hostname과 같은 이름의 서비스로 정의하고 `mcp-gateway-network`에 연결해야 합니다. Gateway는 upstream의 `initialize` 응답에서 받은 `Mcp-Session-Id`를 `notifications/initialized`, `tools/list`, `tools/call`에 전달하므로 세션을 요구하는 Streamable HTTP MCP 서버와도 연동됩니다.

### 동시 호출과 일괄 호출

도구 호출은 catalog 잠금 밖에서 upstream을 호출하므로 여러 요청의 호출이 동시에 진행되며, discovery(`tools/list`) 중에도 기존 catalog로 호출을 계속 처리합니다. upstream 연결은 keep-alive로 재사용하고, 성공한 health 확인은 `GATEWAY_HEALTH_TTL_SECONDS`(기본 5초) 동안 재사용합니다. upstream 호출이 실패하면 health 캐시를 버립니다. `tools/list`는 upstream 목록을 `GATEWAY_CATALOG_TTL_SECONDS`(기본 30초) 동안 재사용하고, 연결되지 않은 upstream만 5초 간격으로 다시 찾습니다. discovery는 upstream마다 동시에 진행합니다.

여러 도구를 한 번에 호출하려면 `POST /v1/tools/batch-call`을 사용합니다. 각 결과는 `tools/call`과 같은 MCP tool result 계약과 감사 로그를 따르며, 한 호출의 실패가 다른 호출에 영향을 주지 않습니다.

```json
// 요청 (X-Request-ID 헤더는 모든 upstream 호출에 전달된다)
{"calls": [{"name": "korean__review_standard_korean_text", "arguments": {"text": "..."}},
           {"name": "medic__analyze_document_terms", "arguments": {"text": "..."}}]}
// 응답 200
{"results": [{"structuredContent": {...}, "isError": false}, {"structuredContent": {"error": {...}}, "isError": true}]}
```

빈 `calls`와 `GATEWAY_BATCH_MAX_CALLS`(기본 32)를 넘는 요청은 400, 본문 제한 초과는 413입니다. 일괄 호출은 Gateway 전체에서 `GATEWAY_BATCH_MAX_CONCURRENCY`(기본 8)개까지 동시에 upstream으로 보냅니다.

Gateway가 upstream보다 먼저 기동돼도 다음 클라이언트의 `tools/list` 요청에서 discovery를 다시 수행합니다. 새 MCP 서버를 등록하거나 도구 schema를 바꾼 뒤에는 클라이언트에서 MCP 연결/도구 목록을 새로고침하세요. 설정을 즉시 검증하거나 현재 연결을 모두 새로 만들려면 YAML 저장 후 `docker compose restart gateway`를 실행합니다. `UPSTREAM_CONFIG_PATH`로 다른 read-only 설정 파일 경로를 지정할 수도 있습니다.

검증 후에는 root에서 `docker compose down`으로 컨테이너와 네트워크를 정리합니다. 인증·인가·claim 전달은 현재 요구사항의 범위 밖입니다.

Linux 서버 배포 절차는 [Linux 서버 Docker 배포 가이드](../../docs/linux_docker_model_deployment.md)를 참고하세요.
