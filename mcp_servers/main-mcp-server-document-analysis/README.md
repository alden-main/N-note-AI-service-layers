# Report Validation MCP

## Docker 실행

이 서비스는 host port를 공개하지 않고 Gateway와 공유하는 Docker network에서만 실행한다.

```bash
docker network create --internal mcp-gateway-network
docker compose up --build -d
```

Gateway 컨테이너는 같은 `mcp-gateway-network`에 연결한 뒤
`http://report-validation-mcp:8000/mcp` 및 `http://report-validation-mcp:8000/healthz`로 접근한다.

## Gateway 운영 계약

이 이미지는 `mcp-gateway-network`의 Docker 네트워크 멤버십만 신뢰합니다. 따라서 Gateway와
이 MCP 컨테이너만 이 네트워크에 연결해야 하며 AI 서비스는 Gateway를 통해서만 접근해야 합니다.
MCP는 `Authorization`, JWT, API key, claim, scope 헤더를 요구·해석·저장하지 않습니다. 사용자별
인증·인가와 request ID 감사는 Gateway와 AI 서비스의 경계에서 수행합니다.

| 항목 | 값 |
| --- | --- |
| service_id | `report-validation` |
| service_version | `0.1.0` |
| mcp_endpoint / transport | `/mcp` / `streamable-http` |
| tool_contract_version | `v1` |
| data_classification | `sensitive-medical` |
| 도구 변경 | 없음(모두 읽기 전용) |

`/healthz`는 비인증 liveness endpoint입니다. 최대 MCP 요청 본문은 기본 1 MiB이며
`MCP_MAX_REQUEST_BYTES`로 조정할 수 있습니다. Gateway는 이보다 작거나 같은 본문 제한, 30초
timeout, 호출자별 rate limit을 적용해야 합니다. 이 서버는 외부 의존성을 재시도하지 않습니다.

## 긴 입력 분할

전체 JSON-RPC 요청 본문은 기본 1 MiB 이하이고, 요청 안의 각 문자열은 기본 100,000자
(`MCP_MAX_TEXT_CHARS`) 이하여야 한다. 보고서가 이를 넘으면 바이트 수로 임의 절단하지 말고
문단 또는 문장 경계에서 조각으로 나누어 각 조각을 별도 도구 호출로 보낸다. JSON 구조와
`validation_standard`, `queries`, `semantic_assessment`도 본문 크기에 포함된다.

`find_evidence_candidates`가 반환하는 citation offset과 `evaluate_report_against_standard`에
전달하는 citation offset은 **각 조각의 `report.text`를 기준**으로 한다. 원본 문서 위치가
필요하면 Gateway/호출자가 조각의 원본 시작 offset을 함께 관리해 결과를 원본 좌표로 환산한다.

## 근거 후보 탐색 확장

`find_evidence_candidates.queries[]`는 기존 `criterion_id`, 선택 `element_id`, `terms` 외에
Station C 승인 기준에서 온 `aliases`와 서버 allow-list의 `pattern_ids`를 받을 수 있다. 현재
지원하는 pattern ID는 `korean_duration_expression/v1`이며 숫자 또는 한글 수사와
`일`·`주`·`개월`·`년`, 선택 `동안`·`간`으로 된 기간 표현만 찾는다. raw regex와 미등록 ID는
`invalid_input`이다.

모든 match는 원문 code-point offset과 원문 quote를 유지하며 다음처럼 출처 전략을 제공한다.

```json
{
  "start": 12,
  "end": 19,
  "quote": "3개월 동안",
  "matched_terms": ["3개월 동안"],
  "match_strategy": "registered_pattern"
}
```

허용 전략은 `literal`, `normalized_literal`, `alias`, `normalized_alias`,
`registered_pattern`이다. 공백·탭·줄바꿈·전각 공백 차이는 탐색 시 정규화하지만 결과의
offset과 quote는 항상 원문 그대로다. 이 도구의 빈 `matches`는 문서 부재나 임상적 판단을
뜻하지 않는다.
