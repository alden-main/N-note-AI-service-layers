# N-note AI service layers

간호사 구술 음성을 STT로 전사하고, 선택한 간호기록 양식에 맞춰 초안을 만듭니다. 이 과정에서 MCP Gateway를 거쳐 표준어와 의학용어를 보정하고 양식 완결성을 검증합니다. 간호사가 검토·확정한 기록은 PDF로 저장합니다. 실제 EMR 쓰기는 포함하지 않습니다.

```
음성 ─▶ STT ─▶ 비식별화 ─▶ 전사문 용어 검토(Gateway) ─▶ 양식 초안 생성(LLM + 검토 힌트·사전 도구)
     ─▶ 초안 수치 대조 ─▶ 초안 필드별 용어 후보(Gateway) ─▶ 양식 완결성 검증(필수 항목은 로컬, 추가 기준은 Gateway) ─▶ 간호사 검토·후보 적용 ─▶ 확정 ─▶ PDF
```

| Gateway 도구 | MCP 서버 | 용도 |
|---|---|---|
| `korean__review_standard_korean_text` | main-mcp-server-kodict | 전사문·초안 표준어 후보 |
| `medic__analyze_document_terms` | main-mcp-server-medicaldict | 전사문·초안 의학용어 후보 |
| `korean__lookup_standard_dictionary`, `medic__search_medical_terms` | 위 두 서버 | 초안 생성 중 LLM 사전 조회(읽기 전용) |
| `report__evaluate_report_against_standard` | main-mcp-server-document-analysis | 양식별 추가 기준(`validation_criteria`) 검증. 섹션별 필수 항목은 AI 서비스가 직접 판정 |

AI 서비스는 `MCP_GATEWAY_URL`(기본 `http://gateway:8080/mcp`)만 호출합니다. 후보는 항상 `auto_apply=false`이며, 간호사가 승인한 후보만 필드에 반영됩니다.

## 실행

`.env`(Git 제외) 예시:

```dotenv
OPENAI_API_KEY=...            # OpenAI STT·LLM을 쓸 때만 필요
OPENAI_STT_MODEL=gpt-4o-transcribe
OPENAI_DRAFT_MODEL=gpt-4o-mini
# 선택: 병동 약어·의학용어 인식률을 높이는 STT 문맥 힌트(비우면 settings.py의 기본 문맥을 쓴다)
OPENAI_STT_PROMPT=간호기록 구술. V/S, NPO, I/O, POD, 욕창, 낙상
N_NOTE_DEFAULT_FORM=nursing-daily-note
```

```bash
uv run n-note-api                 # 로컬 (http://localhost:8000/docs)
docker compose up --build -d      # ai-service + Gateway + MCP 서버 3종을 함께 실행(STT·LLM은 OpenAI)
docker compose ps                 # 5개 서비스가 healthy인지 확인
```

`docker compose` 하나로 다음 서비스가 함께 뜹니다. 네트워크는 Compose가 만들므로 `docker network create`가 필요 없습니다.

### 외부 API 테스트

테스트 시 다음 명령으로 API 포트만 공개할 수 있습니다. Gateway, MCP 서버와 로컬 모델 서버의 포트는 공개하지 않습니다.

```bash
docker compose -f docker-compose.yml -f docker-compose.api-test.yml up -d --build
curl http://127.0.0.1:8100/healthz
# 다른 컴퓨터에서: curl http://<서버-IP>:8100/healthz
# API 문서: http://<서버-IP>:8100/docs
```

기본적으로 모든 호스트 인터페이스의 8100 포트에 바인딩합니다. `N_NOTE_API_BIND_ADDRESS`와 `N_NOTE_API_PORT` 환경 변수 또는 `.env`로 변경할 수 있습니다(예: 로컬에서만 접근하려면 `N_NOTE_API_BIND_ADDRESS=127.0.0.1`). 원격 접속에는 서버 방화벽과 네트워크의 해당 포트 허용이 필요합니다. 테스트 종료 후 기본 설정으로 되돌리려면 `docker compose -f docker-compose.yml -f docker-compose.api-test.yml down` 후 `docker compose up -d`를 실행합니다.

이 설정은 API 앱 전체(`/internal/v1`, `/docs` 포함)를 공개하며 별도의 인증을 추가하지 않습니다. `X-Authenticated-User` 헤더는 인증 프록시가 검증한 값을 신뢰하는 용도이므로 직접 공개된 API의 인증 수단이 아닙니다. 신뢰할 수 있는 테스트 네트워크에서 합성 데이터만 사용하고, 인터넷에 공개하려면 인증 프록시와 접근 제한을 먼저 구성하세요.

| 서비스 | 소스 | 네트워크 | 역할 |
|---|---|---|---|
| `ai-service` | 이 저장소 | `n-note-edge`, `ai-gateway-network` | API·STT·초안 생성 |
| `gateway` | `mcp_servers/main-mcp-gateway-server` | `ai-gateway-network`, `mcp-gateway-network` | ai-service가 호출하는 유일한 MCP endpoint |
| `korean` | `mcp_servers/main-mcp-server-kodict` | `mcp-gateway-network` | 표준국어 검토 |
| `medic` | `mcp_servers/main-mcp-server-medicaldict` | `mcp-gateway-network` | 의학용어 검토 |
| `report` | `mcp_servers/main-mcp-server-document-analysis` | `mcp-gateway-network` | 양식 완결성 검증 |

MCP 서버 하나가 기동에 실패해도 Gateway와 ai-service는 뜨고, 해당 검토만 `unavailable`로 표시됩니다(`docker compose ps`에서 `unhealthy` 확인). 서버가 복구되면 Gateway가 다음 도구 목록 조회에서 다시 찾고, ai-service는 필요한 도구가 빠진 동안 15초마다 목록을 다시 조회하므로 재시작할 필요가 없습니다.

외부와 연결되는 네트워크는 `n-note-edge`뿐이고 ai-service만 여기에 붙습니다. API 포트 공개, OpenAI 호출, 브리지·프록시 컨테이너 연동은 모두 이 네트워크를 씁니다. `ai-gateway-network`와 `mcp-gateway-network`는 `internal` 네트워크라 인터넷과 외부 컨테이너가 Gateway·로컬 모델 서버·MCP 서버에 직접 접근할 수 없고, ai-service도 MCP 서버에는 직접 접근할 수 없습니다. 표준국어 사전 DB는 kodict 이미지 빌드 시 `mcp_servers/main-mcp-server-kodict/data/releases/<KODICT_RELEASE>/`(기본 `2026.10`)에서 이미지 안으로 들어갑니다. 사전을 바꾸면 이미지를 다시 빌드합니다. 예전 방식으로 `docker network create`로 만든 같은 이름의 네트워크가 남아 있으면 Compose가 그대로 써서 `internal` 설정이 적용되지 않으므로, 이를 쓰던 컨테이너를 내리고 `docker network rm n-note-edge ai-gateway-network mcp-gateway-network` 중 해당하는 네트워크를 지운 뒤 기동합니다.

서버 하나만 다시 빌드해 교체할 때는 root에서 `docker compose up -d --build --no-deps medic`처럼 실행합니다. `mcp_servers/main-mcp-server-*/compose.yaml`은 서버 단독 기동 확인용이라 전용 internal 네트워크에서 실행되고 Gateway와 연결되지 않습니다.

### Linux 서버 배포 (macOS에서 빌드)

```bash
scripts/bundle.sh --version 1.0.0        # 모든 이미지(linux/amd64)와 실행 파일을 tar.gz 하나로
# 서버로 옮긴 뒤
tar -xzf n-note-1.0.0-linux-amd64.tar.gz && cd n-note-1.0.0-linux-amd64 && ./scripts/install.sh
```

업그레이드·롤백·포트 공개·로컬 GPU 모델 포함 방법은 [docs/linux_docker_model_deployment.md](docs/linux_docker_model_deployment.md)에 있습니다.

### STT·LLM 모델 선택 (SaaS / 온프레미스)

STT와 LLM은 OpenAI 호환 API로만 호출하므로, OpenAI와 로컬 모델 서버(vLLM 등)를 설정으로 바꿀 수 있습니다. 모델별 설정은 `deploy/models/<profile>.env`에 있습니다(MedGemma 1.5 4B·27B, Qwen3.5 9B·27B, Gemma 4 12B·26B-A4B, STT 전용 Whisper large-v3-turbo). 프로필 목록과 A100 40GB 메모리 계획, 폐쇄망 반입 방법은 [deploy/models/README.md](deploy/models/README.md)에 있습니다.

```bash
scripts/compose-model.sh medgemma-27b up -d    # Whisper + MedGemma 27B 로컬 서버와 함께 실행
scripts/smoke-test.sh gemma-4-12b             # GPU 서버: 띄우기 → GPU·적재 로그 → 전사·녹음 세션·초안 생성 실제 호출
scripts/smoke-test.sh whisper-large-v3-turbo --stt-only   # STT만 확인
```

### 모델 비교 평가

`eval/cases/`의 합성 사례(전사문 또는 음성)를 현재 설정한 모델로 실행하고 채점합니다. 채점 항목은 다음과 같습니다.

- 필수 항목 작성률
- 핵심어 포함률
- 금지어 검출: 구술에 없는 날짜·수치 같은 지어낸 내용
- STT 문자 오류율(CER)
- 처리 시간

```bash
uv run python -m n_note_ai_service_layers.run_model_eval run --profile openai     # → eval/results/<profile>/<시각>/
uv run python -m n_note_ai_service_layers.run_model_eval compare                  # 프로필별 최근 결과 비교표
```

사례 형식은 [eval/cases/README.md](eval/cases/README.md)에 있습니다.

초안 생성 중 사전 도구 사용량(`N_NOTE_LLM_MAX_TOOL_ROUNDS`)을 바꿀 때는 같은 사례로 비교한 뒤 정합니다. 평가는 별도의 개발 환경에서 실행하며, 사전 도구 비교에는 그 환경에서 접근할 수 있는 테스트용 Gateway가 필요합니다. 운영용 ai-service 이미지에는 평가기를 포함하지 않습니다.

```bash
for rounds in 0 1 2; do   # 0 = 힌트만, 2 = 기본값
  N_NOTE_LLM_MAX_TOOL_ROUNDS=$rounds uv run python -m n_note_ai_service_layers.run_model_eval run --profile tools-$rounds
done
uv run python -m n_note_ai_service_layers.run_model_eval compare   # 핵심어 포함률·금지어·처리 시간 비교
```

양식 7종은 `forms/` 아래에 디렉터리로 추가합니다. 형식과 PDF 배치 방법은 [forms/README.md](forms/README.md)에 있습니다.

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `N_NOTE_STT_BASE_URL` / `N_NOTE_LLM_BASE_URL` | 없음(OpenAI) | 로컬 OpenAI 호환 서버 주소(예: `http://local-llm:8000/v1`) |
| `N_NOTE_STT_MODEL` / `N_NOTE_LLM_MODEL` | `OPENAI_STT_MODEL` / `OPENAI_DRAFT_MODEL` 값 | 서버의 모델(서빙) 이름 |
| `N_NOTE_LLM_API` | base URL 있으면 `chat`, 없으면 `responses` | 초안 생성 API 형식 |
| `N_NOTE_LLM_TOOL_CALLING` | `true` | 초안 생성 중 사전 MCP 도구 호출. tool call 미지원 모델은 `false` |
| `N_NOTE_LLM_MAX_TOOL_ROUNDS` / `N_NOTE_LLM_MAX_TOOL_CALLS` | `2` / `6` | 초안 하나에서 사전 도구 결과를 LLM에 돌려주는 왕복 수 / 도구 호출 총수. LLM 호출은 최대 `1 + 왕복 수`번이고, 상한에 닿으면 도구 없이 초안을 받는다. `0`이면 도구를 주지 않고 전사문 검토 힌트만 쓴다 |
| `N_NOTE_LLM_EXTRA_BODY` | 없음 | 서버에 넘길 추가 JSON(예: thinking 끄기) |
| `N_NOTE_LLM_MAX_TOKENS` / `N_NOTE_LLM_TEMPERATURE` | 서버 기본값 | 초안 생성 옵션 |
| `N_NOTE_FORMS_DIR` | `forms` | 양식 정의 디렉터리 |
| `N_NOTE_ENABLED_FORMS` | (전체) | 사용할 양식 ID 목록 |
| `N_NOTE_DEFAULT_FORM` | 없음 | 요청에 양식 ID가 없을 때 사용할 양식 |
| `N_NOTE_ARTIFACT_DIR` | `data/artifacts` | 확정본 JSON·PDF 저장 위치 |
| `N_NOTE_PDF_FONT_PATH` | NanumGothic/AppleGothic 자동 탐색 | PDF 한글 TrueType 글꼴 |
| `MCP_GATEWAY_TIMEOUT_SECONDS` | `15` | Gateway 요청 timeout |
| `N_NOTE_LLM_TOOLS` | 사전 조회 도구 2종 | LLM에 노출할 Gateway 도구(쉼표 구분) |
| `N_NOTE_TERM_REVIEW_TOOLS` | 표준어·의학용어 검토 도구 2종 | 후보 생성 도구 |
| `N_NOTE_TERM_HINT_BUDGET_SECONDS` | `3` | 전사문 용어 검토를 LLM 힌트로 쓰려고 기다리는 최대 시간. 넘으면 힌트 없이 초안 생성을 시작하고 검토 후보는 나중에 응답에 담음 |
| `N_NOTE_REPORT_VALIDATION_TOOL` | `report__evaluate_report_against_standard` | 양식별 추가 기준 검증 도구. 비우면 추가 기준 검증을 생략(필수 항목 판정은 계속) |
| `N_NOTE_STATE_DB` | 없음(메모리) · Docker는 `/var/lib/n-note/state/state.sqlite3` | 초안·핸드오프·감사 이력·양식 상태 SQLite. 미설정이면 재시작 시 사라지고 worker 간에 공유되지 않는다 |
| `N_NOTE_WORKERS` | `1` | uvicorn worker 수. 녹음 세션이 프로세스 메모리에 있으므로 1을 권장합니다. 2 이상은 `N_NOTE_STATE_DB`와 세션 단위 sticky 라우팅이 함께 필요합니다 |
| `N_NOTE_DRAFT_RETENTION_HOURS` | `72` | 확정 전 초안(전사문·환자 ID 포함) 보관 시간 |
| `N_NOTE_TIMEZONE` | `Asia/Seoul` | 구술 시각("오후 2시")·감사 시각을 해석하는 병동 시간대. 컨테이너 시간대(UTC)와 무관 |
| `N_NOTE_STT_TIMEOUT_SECONDS` / `N_NOTE_LLM_TIMEOUT_SECONDS` | `60` / `90` | STT·LLM 요청 한 번의 제한 시간 |
| `N_NOTE_MODEL_MAX_RETRIES` | `1` | STT·LLM SDK 재시도 횟수 |
| `N_NOTE_REQUIRE_LOCAL_STT` | `false` | `true`면 로컬 STT(`N_NOTE_STT_BASE_URL`)만 허용. 비식별화는 전사 이후이므로 실환자 운영에서는 `true` |
| `N_NOTE_USER_HEADER` | `X-Authenticated-User` | 앞단이 인증한 사용자 ID 헤더. 있으면 `edited_by`·`finalized_by`·`operator_id` 대신 사용하고, 본문 값과 다르면 403 |
| `N_NOTE_REQUIRE_AUTHENTICATED_USER` | `false` | `true`면 수정·확정·양식 승인에 인증 헤더 필수(없으면 401) |
| `MCP_GATEWAY_CATALOG_TTL_SECONDS` | `300` | Gateway 도구 catalog 재사용 시간(요청마다 initialize·tools/list 왕복 생략) |
| `N_NOTE_JOB_CONCURRENCY` / `N_NOTE_JOB_MAX_PENDING` | `4` / `32` | worker당 초안 생성 작업 동시 실행 수 / 대기열 상한(넘으면 429) |
| `N_NOTE_FORM_CONCURRENCY` / `N_NOTE_MAX_FORMS_PER_JOB` | `3` / `7` | 여러 양식 작업(`schema_ids`)에서 동시에 초안을 만들 양식 수 / 작업 하나에 요청할 수 있는 양식 수. worker당 LLM 동시 호출은 최대 `JOB_CONCURRENCY × FORM_CONCURRENCY` |
| `N_NOTE_JOB_TIMEOUT_SECONDS` | `600` | 작업 상태가 이 시간 동안 갱신되지 않으면 `interrupted`로 실패 처리 |
| `N_NOTE_NER_MODEL` | 없음(규칙만) | 로컬 NER 비식별화 모델. 예: `Leo97/KoELECTRA-small-v3-modu-ner`. `uv sync --extra ner`(Docker는 `N_NOTE_INSTALL_NER=true` 빌드) 필요. 설정했는데 못 올리면 기동 실패 |
| `N_NOTE_NER_MIN_SCORE` / `N_NOTE_NER_LABELS` / `N_NOTE_NER_THREADS` | `0.8` / `PS,PER,PERSON` / `2` | NER 최소 신뢰도 / 사람 라벨 / CPU 스레드 |
| `N_NOTE_LOG_LEVEL` | `INFO` | JSON 구조화 로그 수준. 로그에는 추적 ID·단계·소요 시간·오류 유형만 남기고 전사문·필드 값은 남기지 않는다 |

## API 흐름

전체 명세: [docs/api_specification.md](docs/api_specification.md)

```bash
# 1) 양식 목록
curl http://localhost:8000/v1/form-templates

# 2) 음성 → 초안 생성 작업 접수(202, job_id). 처리는 백그라운드에서 진행된다.
curl -i -X POST http://localhost:8000/v1/nursing-note-workflows \
  -H "Idempotency-Key: $(uuidgen)" \
  -F 'audio=@note.wav;type=audio/wav' -F language_hint=ko \
  -F patient_id=P-1 -F encounter_id=E-1 -F schema_id=nursing-daily-note

# 2-1) 진행 상태 구독(SSE) 또는 폴링 → succeeded면 result.draft_id
curl -N http://localhost:8000/v1/nursing-note-jobs/{job_id}/events
curl http://localhost:8000/v1/nursing-note-jobs/{job_id}

# 3) 후보 승인(제안어 선택) 또는 직접 수정 → 새 revision
curl -X POST http://localhost:8000/v1/nursing-note-drafts/{draft_id}/revisions \
  -H 'Content-Type: application/json' \
  -d '{"edited_by":"nurse-1","expected_revision":1,"candidate_decisions":{"draft:korean__review_standard_korean_text:0":"accepted"},"selected_suggestions":{}}'

# 4) 확정 → 불변 확정본 + PDF 생성
#    전사문에 없는 수치가 남아 있으면 확인 표시(numeric_review_confirmed)가 있어야 확정된다.
curl -X POST http://localhost:8000/v1/nursing-note-drafts/{draft_id}/finalize \
  -H 'Content-Type: application/json' -d '{"finalized_by":"nurse-1","numeric_review_confirmed":false}'

# 5) PDF 다운로드
curl -o note.pdf http://localhost:8000/v1/nursing-notes/{note_id}/pdf
```

- 초안 응답의 `draft.sections[].fields[].candidate_ids`가 그 필드에 적용할 수 있는 후보입니다. 같은 필드를 직접 수정하면 해당 필드의 미적용 후보는 `stale`이 되어 더 이상 적용할 수 없습니다.
- 필수 항목이 비어 있으면 확정되지 않습니다(422). 확정 후에는 수정할 수 없습니다(409).
- **수치 대조**: 초안의 숫자(활력징후, NRS, 용량, 시각, 날짜)가 전사문이나 LLM에 준 사실에 없으면 `numeric_verification.findings`에 표시합니다. 예를 들어 "혈압 백에 육십"이 "160/100"으로, "맥박 구십 이"가 "90"으로 바뀐 경우, 또는 구술하지 않은 기록 시각을 채운 경우입니다.
  - 전사문의 한글 수사(삼십팔 점 이, 백이십, 칠오일, 일곱 시, 오후 두 시)도 숫자로 읽어 대조합니다.
  - 간호사가 값을 고치면 표시가 사라집니다. 표시가 남은 채 확정하려면 `numeric_review_confirmed: true`가 필요하며(없으면 422), 확인한 값은 확정본의 `confirmed_numeric_findings`에 남습니다.
  - 한 자리 정수(점수·횟수)와 초안에 한글로 쓴 숫자는 대조하지 않습니다.
- PDF는 확정본만 대상으로 만듭니다. 초안은 JSON(섹션 구조)으로 화면에 표시합니다.
- **동시 수정**: 수정·확정 요청에 화면에서 본 `expected_revision`을 넣으면, 그 사이 다른 사용자가 수정한 경우 409로 거부합니다.
- **LLM이 반영한 용어 교정**: LLM은 전사문 용어 후보를 힌트로 받아 초안에 바로 반영할 수 있습니다. 이렇게 후보 승인 없이 바뀐 표기는 `llm_applied_term_hints`(초안·확정본)에 남기므로 검토 화면에서 확인합니다.
- **여러 양식 동시 생성**: `schema_id` 대신 `schema_ids`(반복 필드 또는 쉼표 구분)를 보내면 STT·비식별화·전사문 검토를 한 번만 하고 양식별 초안을 병렬로 만듭니다. 양식별 진행은 작업의 `forms`에 실시간으로 남고, 먼저 끝난 양식은 `forms[].draft_id`로 바로 검토할 수 있습니다. 자세한 형식은 [API 명세](docs/api_specification.md#여러-양식-동시-생성)에 있습니다.
- **STT 전처리**: 로컬 STT를 쓰면 음성을 16kHz mono WAV로 바꾸고 앞뒤 무음을 잘라 보냅니다. 브라우저 webm/opus도 엔진과 관계없이 처리됩니다. 말소리가 없는 음성은 STT에 보내지 않습니다(에너지 기반 VAD). 전사문의 방송 자막류 환각 문장과 반복 루프는 지웁니다. 설정은 `N_NOTE_STT_AUDIO_NORMALIZE`, `N_NOTE_STT_VAD_*`, `N_NOTE_STT_HALLUCINATION_*`입니다.
- **녹음 중 실시간 전사**: `POST /v1/recording-sessions`로 세션을 열고 녹음 중 20~30초 청크를 `.../chunks`로 보내면 청크마다 누적 전사문(비식별화)을 바로 돌려줍니다. 녹음이 끝나면 `.../workflows`로 모은 전사문을 써서 초안 생성 작업을 시작합니다. STT를 다시 하지 않으므로 STT 처리량은 파일 업로드와 같습니다. 녹음 중에는 LLM을 호출하지 않습니다. 자세한 형식은 [API 명세](docs/api_specification.md#녹음-세션녹음-중-실시간-전사)에 있습니다.
- **비동기 작업**: 초안 생성은 `202` + `job_id`로 접수되고, 상태(`queued → running → succeeded|failed`)와 단계(`stt`, `redaction`, …)를 조회·SSE로 확인합니다. 실패하면 `error.code`(`stt_unavailable`, `redaction_failed`, `draft_generation_unavailable`, `interrupted` 등)로 구분합니다. Gateway 장애는 실패가 아니라 각 검증 결과의 `status: unavailable`로 표시합니다.
- **비식별화**: 전화번호·주민번호·이메일, 라벨이 붙은 이름·번호(구술형 "성함은 홍길동" 포함), "김영희님"처럼 호칭이 붙은 이름, 한글로 읽은 전화번호, 요청의 `patient_id`·`encounter_id`를 보호 토큰으로 치환합니다. 로컬 NER(`N_NOTE_NER_MODEL`)을 켜면 "영희가 열이 나요"처럼 문맥만 있는 이름도 가립니다. NER 실패 시 작업은 외부 호출 없이 중단됩니다.

### 로컬 NER 비식별화 켜기

```bash
uv sync --extra ner                                             # transformers + CPU torch
N_NOTE_NER_MODEL=Leo97/KoELECTRA-small-v3-modu-ner scripts/download_models.sh openai   # 모델을 ./model-cache로 받기
# Docker: N_NOTE_INSTALL_NER=true docker compose build && N_NOTE_NER_MODEL=Leo97/KoELECTRA-small-v3-modu-ner docker compose up -d
N_NOTE_TEST_NER_MODEL=Leo97/KoELECTRA-small-v3-modu-ner uv run python -m unittest tests.test_operational_hardening   # 실제 모델 검증
```

모델 카드에 라이선스 표기가 없으므로, 운영 반입 전 라이선스를 확인하거나 자체 학습·라이선스가 명확한 모델로 바꿉니다(`N_NOTE_NER_MODEL`은 경로도 받습니다).

## 검증

```bash
uv run python -m unittest discover -s tests -v
```
