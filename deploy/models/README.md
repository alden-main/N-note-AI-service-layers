# 로컬 모델 프로필 (온프레미스·폐쇄망)

AI 서비스는 STT와 초안 생성 LLM을 **OpenAI 호환 API**로만 호출합니다. 그래서 OpenAI(SaaS)와 로컬 모델 서버(vLLM 등)를 코드 변경 없이 **설정만으로** 바꿀 수 있습니다. 모델 하나는 `deploy/models/<profile>.env` 파일 하나로 정의하며, 같은 파일이 다음 두 곳에 쓰입니다.

- 모델 서버(`local-stt`, `local-llm` 컨테이너)의 모델과 기동 인자
- AI 서비스의 접속 주소와 호출 방식(API 형식, tool call 사용 여부, 추가 요청 필드)

```bash
scripts/compose-model.sh qwen3.5-27b up -d                 # 프로필로 전체 실행
scripts/compose-model.sh qwen3.5-27b logs -f local-llm      # 모델 적재 확인(수 분 소요)
scripts/compose-model.sh qwen3.5-27b run --rm ai-service \
  uv run --no-sync n-note-model-eval run                    # 같은 평가 사례로 채점
uv run n-note-model-eval compare                            # 프로필별 결과 비교
```

## 프로필 (A100 40GB 1장 기준)

| 프로필 | 모델 | 적재 방식 / 메모리 | tool call | 비고 |
|---|---|---|---|---|
| `openai` | gpt-4o-transcribe + gpt-4o-mini | OpenAI API | ✅ | SaaS 기준선 |
| `medgemma-1.5-4b` | google/medgemma-1.5-4b-it | bf16, 약 9GB | ❌ | 의료 특화, 가장 가벼움, gated |
| `medgemma-27b` | google/medgemma-27b-text-it | FP8 weight-only, 약 28GB | ❌ | 의료 특화. 40GB에 빠듯하며 호스트 RAM 64GB 이상 필요, gated |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B | bf16, 약 19GB | ✅ | Apache 2.0, 다국어. 여유 있는 기준 모델 |
| `qwen3.5-27b` | Qwen/Qwen3.5-27B-GPTQ-Int4 | 공식 int4, 약 15GB | ✅ | Apache 2.0. 27B급 중 40GB에서 가장 안정적 |
| `gemma-4-26b-a4b` | google/gemma-4-26B-A4B-it | FP8 weight-only, 약 26GB | ❌(설정 시 가능) | Apache 2.0, MoE라 빠름. **A100 동작 실측 필요(실험)** |

공통: STT는 `openai/whisper-large-v3-turbo`(GPU 메모리 10%)를 같은 GPU에서 함께 띄웁니다. LLM은 GPU 메모리 80~85%를 씁니다.

- **tool call ❌**: 초안 생성 중 사전 MCP 도구를 직접 호출하지 않습니다. 대신 전사문 표준어·의학용어 검토 결과를 힌트로 받아 초안을 만듭니다. 초안 작성 후의 용어 후보 생성과 양식 검증은 모델과 무관하게 MCP Gateway가 똑같이 수행합니다. MedGemma는 tool call을 공식 지원하지 않습니다.
- **thinking**: Qwen3.5는 thinking을 끕니다(`N_NOTE_LLM_EXTRA_BODY`). 지연 시간이 더 중요하기 때문입니다. 품질 비교를 위해 켤 수도 있습니다.
- **메모리 수치는 추정치입니다.** A100에서 처음 띄울 때 `local-llm` 로그의 KV cache 크기와 OOM 여부를 확인하고 `--max-model-len`과 `--gpu-memory-utilization`을 조정하세요.
- **라이선스**: MedGemma는 Health AI Developer Foundations 약관, Gemma 4와 Qwen3.5는 Apache 2.0입니다. EXAONE은 비상업(NC) 라이선스라 상용 후보에서 제외했습니다. HyperCLOVA X SEED 등 한국어 특화 모델은 라이선스와 vLLM 지원을 확인한 뒤 프로필을 추가하면 됩니다.

### 새 모델 추가

기존 `.env`를 복사해 아래 항목을 바꾸면 됩니다.

| 키 | 의미 |
|---|---|
| `N_NOTE_MODEL_PROFILE` | 프로필 이름(파일명과 같게) |
| `N_NOTE_COMPOSE_PROFILES` | 함께 띄울 서버: `local-stt`, `local-llm`(쉼표 구분, 비우면 OpenAI) |
| `N_NOTE_LOCAL_LLM_MODEL` / `N_NOTE_LOCAL_LLM_ARGS` | vLLM에 넘길 Hugging Face 모델 ID와 `vllm serve` 인자 |
| `N_NOTE_LLM_MODEL` | vLLM `--served-model-name`이자 초안의 `model_version`에 남는 이름 |
| `N_NOTE_LLM_API` | `chat`(로컬 서버) 또는 `responses`(OpenAI) |
| `N_NOTE_LLM_TOOL_CALLING` | 모델과 vLLM tool parser가 tool call을 지원할 때만 `true` |
| `N_NOTE_LLM_EXTRA_BODY` | 서버에 그대로 넘길 JSON(예: `chat_template_kwargs`) |
| `N_NOTE_LOCAL_STT_MODEL` / `N_NOTE_STT_MODEL` | Whisper 모델(한국어 미세조정 모델로 교체 가능)과 그 서빙 이름 |

STT만 로컬로 두고 LLM은 OpenAI를 쓰는(또는 반대) 혼합 구성도 가능합니다. 해당 서버의 `*_BASE_URL`만 비우면 OpenAI로 갑니다.

## 다른 서버 엔진

AI 서비스가 요구하는 조건은 두 가지입니다.

1. OpenAI 호환 `/v1/chat/completions`와 `/v1/audio/transcriptions`를 제공할 것
2. `response_format: json_schema`를 지원할 것. 지원하지 않는 서버에도 프롬프트로 스키마를 전달하지만, 구조가 틀린 응답은 오류로 처리합니다.

이 조건만 맞으면 SGLang, TGI, llama.cpp 서버, 사내 추론 게이트웨이를 그대로 쓸 수 있습니다. `N_NOTE_LLM_BASE_URL`과 `N_NOTE_STT_BASE_URL`을 그 주소로 지정하세요. 온프레미스 구성이 아직 정해지지 않았으므로 엔진 종류에 종속되지 않도록 이 경계만 유지합니다.

## 폐쇄망 반입

인터넷이 되는 곳에서 준비해 반입합니다.

```bash
# 1) 모델 가중치 (gated 모델은 Hugging Face에서 약관 동의 후 HF_TOKEN 필요)
HF_TOKEN=... scripts/download_models.sh qwen3.5-27b      # → ./model-cache/hub

# 2) 이미지 빌드 후 저장 (local-stt는 vLLM 이미지에 음성 의존성만 추가한 이미지)
scripts/compose-model.sh qwen3.5-27b build
scripts/compose-model.sh qwen3.5-27b pull local-llm
docker save -o n-note-images.tar \
  vllm/vllm-openai:v0.31.0-cu129 n-note-local-stt:0.31.0 n-note-ai-service-layers-ai-service
```

폐쇄망 서버에서 실행합니다.

```bash
docker load -i n-note-images.tar
# model-cache/를 복사한 뒤 .env에 HF_HUB_OFFLINE=1
scripts/compose-model.sh qwen3.5-27b up -d --no-build
```

- vLLM 이미지는 `v0.31.0-cu129`(CUDA 12.9)로 고정했습니다. A100 서버의 NVIDIA 드라이버가 CUDA 12.9를 지원하는지 확인하세요. 버전은 `N_NOTE_VLLM_IMAGE`와 `N_NOTE_VLLM_VERSION`으로 바꿉니다.
- MCP Gateway와 MCP 서버 이미지는 각 저장소의 `dist/*.tar.gz`로 함께 반입합니다.
