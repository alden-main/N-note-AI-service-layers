# 합성 전사 벤치마크

이 디렉터리는 운영 이미지에서 제외된다. `eval/benchmark/v1/inputs.jsonl`은 음성 API의 `patient_id`, `encounter_id`, `schema_id`, `language_hint`와 음성 대신 STT 이후 합성 전사문을 담는다. 공개 음성 업로드 API 호출이 아니라 내부 workflow의 전사문 입력을 시험한다. `references.jsonl`은 사례별 요약, 근거, 선별 기준을 담으며 모델 입력에 포함되지 않는다. 100건은 20개 시나리오 계열의 각 5개 표현 변형이며 독립된 임상 상황 100개는 아니다. 임상 정답 검토를 받지 않은 개발용 초안이다.

```bash
# 데이터 검증 (네트워크·모델 호출 없음)
uv run python -m benchmark.bench validate
# 모델 접근 설정 후, no-tools 고정 대역을 사용 (각 명령은 모델 300회 호출)
N_NOTE_LLM_TOOL_CALLING=false uv run python -m benchmark.bench run --run-id baseline --repeat 3
# 프롬프트/모델 변경 후 동일한 환경에서 실행
N_NOTE_LLM_TOOL_CALLING=false uv run python -m benchmark.bench run --run-id candidate --repeat 3
uv run python -m benchmark.bench score eval/benchmark/results/baseline
uv run python -m benchmark.bench score eval/benchmark/results/candidate
uv run python -m benchmark.bench compare --baseline eval/benchmark/results/baseline --candidate eval/benchmark/results/candidate --output eval/benchmark/results/comparison.json
```

## 로컬 모델 서버로 실행 (GPU 서버 + 개발 PC)

벤치마크는 운영 이미지에 없고 ai-service API도 거치지 않는다. 개발 PC에서 실행하고, GPU 서버의 vLLM(`local-llm`)에는 SSH 터널로 접근한다. 서버에는 번들(이미지 압축 파일)만 설치돼 있으면 된다.

```bash
# 1) GPU 서버(번들 디렉터리): 측정하는 동안만 local-llm을 서버의 127.0.0.1:18000에 공개
N_NOTE_COMPOSE_EXTRA_FILES=docker-compose.benchmark.yml scripts/compose-model.sh qwen3.5-9b up -d local-llm

# 2) 개발 PC: 터널 연결(별도 터미널에서 유지)
ssh -N -L 18000:127.0.0.1:18000 <gpu-server>

# 3) 개발 PC: 프로필 설정(max_tokens·temperature·extra_body 등)을 그대로 쓰고 주소만 터널로 바꿈
N_NOTE_LLM_BASE_URL=http://127.0.0.1:18000/v1 N_NOTE_LLM_TOOL_CALLING=false \
  uv run python -m benchmark.bench run --profile qwen3.5-9b --run-id qwen9b-baseline --repeat 3

# 4) GPU 서버: 측정이 끝나면 override 없이 다시 기동해 포트를 닫음
scripts/compose-model.sh qwen3.5-9b up -d local-llm
```

- `--profile`은 `deploy/models/<profile>.env`를 읽는다. 우선순위는 셸 환경 변수 > 프로필 > `.env`이다. 프로필의 `http://local-llm:8000/v1`은 Compose 내부 주소라, 셸에서 `N_NOTE_LLM_BASE_URL`을 바꾸지 않으면 실행 전에 거부한다. 결과 `manifest.json`에 `model_profile`과 `llm_base_url`이 남는다.
- 1·4단계는 `local-llm` 컨테이너를 다시 만들므로 모델을 다시 적재한다(수 분). 운영 중인 서버라면 적재하는 동안 초안 생성이 멈춘다.
- override를 적용하는 동안 `local-llm`은 벤치마크 전용 bridge 네트워크에 추가로 연결되어 외부 통신이 가능하다. 포트는 서버의 127.0.0.1에만 열린다.
- 측정한 시간에는 터널 왕복 시간이 포함된다. 품질 비교에는 문제가 없지만, 지연시간은 같은 경로(같은 PC·같은 터널)로 측정한 결과끼리만 비교한다. `compare`는 실행한 PC의 정보만 검사하고 모델 서버나 네트워크 경로는 검사하지 않는다.
- 입력은 모두 합성 데이터이므로 터널로 보내도 실제 환자 정보는 나가지 않는다.

기존 `n-note-model-eval`은 호스트에서 `uv run python -m n_note_ai_service_layers.run_model_eval`로 사용할 수 있다. 모델 평가 코드는 운영 이미지에서 제외된다.

## 검토 및 판정

정답 세트는 현재 `unreviewed`이며 이 상태에서 PASS를 낼 수 없다. 생성기의 사실과 요약은 임상 전문가 2명이 검토·조정해야 한다. 검토 후 입력과 정답 파일의 변경 내용을 검토하고 `manifest.json`의 파일별 SHA-256과 `review_status`를 명시적으로 갱신한다. `generate_dataset.py`를 다시 실행하면 검토·수정 사항이 덮어써지므로 주의한다. 데이터 해시가 다르면 같은 기준과 비교하지 않는다.

`score`는 키워드 존재를 선별 신호로만 표시하며, 이것으로 임상적 정확성이나 부정 표현을 판단하지 않는다. 전체 출력의 허위 사실·임상 오류에 대한 사람이 검토한 JSONL을 `--reviews`로 전달한다. 결과의 각 `(case_id, repeat)`에 대해 한 줄이 필요하다.

```json
{"case_id":"syn-001","repeat":1,"reviewer":"reviewer-a","facts":{"primary":"correct"},"unsupported_fact":false,"critical_error":false}
```

`facts` 값은 `correct`, `incorrect`, `missing`, `uncertain` 중 하나이고 uncertain은 검토 미완료다. 추가 사실 또는 중대 오류 여부는 초안 전체를 검토한다. 단일 실행 판정과 두 실행 비교 판정은 다르다. 문서의 p95 비교 및 사실 재현율 차이는 `compare`에서만 확인한다. 종료 코드는 PASS 0, FAIL 1, INCONCLUSIVE 2다. 미검토·불완전한 결과는 PASS가 아니다. 품질 100건의 지연은 참고값으로 p99의 통계적 안정성을 보장하지 않으며 부하 시험은 별도 구현이 필요하다.

## 누출 방지와 한계

모델 호출은 별도 프로세스에서 실행하며 allowlist로 환자 요청 필드·기준 시각·전사문만 전달한다. 정답, 요약, 분류, 사례 ID는 전달하지 않는다. 호출 순서는 무작위이며 데이터와 양식 해시를 보존하고 결과 건수·동일 사례를 비교한다. 이것은 실수로 답을 프롬프트에 넣는 것을 막는 경계이지, 같은 컴퓨터에서 실행되는 악의적인 코드에 대한 보안 샌드박스가 아니다. 저장소에 공개된 정답·보류 사례가 모델 학습에 이미 포함됐는지 검증할 수도 없다. 독립 평가가 필요하면 별도 비공개 데이터와 격리된 실행 환경을 사용한다. `holdout`은 개발 시 분리해 두는 표시이며, 실행기는 기본적으로 100건을 모두 평가한다.
