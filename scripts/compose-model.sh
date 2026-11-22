#!/usr/bin/env bash
# 모델 프로필(deploy/models/<profile>.env)로 docker compose를 실행한다.
#   scripts/compose-model.sh qwen3.5-27b up -d
#   scripts/compose-model.sh qwen3.5-27b logs -f local-llm
# 모델 비교 평가기는 운영 이미지에 없으므로 호스트에서 실행한다(uv run python -m n_note_ai_service_layers.run_model_eval).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
profile="${1:?usage: $0 <profile> <docker compose args...>}"
shift
env_file="$root/deploy/models/$profile.env"
if [[ ! -f "$env_file" ]]; then
  echo "unknown profile: $profile (available: $(ls "$root/deploy/models" | sed -n 's/\.env$//p' | tr '\n' ' '))" >&2
  exit 2
fi
args=(--project-directory "$root" -f "$root/docker-compose.yml")
# 서버별 설정(포트 공개 등)은 docker-compose.override.yml에 둔다. 일반 docker compose처럼 함께 읽는다.
[[ -f "$root/docker-compose.override.yml" ]] && args+=(-f "$root/docker-compose.override.yml")
# 추적 화면(Phoenix)을 함께 설치한 경우(.env의 N_NOTE_OBSERVABILITY=true, install.sh가 정한다) overlay도 읽는다.
observability_file="docker-compose.observability.yml"
if [[ -f "$root/.env" && -f "$root/$observability_file" ]] && grep -q '^N_NOTE_OBSERVABILITY=true$' "$root/.env" \
  && [[ ",${N_NOTE_COMPOSE_EXTRA_FILES:-}," != *",$observability_file,"* ]]; then
  args+=(-f "$root/$observability_file")
fi
# 추가 compose 파일(쉼표 구분, 저장소·번들 기준 경로). 예: N_NOTE_COMPOSE_EXTRA_FILES=docker-compose.benchmark.yml
IFS=',' read -ra extra_files <<< "${N_NOTE_COMPOSE_EXTRA_FILES:-}"
for item in ${extra_files[@]+"${extra_files[@]}"}; do
  if [[ -n "$item" ]]; then
    [[ -f "$root/$item" ]] || { echo "compose file not found: $item" >&2; exit 2; }
    args+=(-f "$root/$item")
  fi
done
# .env(API 키 등) 다음에 프로필을 읽어 프로필 값이 우선한다.
[[ -f "$root/.env" ]] && args+=(--env-file "$root/.env")
args+=(--env-file "$env_file")
IFS=',' read -ra profiles <<< "$(sed -n 's/^N_NOTE_COMPOSE_PROFILES=//p' "$env_file")"
for item in ${profiles[@]+"${profiles[@]}"}; do
  [[ -n "$item" ]] && args+=(--profile "$item")
done
exec docker compose "${args[@]}" "$@"
