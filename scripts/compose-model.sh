#!/usr/bin/env bash
# 모델 프로필(deploy/models/<profile>.env)로 docker compose를 실행한다.
#   scripts/compose-model.sh qwen3.5-27b up -d
#   scripts/compose-model.sh qwen3.5-27b run --rm ai-service uv run --no-sync n-note-model-eval run
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
# .env(API 키 등) 다음에 프로필을 읽어 프로필 값이 우선한다.
[[ -f "$root/.env" ]] && args+=(--env-file "$root/.env")
args+=(--env-file "$env_file")
IFS=',' read -ra profiles <<< "$(sed -n 's/^N_NOTE_COMPOSE_PROFILES=//p' "$env_file")"
for item in ${profiles[@]+"${profiles[@]}"}; do
  [[ -n "$item" ]] && args+=(--profile "$item")
done
exec docker compose "${args[@]}" "$@"
