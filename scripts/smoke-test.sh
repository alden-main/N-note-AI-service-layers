#!/usr/bin/env bash
# 모델 프로필을 띄우고, 실제 음성으로 STT·녹음 세션·초안 생성이 끝까지 동작하는지 확인한다(GPU 서버용).
#   scripts/smoke-test.sh gemma-4-12b
#   scripts/smoke-test.sh whisper-large-v3-turbo --stt-only
#   scripts/smoke-test.sh gemma-4-12b --chunk-seconds 30 --audio eval/cases/<case>/audio.wav
# 프로필 이름 뒤의 인자는 n-note-smoke로 그대로 넘긴다(--help로 확인).
# 출력: GPU 메모리 사용량, vLLM 적재 로그(KV cache·최대 동시 처리), 단계별 결과. 보고서는 eval/smoke/<profile>/에 남는다.
# 환경 변수: N_NOTE_SMOKE_WAIT_SECONDS(서비스 healthy 대기, 기본 1800), N_NOTE_SMOKE_KEEP_RUNNING=false면 끝난 뒤 내린다.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
profile="${1:?usage: $0 <profile> [n-note-smoke args...]}"
shift
compose=("$root/scripts/compose-model.sh" "$profile")
wait_seconds="${N_NOTE_SMOKE_WAIT_SECONDS:-1800}"

services=(ai-service)
profiles="$(sed -n 's/^N_NOTE_COMPOSE_PROFILES=//p' "$root/deploy/models/$profile.env")"
[[ ",$profiles," == *",local-stt,"* ]] && services+=(local-stt)
[[ ",$profiles," == *",local-llm,"* ]] && services+=(local-llm)

echo "== [$profile] starting: ${services[*]}" >&2
"${compose[@]}" up -d --build

# 모델 적재는 수 분 걸린다(vLLM start_period 900초). 모든 서비스가 healthy가 될 때까지 기다린다.
deadline=$((SECONDS + wait_seconds))
for service in "${services[@]}"; do
  while :; do
    id="$("${compose[@]}" ps -q "$service")"
    status="$( [[ -n "$id" ]] && docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$id" || echo missing)"
    [[ "$status" == healthy ]] && { echo "== $service: healthy" >&2; break; }
    if [[ "$status" == unhealthy || "$status" == exited || "$status" == missing ]] || (( SECONDS > deadline )); then
      echo "== $service: $status (giving up). recent logs:" >&2
      "${compose[@]}" logs --tail 80 "$service" >&2 || true
      exit 1
    fi
    echo "   waiting for $service ($status, $((deadline - SECONDS))s left)" >&2
    sleep 15
  done
done

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== GPU memory" >&2
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv >&2 || true
fi
for service in "${services[@]}"; do
  [[ "$service" == ai-service ]] && continue
  echo "== $service load log (KV cache / concurrency / model memory)" >&2
  "${compose[@]}" logs --no-log-prefix "$service" 2>&1 \
    | grep -Ei 'kv cache|maximum concurrency|model loading took|model weights took|out of memory|error' | tail -n 12 >&2 || true
done

echo "== smoke test" >&2
set +e
# 실행 중인 ai-service 안에서 자기 자신(127.0.0.1)을 호출한다. 보고서는 마운트된 eval/smoke/에 남는다.
"${compose[@]}" exec -T ai-service uv run --no-sync n-note-smoke --api-url http://127.0.0.1:8000 "$@"
result=$?
set -e
if [[ "${N_NOTE_SMOKE_KEEP_RUNNING:-true}" == false ]]; then
  "${compose[@]}" down
fi
exit $result
