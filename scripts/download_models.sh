#!/usr/bin/env bash
# 모델 프로필의 STT·LLM 가중치를 Hugging Face 캐시(N_NOTE_MODEL_CACHE, 기본 ./model-cache)로 받는다.
# 인터넷이 되는 곳에서 실행한 뒤 캐시 디렉터리를 폐쇄망 서버로 옮기고 HF_HUB_OFFLINE=1로 실행한다.
# gated 모델(MedGemma 등)은 Hugging Face에서 약관에 동의한 계정의 HF_TOKEN이 필요하다.
#   HF_TOKEN=... scripts/download_models.sh medgemma-27b
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
profile="${1:?usage: $0 <profile>}"
env_file="$root/deploy/models/$profile.env"
[[ -f "$env_file" ]] || { echo "unknown profile: $profile" >&2; exit 2; }
cache="${N_NOTE_MODEL_CACHE:-$root/model-cache}"
mkdir -p "$cache/hub"
# 로컬 NER 비식별화 모델(N_NOTE_NER_MODEL)은 환경 변수 또는 프로필 env에서 읽는다.
for key in N_NOTE_LOCAL_STT_MODEL N_NOTE_LOCAL_LLM_MODEL N_NOTE_NER_MODEL; do
  model="$(sed -n "s/^$key=//p" "$env_file")"
  [[ "$key" == N_NOTE_NER_MODEL && -n "${N_NOTE_NER_MODEL:-}" ]] && model="$N_NOTE_NER_MODEL"
  [[ -z "$model" ]] && continue
  echo "downloading $model -> $cache/hub" >&2
  uvx --from 'huggingface_hub>=0.34' hf download "$model" --cache-dir "$cache/hub"
done
