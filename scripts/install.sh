#!/usr/bin/env bash
# Linux 서버에서 배포 번들(scripts/bundle.sh로 만든 것)을 설치하고 기동한다.
#   무결성 확인 → 이미지 로드 → .env 준비 → docker compose up(빌드 없음) → healthy 대기
#
#   tar -xzf n-note-1.2.0-linux-amd64.tar.gz && cd n-note-1.2.0-linux-amd64
#   ./scripts/install.sh                                   # 처음 설치
#   ./scripts/install.sh --from /opt/n-note/n-note-1.1.0-linux-amd64   # 업그레이드: 이전 설치의 .env·override를 가져온다
#   ./scripts/install.sh --skip-load                       # .env만 고친 뒤 다시 기동(이미지 로드 생략)
#   ./scripts/install.sh --no-start                        # 이미지 로드·.env 준비까지만
#
# 컨테이너 상태(초안·작업·감사 이력, 확정본)는 Docker volume에 있고, COMPOSE_PROJECT_NAME(기본 n-note)이
# 같으면 버전을 바꿔도 그대로 이어진다.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

previous=""
skip_load=false
start=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) previous="${2:?--from needs the previous install directory}"; shift 2 ;;
    --skip-load) skip_load=true; shift ;;
    --no-start) start=false; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -f BUNDLE_INFO ]] || { echo "BUNDLE_INFO not found: run this from an extracted bundle" >&2; exit 1; }
# shellcheck disable=SC1091 # bundle.sh가 만든 KEY=VALUE 파일이다.
source BUNDLE_INFO
profile="$N_NOTE_BUNDLE_PROFILE"
echo "n-note $N_NOTE_BUNDLE_VERSION ($N_NOTE_BUNDLE_PLATFORM${profile:+, profile $profile}, commit ${N_NOTE_BUNDLE_GIT_COMMIT:0:12})"

echo "[1/5] 사전 점검"
command -v docker >/dev/null || { echo "docker is not installed" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose v2 plugin is required" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "cannot reach the Docker daemon (permission or service not running)" >&2; exit 1; }
case "$(uname -m)" in
  x86_64|amd64) host_platform="linux/amd64" ;;
  aarch64|arm64) host_platform="linux/arm64" ;;
  *) host_platform="unknown/$(uname -m)" ;;
esac
if [[ "$host_platform" != "$N_NOTE_BUNDLE_PLATFORM" && "${N_NOTE_ALLOW_PLATFORM_MISMATCH:-}" != 1 ]]; then
  echo "this bundle is for $N_NOTE_BUNDLE_PLATFORM but the server is $host_platform" >&2
  exit 1
fi

if ! $skip_load; then
  echo "[2/5] 파일 무결성 확인"
  checksum="sha256sum"
  command -v sha256sum >/dev/null || checksum="shasum -a 256"
  # 실패한 파일만 보여준다(GNU·BusyBox·macOS 공통).
  if ! result="$($checksum -c SHA256SUMS 2>&1)"; then
    echo "$result" | grep -v ': OK$' >&2
    exit 1
  fi
  echo "[3/5] 이미지 로드"
  docker load -i images.tar
else
  echo "[2/5] 무결성 확인 생략(--skip-load)"
  echo "[3/5] 이미지 로드 생략(--skip-load)"
fi

echo "[4/5] 설정(.env)"
if [[ -n "$previous" ]]; then
  [[ -f "$previous/.env" ]] || { echo "$previous/.env not found" >&2; exit 1; }
  cp "$previous/.env" .env
  echo "  $previous/.env 를 가져왔습니다"
  # 서버별 설정(포트 공개 등)
  if [[ -f "$previous/docker-compose.override.yml" ]]; then
    cp "$previous/docker-compose.override.yml" .
    echo "  $previous/docker-compose.override.yml 을 가져왔습니다"
  fi
elif [[ ! -f .env ]]; then
  cp .env.example .env
  echo "  .env.example로 .env를 만들었습니다"
fi
chmod 600 .env

set_env() {  # 값이 있으면 바꾸고, 없으면 추가한다.
  local key="$1" value="$2"
  if grep -q "^$key=" .env; then
    sed -i.bak "s|^$key=.*|$key=$value|" .env && rm -f .env.bak
  else
    printf '%s=%s\n' "$key" "$value" >> .env
  fi
}
env_value() { sed -n "s/^$1=//p" .env | tail -n 1; }

set_env N_NOTE_IMAGE_TAG "$N_NOTE_BUNDLE_VERSION"
# 볼륨 이름이 프로젝트 이름을 따르므로, 버전·설치 경로가 바뀌어도 같은 이름을 쓴다.
[[ -n "$(env_value COMPOSE_PROJECT_NAME)" ]] || set_env COMPOSE_PROJECT_NAME n-note
project="$(env_value COMPOSE_PROJECT_NAME)"

if [[ -z "$profile" && -z "$(env_value OPENAI_API_KEY)" ]]; then
  echo "  경고: OPENAI_API_KEY가 비어 있습니다. STT·초안 생성이 실패합니다." >&2
  echo "        .env에 키를 넣고 ./scripts/install.sh --skip-load 로 다시 기동하세요." >&2
fi

# 예전 절차(docker network create)로 만든 같은 이름의 네트워크는 Compose가 쓰지 못한다.
for network in ai-gateway-network mcp-gateway-network; do
  if docker network inspect "$network" >/dev/null 2>&1; then
    owner="$(docker network inspect "$network" --format '{{index .Labels "com.docker.compose.project"}}')"
    if [[ "$owner" != "$project" ]]; then
      echo "network $network already exists (owner: ${owner:-manual}). Stop the containers using it and run:" >&2
      echo "  docker network rm $network" >&2
      exit 1
    fi
  fi
done

if ! $start; then
  echo "[5/5] 기동 생략(--no-start). 기동: ./scripts/install.sh --skip-load"
  exit 0
fi

echo "[5/5] 기동"
# 로컬 LLM은 모델 적재에 수 분이 걸린다.
timeout="${N_NOTE_WAIT_TIMEOUT:-$([[ -n "$profile" ]] && echo 1800 || echo 300)}"
compose() {
  if [[ -n "$profile" ]]; then
    scripts/compose-model.sh "$profile" "$@"
  else
    docker compose "$@"
  fi
}
if compose up -d --no-build --wait --wait-timeout "$timeout"; then
  compose ps
  echo
  echo "기동 완료. 로그: docker compose logs -f ai-service gateway"
else
  compose ps -a
  echo >&2
  echo "일부 서비스가 ${timeout}초 안에 healthy가 되지 않았습니다. 로그를 확인하세요:" >&2
  echo "  docker compose logs --tail 100 <service>" >&2
  exit 1
fi
