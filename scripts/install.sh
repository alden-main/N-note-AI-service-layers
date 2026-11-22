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
# 추적 화면(Phoenix)이 든 번들(bundle.sh --with-observability)이면 Phoenix 비밀값을 만들어 .env에 넣고 함께 띄운다.
# 띄우지 않으려면 .env에 N_NOTE_OBSERVABILITY=false를 두고 --skip-load로 다시 기동한다.
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
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
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

# 추적 화면(Phoenix): 번들에 들어 있고 .env에서 끄지 않았으면 함께 띄운다.
observability_file="docker-compose.observability.yml"
observability=false
if [[ "${N_NOTE_BUNDLE_OBSERVABILITY:-false}" == true && -f "$observability_file" ]]; then
  [[ -n "$(env_value N_NOTE_OBSERVABILITY)" ]] || set_env N_NOTE_OBSERVABILITY true
  if [[ "$(env_value N_NOTE_OBSERVABILITY)" == true ]]; then
    observability=true
  fi
elif [[ "$(env_value N_NOTE_OBSERVABILITY)" == true ]]; then
  echo "  경고: .env는 N_NOTE_OBSERVABILITY=true이지만 이 번들에는 Phoenix가 없습니다(bundle.sh --with-observability)." >&2
  echo "        추적 화면 없이 기동합니다." >&2
fi
random_secret() {  # 영숫자 48자. head가 먼저 끝나므로 pipefail에 걸리지 않는다.
  local value
  value="$(head -c 512 /dev/urandom | LC_ALL=C tr -dc 'A-Za-z0-9')"
  printf '%s' "${value:0:48}"
}
if $observability; then
  # 한 번 만든 값은 바꾸지 않는다. PHOENIX_SECRET이 바뀌면 로그인 세션·API 키가 무효가 되고,
  # PHOENIX_ADMIN_PASSWORD는 Phoenix가 처음 기동할 때만 쓴다(이후 비밀번호는 화면에서 바꾼 값이 Phoenix volume에 남는다).
  for key in PHOENIX_SECRET PHOENIX_ADMIN_SECRET PHOENIX_ADMIN_PASSWORD; do
    if [[ -z "$(env_value "$key")" ]]; then
      set_env "$key" "$(random_secret)"
      echo "  $key 를 만들어 .env에 넣었습니다"
    fi
  done
fi
# 이후 직접 실행하는 docker compose ps·logs·down도 같은 파일 묶음을 쓰도록 .env의 COMPOSE_FILE에 적는다
# (compose-model.sh는 -f를 직접 넘기므로 이 값을 쓰지 않고 N_NOTE_OBSERVABILITY를 본다).
compose_files="docker-compose.yml"
[[ -f docker-compose.override.yml ]] && compose_files+=":docker-compose.override.yml"
if $observability; then
  set_env COMPOSE_FILE "$compose_files:$observability_file"
elif grep -q "^COMPOSE_FILE=.*$observability_file" .env; then
  # Phoenix 없는 번들로 바꿨거나 끈 경우다. 없는 파일을 가리키면 compose가 실패하므로 지운다.
  sed -i.bak '/^COMPOSE_FILE=/d' .env && rm -f .env.bak
  echo "  COMPOSE_FILE에서 Phoenix overlay를 뺐습니다"
fi

if [[ -z "$profile" && -z "$(env_value OPENAI_API_KEY)" ]]; then
  echo "  경고: OPENAI_API_KEY가 비어 있습니다. STT·초안 생성이 실패합니다." >&2
  echo "        .env에 키를 넣고 ./scripts/install.sh --skip-load 로 다시 기동하세요." >&2
fi

# 예전 절차(docker network create)로 만든 같은 이름의 네트워크가 있으면 Compose가 그대로 써서 internal 설정이 적용되지 않는다.
for network in n-note-edge ai-gateway-network mcp-gateway-network; do
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
# 버전 사이에 서비스 이름이 바뀌면 이전 컨테이너가 남으므로 함께 정리한다(volume은 유지된다).
if compose up -d --no-build --remove-orphans --wait --wait-timeout "$timeout"; then
  compose ps
  echo
  echo "기동 완료. 로그: docker compose logs -f ai-service gateway"
  if $observability; then
    echo "추적 화면(Phoenix): 서버에서 http://127.0.0.1:$(env_value PHOENIX_PORT | grep . || echo 6006)"
    echo "  다른 PC에서는 ssh -L 6006:127.0.0.1:$(env_value PHOENIX_PORT | grep . || echo 6006) <서버> 후 http://127.0.0.1:6006"
    echo "  로그인: admin@localhost / 첫 비밀번호는 .env의 PHOENIX_ADMIN_PASSWORD (로그인 후 바꾸세요)"
  fi
else
  compose ps -a
  echo >&2
  echo "일부 서비스가 ${timeout}초 안에 healthy가 되지 않았습니다. 로그를 확인하세요:" >&2
  echo "  docker compose logs --tail 100 <service>" >&2
  exit 1
fi
