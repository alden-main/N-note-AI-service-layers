#!/usr/bin/env bash
# 개발 PC(macOS 등)에서 Linux 서버용 배포 번들 하나를 만든다.
#   이미지 일괄 빌드(대상 플랫폼) → 이미지 저장 → 실행에 필요한 파일과 함께 tar.gz 하나로 묶는다.
#
#   scripts/bundle.sh                                     # OpenAI 구성, linux/amd64, 버전 = git 짧은 해시
#   scripts/bundle.sh --version 1.2.0                     # 이미지 태그·번들 이름에 쓸 버전
#   scripts/bundle.sh --version 1.2.0 --profile qwen3.5-27b   # 로컬 STT·LLM(vLLM) 이미지까지 포함
#   scripts/bundle.sh --platform linux/arm64              # ARM 서버용
#   scripts/bundle.sh --version 1.2.0 --with-observability   # LLM 추적 화면(Phoenix) 이미지·설정까지 포함
#
# 결과: dist/bundles/n-note-<version>-linux-<arch>.tar.gz 와 .sha256
# 서버에서는 압축을 풀고 scripts/install.sh 하나로 이미지 로드와 기동까지 한다(docs/linux_docker_model_deployment.md).
# 모델 가중치(model-cache)·API 키·Phoenix 비밀값은 번들에 넣지 않는다(Phoenix 비밀값은 install.sh가 서버에서 만든다).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

version=""
platform="linux/amd64"
profile=""
observability=false
output="$root/dist/bundles"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) version="${2:?--version needs a value}"; shift 2 ;;
    --platform) platform="${2:?--platform needs a value}"; shift 2 ;;
    --profile) profile="${2:?--profile needs a value}"; shift 2 ;;
    --output) output="${2:?--output needs a value}"; shift 2 ;;
    --with-observability) observability=true; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$version" ]]; then
  version="$(git -C "$root" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)"
fi
if [[ ! "$version" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "version must be a valid Docker tag: $version" >&2
  exit 2
fi
case "$platform" in
  linux/amd64|linux/arm64) ;;
  *) echo "platform must be linux/amd64 or linux/arm64: $platform" >&2; exit 2 ;;
esac
if [[ -n "$profile" && ! -f "$root/deploy/models/$profile.env" ]]; then
  echo "unknown profile: $profile" >&2
  exit 2
fi

arch="${platform#linux/}"
name="n-note-$version-linux-$arch"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
stage="$work/$name"
mkdir -p "$stage" "$output"

# 셸 환경 변수가 .env·프로필 값보다 우선하므로 모든 이미지가 같은 태그·플랫폼으로 만들어진다.
export N_NOTE_IMAGE_TAG="$version" DOCKER_DEFAULT_PLATFORM="$platform"

observability_file="docker-compose.observability.yml"
compose_files=(-f "$root/docker-compose.yml")
if $observability; then
  compose_files+=(-f "$root/$observability_file")
  # overlay는 Phoenix 비밀값이 없으면 멈춘다. 여기서는 이미지 목록을 읽을 때만 쓰는 자리 값이며 번들에 저장하지 않는다.
  export PHOENIX_SECRET="${PHOENIX_SECRET:-bundle-placeholder}" \
    PHOENIX_ADMIN_SECRET="${PHOENIX_ADMIN_SECRET:-bundle-placeholder}" \
    PHOENIX_ADMIN_PASSWORD="${PHOENIX_ADMIN_PASSWORD:-bundle-placeholder}"
  # 프로필 구성(compose-model.sh)은 이 변수로 추가 compose 파일을 읽는다.
  export N_NOTE_COMPOSE_EXTRA_FILES="${N_NOTE_COMPOSE_EXTRA_FILES:+$N_NOTE_COMPOSE_EXTRA_FILES,}$observability_file"
fi

compose() {
  if [[ -n "$profile" ]]; then
    "$root/scripts/compose-model.sh" "$profile" "$@"
  else
    docker compose --project-directory "$root" "${compose_files[@]}" "$@"
  fi
}

echo "[1/5] 이미지 빌드 ($platform, tag $version${profile:+, profile $profile}$($observability && echo ', observability'))" >&2
compose build
# 빌드하지 않는 이미지(vLLM·Phoenix 등)는 대상 플랫폼으로 받는다.
compose pull --ignore-buildable

images="$(compose config --images | sort -u)"
echo "[2/5] 이미지 플랫폼 확인" >&2
for image in $images; do
  actual="$(docker image inspect "$image" --format '{{.Os}}/{{.Architecture}}')"
  if [[ "$actual" != "$platform" ]]; then
    echo "  $image is $actual, expected $platform" >&2
    exit 1
  fi
  echo "  $image  $actual" >&2
done

echo "[3/5] 이미지 저장" >&2
# shellcheck disable=SC2086 # 이미지 이름에는 공백이 없다.
docker save -o "$stage/images.tar" $images

echo "[4/5] 실행 파일 수집" >&2
copy() {  # copy <저장소 기준 경로> [번들 안 경로]
  local source="$root/$1" target="$stage/${2:-$1}"
  mkdir -p "$(dirname "$target")"
  cp -R "$source" "$target"
}
copy docker-compose.yml
# 벤치마크할 때만 local-llm을 127.0.0.1에 공개하는 override(N_NOTE_COMPOSE_EXTRA_FILES로 지정할 때만 쓰임)
copy docker-compose.benchmark.yml
copy .env.example
copy forms
copy eval/cases
copy deploy/models
copy scripts/compose-model.sh
copy scripts/install.sh
# 추적 화면(Phoenix)을 포함할 때만 넣는다. install.sh는 BUNDLE_INFO의 N_NOTE_BUNDLE_OBSERVABILITY로 판단한다.
if $observability; then
  copy "$observability_file"
fi
copy mcp_servers/main-mcp-gateway-server/config/upstreams.yaml
find "$stage" -name .DS_Store -delete

commit="$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)"
dirty="$(git -C "$root" status --porcelain 2>/dev/null | grep -q . && echo true || echo false)"
cat > "$stage/BUNDLE_INFO" <<INFO
N_NOTE_BUNDLE_VERSION=$version
N_NOTE_BUNDLE_PLATFORM=$platform
N_NOTE_BUNDLE_PROFILE=$profile
N_NOTE_BUNDLE_OBSERVABILITY=$observability
N_NOTE_BUNDLE_GIT_COMMIT=$commit
N_NOTE_BUNDLE_GIT_DIRTY=$dirty
N_NOTE_BUNDLE_BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
N_NOTE_BUNDLE_IMAGES="$(echo $images)"
N_NOTE_BUNDLE_KODICT_RELEASE=${KODICT_RELEASE:-2026.10}
INFO
(cd "$stage" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | sed 's|^\./||' | while read -r file; do shasum -a 256 "$file"; done > SHA256SUMS)

echo "[5/5] 압축" >&2
archive="$output/$name.tar.gz"
tar_options=()
# macOS tar는 확장 속성(com.apple.*)을 넣어 Linux tar가 "unknown extended header" 경고를 낸다.
if tar --version 2>/dev/null | grep -q bsdtar; then
  tar_options+=(--no-mac-metadata --no-xattrs)
fi
COPYFILE_DISABLE=1 tar ${tar_options[@]+"${tar_options[@]}"} -czf "$archive" -C "$work" "$name"
(cd "$output" && shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256")

size="$(du -h "$archive" | cut -f1)"
cat >&2 <<DONE

번들: $archive ($size)
해시: $archive.sha256

서버로 옮긴 뒤:
  sha256sum -c $name.tar.gz.sha256
  tar -xzf $name.tar.gz && cd $name
  ./scripts/install.sh
DONE
