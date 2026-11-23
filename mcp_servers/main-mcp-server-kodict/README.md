# 한국어 표준어 사전 MCP 서버

분할 XLS 원본을 배포 전 SQLite로 변환하고, 표준어 사전 근거를 MCP 도구로 제공한다.

## 개발

```bash
uv sync --all-groups --extra build
uv run --extra build build-korean-dictionary --version 2026.10
uv run pytest
```

빌드 산출물 `data/releases/<version>/korean_standard_dictionary.sqlite3`를 이미지 빌드 시 `/app/data/`에 읽기 전용으로 포함한다. 릴리스는 빌드 인자 `KODICT_RELEASE`(기본 `2026.10`)로 고르며, 같은 값이 `KODICT_DICTIONARY_VERSION`으로 감사 로그에 남는다. 사전을 바꾸려면 새 릴리스를 `data/releases/`에 만든 뒤 이미지를 다시 빌드한다. 이미지는 `KODICT_DATABASE_IMMUTABLE=true`로 사전을 SQLite `mode=ro&immutable=1`로 열고, 연결은 worker thread마다 재사용한다. 로컬에서 사전 파일을 다시 빌드하며 실행할 때는 이 값을 켜지 않는다. 운영에서는 Gateway와 MCP 서버만 연결된 사설 Docker 네트워크를 신뢰 경계로 사용하며 MCP 포트를 외부에 공개하지 않는다.

macOS 개발 PC에서 Linux A100 서버로 이미지와 DB를 전달해 배포하는 절차는 [Docker 배포 가이드](docs/docker_macos_to_a100_deployment.md)를 참고한다.

## 문서 검토 응답 크기

`review_standard_korean_text`는 선택 입력 `detail`을 받는다.

- `full`(기본값): 모든 후보에 사전 근거(`evidence`: 표제어·발음·품사·뜻풀이·원본 행)를 담는다. 결과 상한(`KODICT_MAX_FINDINGS`, 기본 1,000)은 모든 결과에 적용된다.
- `compact`: 위치·상태·교정 후보만 반환한다. `matched` 결과는 후보를 비우고, 나머지 후보에서는 `evidence`를 뺀다. 결과 상한은 교정 후보가 있는 결과에만 적용하므로, 정확 일치 결과가 상한을 채워 문서 뒷부분이 검토되지 않는 일이 없다. 응답 크기는 `full`의 약 1/10이다.

N-note ai-service는 도구 입력 스키마에 `detail`이 있으면 `compact`로 호출한다. Kiwi 형태소 분석기는 기동할 때 한 번 미리 실행해 두므로, 첫 요청에 모델 로딩 지연(약 0.5초)이 생기지 않는다.
