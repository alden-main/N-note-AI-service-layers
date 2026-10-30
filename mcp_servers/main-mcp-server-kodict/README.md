# 한국어 표준어 사전 MCP 서버

분할 XLS 원본을 배포 전 SQLite로 변환하고, 표준어 사전 근거를 MCP 도구로 제공한다.

## 개발

```bash
uv sync --all-groups --extra build
uv run --extra build build-korean-dictionary --version 2026.10
uv run pytest
```

빌드 산출물 `data/releases/<version>/korean_standard_dictionary.sqlite3`를 이미지 빌드 시 `/app/data/`에 읽기 전용으로 포함한다. 릴리스는 빌드 인자 `KODICT_RELEASE`(기본 `2026.10`)로 고르며, 같은 값이 `KODICT_DICTIONARY_VERSION`으로 감사 로그에 남는다. 사전을 바꾸려면 새 릴리스를 `data/releases/`에 만든 뒤 이미지를 다시 빌드한다. 운영에서는 Gateway와 MCP 서버만 연결된 사설 Docker 네트워크를 신뢰 경계로 사용하며 MCP 포트를 외부에 공개하지 않는다.

macOS 개발 PC에서 Linux A100 서버로 이미지와 DB를 전달해 배포하는 절차는 [Docker 배포 가이드](docs/docker_macos_to_a100_deployment.md)를 참고한다.
