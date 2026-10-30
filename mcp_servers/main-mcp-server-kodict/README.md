# 한국어 표준어 사전 MCP 서버

분할 XLS 원본을 배포 전 SQLite로 변환하고, 표준어 사전 근거를 MCP 도구로 제공한다.

## 개발

```bash
uv sync --all-groups --extra build
uv run --extra build build-korean-dictionary --version 2026.10
uv run pytest
```

빌드 산출물 `data/releases/<version>/korean_standard_dictionary.sqlite3`만 운영 컨테이너의 `/app/data/korean_standard_dictionary.sqlite3`에 읽기 전용으로 마운트한다. 호스트 DB 파일은 Compose 설정의 `KODICT_DATABASE_HOST_PATH`로 지정한다. 운영에서는 Gateway와 MCP 서버만 연결된 사설 Docker 네트워크를 신뢰 경계로 사용하며 MCP 포트를 외부에 공개하지 않는다.

macOS 개발 PC에서 Linux A100 서버로 이미지와 DB를 전달해 배포하는 절차는 [Docker 배포 가이드](docs/docker_macos_to_a100_deployment.md)를 참고한다.
