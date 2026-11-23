# 검증된 사전 릴리스(data/releases/<KODICT_RELEASE>/)를 이미지에 포함한다. 사전 교체는 이미지 재빌드로 한다.
#   docker build --build-arg KODICT_RELEASE=2026.10 .
FROM python:3.12-slim

ARG KODICT_RELEASE=2026.10
WORKDIR /app
# uv가 만든 /app/.venv를 PATH 앞에 두어 CMD의 진입점과 HEALTHCHECK의 python이 venv를 쓴다.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH \
    KODICT_DATABASE_PATH=/app/data/korean_standard_dictionary.sqlite3 \
    KODICT_DATABASE_IMMUTABLE=true \
    KODICT_DICTIONARY_VERSION=${KODICT_RELEASE}
RUN useradd --system --uid 10001 appuser
# 사전은 코드보다 드물게 바뀌므로 먼저 복사해, 코드만 바뀔 때 이 계층(약 500MB)을 캐시로 재사용한다.
# 읽기 전용(0444, root 소유)이며 서버는 SQLite mode=ro&immutable=1로만 연다.
COPY --chmod=0444 data/releases/${KODICT_RELEASE}/korean_standard_dictionary.sqlite3 data/releases/${KODICT_RELEASE}/manifest.json /app/data/
# --chmod가 만든 디렉터리에도 적용되므로 진입 권한(x)을 되돌린다.
RUN chmod 0555 /app/data
RUN pip install --no-cache-dir uv
# 의존성은 uv.lock 그대로 먼저 설치해, 코드만 바뀔 때 이 계층을 캐시로 재사용한다.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["main-mcp-server-kodict"]
