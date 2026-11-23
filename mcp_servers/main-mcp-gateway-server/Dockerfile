FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

RUN groupadd --system mcp && useradd --system --gid mcp --home-dir /nonexistent --shell /usr/sbin/nologin mcp

RUN pip install --no-cache-dir uv

# uv.lock에 고정한 의존성만 설치한다. 의존성 계층을 먼저 만들어 소스 변경 시 재사용한다.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev
COPY config ./config

USER mcp

ENTRYPOINT ["main-mcp-gateway-server"]
