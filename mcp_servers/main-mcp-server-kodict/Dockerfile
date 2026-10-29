# 런타임 이미지에는 검증 완료된 SQLite DB만 읽기 전용으로 마운트한다.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 KODICT_DATABASE_PATH=/app/data/korean_standard_dictionary.sqlite3
RUN useradd --system --uid 10001 appuser
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["main-mcp-server-kodict"]
