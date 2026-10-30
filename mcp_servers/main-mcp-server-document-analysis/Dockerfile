FROM python:3.12-slim

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN useradd --create-home --uid 10001 mcp

COPY --chown=mcp:mcp src/ ./src/

USER mcp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"

CMD ["python", "-c", "from main_mcp_server_document_analysis.mcp_server import main; main()"]
