FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    N_NOTE_FORMS_DIR=/app/forms \
    N_NOTE_ARTIFACT_DIR=/var/lib/n-note/artifacts \
    N_NOTE_PDF_FONT_PATH=/usr/share/fonts/truetype/nanum/NanumGothic.ttf

WORKDIR /app

# 확정 간호기록 PDF에 포함할 한글 TrueType 글꼴이다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# 기본 양식 정의다. 운영 양식(7종)과 PDF 양식은 /app/forms에 read-only volume으로 덮어쓸 수 있다.
COPY forms ./forms

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "n_note_ai_service_layers.api:app", "--host", "0.0.0.0", "--port", "8000"]
