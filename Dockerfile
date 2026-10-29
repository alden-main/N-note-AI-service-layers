FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    N_NOTE_FORMS_DIR=/app/forms \
    N_NOTE_ARTIFACT_DIR=/var/lib/n-note/artifacts \
    N_NOTE_STATE_DB=/var/lib/n-note/state/state.sqlite3 \
    N_NOTE_PDF_FONT_PATH=/usr/share/fonts/truetype/nanum/NanumGothic.ttf \
    N_NOTE_TIMEZONE=Asia/Seoul \
    TZ=Asia/Seoul \
    N_NOTE_WORKERS=2

WORKDIR /app

# 확정 간호기록 PDF에 포함할 한글 TrueType 글꼴과, 병동 시간대(Asia/Seoul) 해석용 tzdata이다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-nanum tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# true면 로컬 NER 비식별화(transformers·CPU torch, 약 1GB)를 함께 설치한다. 모델은 HF 캐시 volume에서 읽는다.
ARG N_NOTE_INSTALL_NER=false
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project $([ "$N_NOTE_INSTALL_NER" = "true" ] && echo --extra ner)

COPY src ./src
RUN uv sync --frozen --no-dev $([ "$N_NOTE_INSTALL_NER" = "true" ] && echo --extra ner)

# 기본 양식 정의다. 운영 양식(7종)과 PDF 양식은 /app/forms에 read-only volume으로 덮어쓸 수 있다.
COPY forms ./forms

EXPOSE 8000

# 초안·핸드오프·양식 상태는 N_NOTE_STATE_DB(SQLite)로 worker 간에 공유되므로 다중 worker로 실행할 수 있다.
CMD ["sh", "-c", "exec uv run --no-sync uvicorn n_note_ai_service_layers.api:app --host 0.0.0.0 --port 8000 --workers ${N_NOTE_WORKERS}"]
