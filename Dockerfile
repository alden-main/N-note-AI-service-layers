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
    N_NOTE_WORKERS=1

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

# 기본은 worker 1개다. 녹음 세션(녹음 중 청크 전사)은 원본 전사문을 디스크에 쓰지 않으려고 프로세스 메모리에 두므로,
# worker가 여럿이면 같은 세션의 청크가 다른 worker로 가 404가 난다. 동시 처리는 worker 안의 스레드풀
# (N_NOTE_JOB_CONCURRENCY·N_NOTE_FORM_CONCURRENCY)이 맡는다. 2 이상으로 늘릴 때는 세션 단위 sticky 라우팅을 둔다.
CMD ["sh", "-c", "exec uv run --no-sync uvicorn n_note_ai_service_layers.api:app --host 0.0.0.0 --port 8000 --workers ${N_NOTE_WORKERS}"]
