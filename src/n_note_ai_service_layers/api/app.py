"""FastAPI 앱을 조립한다."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from n_note_ai_service_layers.api.routes import bridge, internal_stt, nursing_notes, templates, transcription, workflow_jobs
from n_note_ai_service_layers.api.state import get_services


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 잘못된 양식 정의·저장소·NER 설정은 첫 요청이 아니라 기동 시점에 실패시킨다.
    services = get_services()
    yield
    # 대기 중인 작업은 실패(server_shutdown)로 남기고, 실행 중인 작업은 마칠 때까지 기다린다.
    services.job_runner.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="N-note Bridge API", version="0.1.0", lifespan=_lifespan)

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(templates.router)
    app.include_router(transcription.router)
    app.include_router(internal_stt.router)
    app.include_router(workflow_jobs.router)
    app.include_router(nursing_notes.router)
    app.include_router(bridge.router)
    return app


def main() -> None:
    import uvicorn
    uvicorn.run("n_note_ai_service_layers.api:app", host="0.0.0.0", port=8000, workers=int(os.getenv("N_NOTE_WORKERS", "1")))
