"""Раздел «Загрузка»: шаги конвейера данных, запуск из пульта, журнал и логи.

Шаги и их команды задаёт движок (data.pipeline); пульт только запускает их в
фоне и показывает ход. Данные пишут скрипты шагов, сам пульт пишет лишь журнал
запусков в app/runs/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..runs import LogChunk, PipelineState, Run, RunManager, RunRejected

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@dataclass
class RunRequest:
    """Что запустить: id шагов (порядок — всегда как в реестре шагов)."""

    steps: list[str]


@dataclass
class StopRequest:
    """Какой запуск остановить после текущего шага."""

    run_id: str


def manager(request: Request) -> RunManager:
    return request.app.state.runs


Manager = Annotated[RunManager, Depends(manager)]


def rejected(error: RunRejected) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=str(error))


@router.get("")
def pipeline_state(runs: Manager) -> PipelineState:
    """Шаги, порядок «Обновить всё», журнал запусков и кто пишет данные вне пульта."""
    return runs.state()


@router.post("/runs")
def start_run(request: RunRequest, runs: Manager) -> Run:
    """Запускает шаги в фоне. 409 — такой запуск уже идёт или данные пишет другой процесс."""
    try:
        return runs.start(request.steps)
    except RunRejected as error:
        raise rejected(error) from None


@router.post("/stop")
def stop_run(request: StopRequest, runs: Manager) -> Run:
    """Остановить после текущего шага: текущий доработает, чтобы не испортить файлы."""
    try:
        return runs.stop(request.run_id)
    except RunRejected as error:
        raise rejected(error) from None


@router.get("/log/file", include_in_schema=False)
def run_log_file(run_id: str, runs: Manager) -> FileResponse:
    """Лог запуска целиком как текст: ссылка «Открыть файл» на экране «Загрузка»."""
    try:
        path = runs.log_file(run_id)
    except RunRejected as error:
        raise rejected(error) from None
    # inline: браузер показывает текст во вкладке, а не скачивает файл.
    return FileResponse(path, media_type="text/plain; charset=utf-8", content_disposition_type="inline")


@router.get("/log")
def run_log(run_id: str, runs: Manager, offset: int = 0) -> LogChunk:
    """Лог запуска с байта offset; пульт дочитывает его, пока запуск идёт."""
    try:
        return runs.read_log(run_id, offset)
    except RunRejected as error:
        raise rejected(error) from None
