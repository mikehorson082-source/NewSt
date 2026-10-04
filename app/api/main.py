"""FastAPI-приложение пульта: API поверх движка src/ и раздача собранного фронтенда.

API ничего не считает сам: вызывает функции движка и отдаёт результат в JSON.
Собранный фронтенд (app/frontend/dist) отдаёт этот же процесс, отдельный
веб-сервер не нужен.

Запуск из корня проекта:
    uv run uvicorn app.api.main:app --port 8000            # пульт целиком: http://127.0.0.1:8000
    uv run uvicorn app.api.main:app --port 8000 --reload   # разработка API (вместе с npm run dev)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from core.config import DATA_DIR

from .routers import data as data_routes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "app" / "frontend" / "dist"

NOT_BUILT = """<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>Кабинет исследователя</title>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem">
<h1>Фронтенд не собран</h1>
<p>API работает: <a href="/docs">/docs</a>.</p>
<p>Собрать фронтенд: <code>cd app/frontend &amp;&amp; npm run build</code> — и обновить страницу.</p>
<p>Режим разработки: <code>npm run dev</code> в app/frontend и открыть
<a href="http://localhost:5173">localhost:5173</a>.</p>
</body></html>
"""


def create_app(data_root: Path = DATA_DIR, frontend_dist: Path = FRONTEND_DIST) -> FastAPI:
    """Приложение пульта. Пути — параметры, чтобы тесты подставляли свои папки."""
    app = FastAPI(title="NewSt · Кабинет исследователя", version="0.1.0")
    app.state.data_root = data_root

    @app.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(data_routes.router)

    # Фронтенд проверяется на каждом запросе: собрал его при запущенном
    # сервере — достаточно обновить страницу.
    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> Response:
        """Файлы собранного фронтенда; неизвестный путь — index.html (маршруты внутри SPA)."""
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Нет такого метода API")
        index = frontend_dist / "index.html"
        if not index.is_file():
            return HTMLResponse(NOT_BUILT, status_code=503)
        file = (frontend_dist / path).resolve()
        if path and file.is_file() and file.is_relative_to(frontend_dist.resolve()):
            return FileResponse(file)
        # index.html без кэша: после пересборки браузер сразу увидит новую версию.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
