"""Запуск пульта одной командой (из корня проекта).

    uv run python -m app            # пульт на http://127.0.0.1:8000
    uv run python -m app --phone    # то же, плюс доступ с телефона в домашней сети
    uv run python -m app --dev      # разработка: Vite с горячей перезагрузкой на :5173

Перед запуском фронтенд пересобирается, если его исходники новее сборки.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import uvicorn

FRONTEND = Path(__file__).resolve().parent / "frontend"
DIST_INDEX = FRONTEND / "dist" / "index.html"
SOURCES = ("src", "public", "index.html", "package.json", "vite.config.ts")


def frontend_outdated() -> bool:
    """Сборки нет или какой-то исходник фронтенда новее её."""
    if not DIST_INDEX.is_file():
        return True
    built = DIST_INDEX.stat().st_mtime
    for name in SOURCES:
        path = FRONTEND / name
        files = path.rglob("*") if path.is_dir() else [path]
        if any(f.is_file() and f.stat().st_mtime > built for f in files):
            return True
    return False


def require_npm() -> str:
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("Не найден npm: поставьте Node.js (brew install node).")
    if not (FRONTEND / "node_modules").is_dir():
        sys.exit("Нет пакетов фронтенда: npm install --prefix app/frontend")
    return npm


def lan_address() -> str:
    """Адрес Мака в домашней сети (для телефона)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.168.0.1", 1))  # пакет не отправляется, только выбор интерфейса
            return sock.getsockname()[0]
        except OSError:
            return "<адрес-мака>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Кабинет исследователя: запуск пульта")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--phone", action="store_true", help="открыть доступ из домашней сети")
    parser.add_argument("--dev", action="store_true", help="режим разработки: Vite на :5173 + API с перезагрузкой")
    args = parser.parse_args()
    host = "0.0.0.0" if args.phone else "127.0.0.1"

    if args.dev:
        vite = subprocess.Popen([require_npm(), "run", "dev"], cwd=FRONTEND)
        print("Разработка: http://localhost:5173 (Ctrl+C — остановить)")
        try:
            uvicorn.run("app.api.main:app", host=host, port=args.port, reload=True, reload_dirs=["src", "app"])
        finally:
            vite.terminate()
        return

    if frontend_outdated():
        print("Собираю фронтенд…")
        subprocess.run([require_npm(), "run", "build"], cwd=FRONTEND, check=True)
    print(f"Пульт: http://127.0.0.1:{args.port}")
    if args.phone:
        print(f"С телефона: http://{lan_address()}:{args.port}")
    uvicorn.run("app.api.main:app", host=host, port=args.port)


if __name__ == "__main__":
    main()
