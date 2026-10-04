"""Граница движка и пульта: src/ ничего не знает про веб. Запуск: uv run pytest app

Пульт (app/) импортирует движок (src/), но не наоборот. Если движок начнёт
импортировать fastapi или app, его функции перестанут работать без веба —
например, из консоли. Тест ловит это сразу.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
WEB = {"app", "fastapi", "starlette", "uvicorn"}


def imported_packages(path: Path) -> set[str]:
    """Верхние пакеты всех абсолютных импортов файла."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_engine_does_not_import_web_layer():
    offenders = [
        f"{path.relative_to(SRC.parent)}: {', '.join(sorted(imported_packages(path) & WEB))}"
        for path in sorted(SRC.rglob("*.py"))
        if imported_packages(path) & WEB
    ]
    assert not offenders, "Движок импортирует веб-слой:\n" + "\n".join(offenders)


def test_check_catches_web_import(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from fastapi import FastAPI\nimport app.api.main\nimport polars\n")
    assert imported_packages(bad) & WEB == {"fastapi", "app"}
