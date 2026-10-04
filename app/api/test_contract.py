"""Файлы контракта API совпадают с тем, что сейчас отдаёт API. Запуск: uv run pytest app

Поменялся ответ API, а openapi.json и schema.ts не перегенерированы — тест
падает: фронтенд иначе работал бы со старым форматом и ошибался молча.
"""

from app.api.contract import contract_files, openapi_schema

HINT = "перегенерируй: uv run python -m app.api.contract"


def test_contract_files_are_current():
    for path, text in contract_files(openapi_schema()).items():
        assert path.is_file(), f"Нет {path.name} — {HINT}"
        assert path.read_text() == text, f"{path.name} не совпадает со схемой API — {HINT}"
