"""Контракт API между Python и TypeScript.

Источник правды — модели ответов в Python (dataclass-ы движка, например
data.inventory.InventoryReport). Из них FastAPI строит схему OpenAPI, а из
схемы получаются:
- app/frontend/openapi.json — схема целиком (её же показывает /docs);
- app/frontend/src/api/schema.ts — типы TypeScript для фронтенда.

Поменял ответ API — перегенерируй. Иначе упадёт app/api/test_contract.py,
а TypeScript (npm run build) покажет, где фронтенд разошёлся с API.

Запуск из корня проекта: uv run python -m app.api.contract
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .main import create_app
from .openapi_ts import typescript

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
OPENAPI_PATH = FRONTEND / "openapi.json"
TYPES_PATH = FRONTEND / "src" / "api" / "schema.ts"


def openapi_schema() -> dict[str, Any]:
    return create_app().openapi()


def contract_files(schema: dict[str, Any]) -> dict[Path, str]:
    """Что должно лежать в файлах контракта для этой схемы."""
    return {
        OPENAPI_PATH: json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        TYPES_PATH: typescript(schema),
    }


def main() -> None:
    root = FRONTEND.parents[1]
    for path, text in contract_files(openapi_schema()).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Записан {path.relative_to(root)}")


if __name__ == "__main__":
    main()
