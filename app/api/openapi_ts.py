"""Типы TypeScript из схемы OpenAPI — без сторонних генераторов.

Схема наших ответов простая: объекты, массивы, списки значений (enum) и «или
null». Этого генератора хватает; если схема усложнится, его можно заменить на
openapi-typescript, не трогая остальной фронтенд.

Все поля ответов обязательные: API всегда отдаёт поле, а пустое значение
приходит как null (это видно по типу «T | null»).
"""

from __future__ import annotations

import json
import re
from typing import Any

HEADER = (
    "// Сгенерировано из схемы OpenAPI (app/api/contract.py). Руками не править:\n"
    "// uv run python -m app.api.contract  (или npm run gen:api в app/frontend)\n"
)

PRIMITIVES = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
}


def type_name(name: str) -> str:
    """Имя схемы → допустимое имя типа TypeScript."""
    return re.sub(r"\W", "_", name)


def ts_type(node: dict[str, Any]) -> str:
    """Тип TypeScript для узла JSON Schema."""
    if "$ref" in node:
        return type_name(node["$ref"].rsplit("/", 1)[-1])
    if "const" in node:
        return json.dumps(node["const"], ensure_ascii=False)
    if "enum" in node:
        return " | ".join(json.dumps(value, ensure_ascii=False) for value in node["enum"])
    for key in ("anyOf", "oneOf"):
        if key in node:
            return " | ".join(dict.fromkeys(ts_type(item) for item in node[key]))
    kind = node.get("type")
    if kind == "array":
        item = ts_type(node.get("items", {}))
        return f"({item})[]" if " | " in item else f"{item}[]"
    if kind == "object":
        extra = node.get("additionalProperties")
        value = ts_type(extra) if isinstance(extra, dict) else "unknown"
        keys = node.get("propertyNames")
        return f"Record<{ts_type(keys) if keys else 'string'}, {value}>"
    return PRIMITIVES.get(kind, "unknown")


def ts_interface(name: str, schema: dict[str, Any]) -> str:
    """Интерфейс для схемы объекта; первая строка описания — комментарий."""
    lines = []
    description = (schema.get("description") or "").strip()
    if description:
        lines.append(f"/** {description.splitlines()[0].replace('*/', '* /')} */")
    lines.append(f"export interface {type_name(name)} {{")
    lines += [f"  {prop}: {ts_type(node)};" for prop, node in schema.get("properties", {}).items()]
    lines.append("}")
    return "\n".join(lines)


def get_responses(schema: dict[str, Any]) -> list[tuple[str, str]]:
    """GET-пути API и тип их успешного JSON-ответа."""
    out = []
    for path, operations in schema.get("paths", {}).items():
        responses = operations.get("get", {}).get("responses", {})
        content = responses.get("200", {}).get("content", {}).get("application/json")
        if content is not None:
            out.append((path, ts_type(content.get("schema", {}))))
    return out


def typescript(schema: dict[str, Any]) -> str:
    """Текст schema.ts: интерфейсы ответов и карта «GET-путь → тип ответа»."""
    parts = [HEADER]
    for name, component in schema.get("components", {}).get("schemas", {}).items():
        parts.append(ts_interface(name, component) + "\n")
    routes = [f"  {json.dumps(path)}: {ts};" for path, ts in get_responses(schema)]
    parts.append(
        "\n".join(
            [
                "/** GET-пути API и тип ответа: apiGet('/api/data/status') вернёт InventoryReport. */",
                "export interface ApiGet {",
                *routes,
                "}",
            ]
        )
        + "\n"
    )
    return "\n".join(parts)
