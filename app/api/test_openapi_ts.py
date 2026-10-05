"""Проверки генератора типов TypeScript. Запуск: uv run pytest app/api/test_openapi_ts.py"""

import pytest

from app.api.openapi_ts import ts_type, typescript


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"type": "string", "format": "date"}, "string"),
        ({"type": "integer"}, "number"),
        ({"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]}, "string | null"),
        # int | float | str: integer и number — оба number, дубль убирается.
        ({"anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}]}, "number | string"),
        ({"type": "string", "enum": ["ok", "stale"]}, '"ok" | "stale"'),
        ({"type": "array", "items": {"$ref": "#/components/schemas/Fact"}}, "Fact[]"),
        ({"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "(string | null)[]"),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, "Record<string, number>"),
        (
            {"type": "object", "additionalProperties": {"type": "integer"}, "propertyNames": {"enum": ["ok"]}},
            'Record<"ok", number>',
        ),
        ({"$ref": "#/components/schemas/Report-Output"}, "Report_Output"),
        ({}, "unknown"),
    ],
)
def test_ts_type(node, expected):
    assert ts_type(node) == expected


def test_interfaces_and_get_routes():
    schema = {
        "paths": {
            "/api/report": {
                "get": {
                    "responses": {
                        "200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Report"}}}}
                    }
                }
            },
            "/api/upload": {"post": {"responses": {"200": {}}}},
        },
        "components": {
            "schemas": {
                "Report": {
                    "description": "Отчёт.\n\nПодробности.",
                    "properties": {
                        "day": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "rows": {"type": "integer"},
                    },
                }
            }
        },
    }
    text = typescript(schema)
    assert "/** Отчёт. */\nexport interface Report {\n  day: string | null;\n  rows: number;\n}" in text
    assert 'export interface ApiGet {\n  "/api/report": Report;\n}' in text
    assert "/api/upload" not in text  # только GET-пути


def test_body_routes_get_their_own_map():
    order = {"$ref": "#/components/schemas/Order"}
    schema = {
        "paths": {
            "/api/order": {
                "get": {"responses": {"200": {"content": {"application/json": {"schema": order}}}}},
                "patch": {
                    "requestBody": {"content": {"application/json": {"schema": order}}},
                    "responses": {"200": {"content": {"application/json": {"schema": order}}}},
                },
            },
        },
        "components": {"schemas": {"Order": {"properties": {"ids": {"type": "array", "items": {"type": "string"}}}}}},
    }
    text = typescript(schema)
    assert 'export interface ApiGet {\n  "/api/order": Order;\n}' in text
    assert 'export interface ApiPatch {\n  "/api/order": { body: Order; response: Order };\n}' in text
    assert "ApiPost" not in text  # нет таких путей — нет и пустой карты
