"""Проверки API пульта. Запуск: uv run pytest app

Приложение создаётся с временными папками данных и фронтенда, реальные data/
и app/frontend/dist не трогаются.
"""

# Даты в данных — naive MSK, как в data/raw.
# ruff: noqa: DTZ001

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    path = root / "raw/futures/brent/BRX6.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"timestamp": [datetime(2026, 9, 1, 10), datetime(2026, 9, 11, 18)]}).write_parquet(path)
    return root


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    folder = tmp_path / "dist"
    (folder / "assets").mkdir(parents=True)
    (folder / "index.html").write_text("<!doctype html><title>cabinet</title>")
    (folder / "assets/app.js").write_text("console.log('cabinet')")
    (tmp_path / "secret.txt").write_text("outside dist")
    return folder


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    return tmp_path / "settings"


@pytest.fixture
def client(data_root: Path, dist: Path, settings: Path) -> TestClient:
    return TestClient(create_app(data_root=data_root, frontend_dist=dist, settings_dir=settings))


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_data_status_is_inventory_of_given_folder(client):
    response = client.get("/api/data/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"generated_at", "expected_date", "data_root", "sections", "datasets", "summary"}
    brent = next(row for row in body["datasets"] if row["id"] == "raw/futures/brent")
    # Даты — строки ISO, пустые поля приходят как null, а не пропадают.
    assert (brent["first_date"], brent["last_date"]) == ("2026-09-01", "2026-09-11")
    assert brent["note"] is None
    assert brent["facts"] == [{"label": "Контрактов", "value": 1}]
    futures = next(section for section in body["sections"] if section["id"] == "futures")
    assert futures["instruments"][0]["title"] == "Нефть Brent"
    assert futures["instruments"][0]["datasets"] == ["raw/futures/brent"]
    assert body["generated_at"].endswith("+03:00")  # время проверки по Москве
    assert sum(body["summary"].values()) == len(body["datasets"])


def test_large_responses_are_gzipped(client):
    response = client.get("/api/data/status", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"


def test_instrument_order_starts_empty(client, settings):
    assert client.get("/api/settings/instrument-order").json() == {"sections": {}}
    assert not settings.exists()  # чтение файл не создаёт


def test_instrument_order_patch_replaces_only_given_sections(client, settings):
    client.patch("/api/settings/instrument-order", json={"sections": {"futures": ["futures/si", "futures/brent"]}})
    response = client.patch("/api/settings/instrument-order", json={"sections": {"stocks": ["stocks/SBER"]}})
    expected = {"sections": {"futures": ["futures/si", "futures/brent"], "stocks": ["stocks/SBER"]}}
    assert response.json() == expected
    assert client.get("/api/settings/instrument-order").json() == expected
    assert json.loads((settings / "instrument_order.json").read_text()) == expected
    assert [path.name for path in settings.iterdir()] == ["instrument_order.json"]  # временных файлов не осталось
    assert (settings / "instrument_order.json").stat().st_mode & 0o777 == 0o644


def test_instrument_order_drops_repeats(client):
    body = {"sections": {"futures": ["futures/si", "futures/brent", "futures/si"]}}
    assert client.patch("/api/settings/instrument-order", json=body).json()["sections"]["futures"] == [
        "futures/si",
        "futures/brent",
    ]


def test_instrument_order_rejects_bad_body(client):
    response = client.patch("/api/settings/instrument-order", json={"sections": {"futures": "futures/si"}})
    assert response.status_code == 422


def test_broken_order_file_is_reported_not_overwritten(client, settings):
    settings.mkdir()
    (settings / "instrument_order.json").write_text("{не json")
    assert client.get("/api/settings/instrument-order").status_code == 500
    response = client.patch("/api/settings/instrument-order", json={"sections": {"futures": []}})
    assert response.status_code == 500
    assert "повреждён" in response.json()["detail"]
    assert (settings / "instrument_order.json").read_text() == "{не json"


def test_unknown_api_path_is_json_404_not_frontend(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_frontend_files_and_spa_fallback(client):
    assert "<title>cabinet</title>" in client.get("/").text
    assert client.get("/assets/app.js").text == "console.log('cabinet')"
    # Маршрут внутри SPA: сервер отдаёт index.html, дальше разбирается фронтенд.
    deep = client.get("/experiments/42")
    assert deep.status_code == 200
    assert "<title>cabinet</title>" in deep.text


def test_frontend_never_serves_files_outside_dist(client):
    response = client.get("/..%2Fsecret.txt")
    assert "outside dist" not in response.text


def test_frontend_not_built_explains_how_to_build(data_root, tmp_path):
    client = TestClient(create_app(data_root=data_root, frontend_dist=tmp_path / "missing"))
    response = client.get("/")
    assert response.status_code == 503
    assert "npm run build" in response.text
    assert client.get("/api/health").status_code == 200  # API работает и без фронтенда
