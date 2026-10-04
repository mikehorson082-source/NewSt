"""Проверки API пульта. Запуск: uv run pytest app

Приложение создаётся с временными папками данных и фронтенда, реальные data/
и app/frontend/dist не трогаются.
"""

# Даты в данных — naive MSK, как в data/raw.
# ruff: noqa: DTZ001

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
def client(data_root: Path, dist: Path) -> TestClient:
    return TestClient(create_app(data_root=data_root, frontend_dist=dist))


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_data_status_is_inventory_of_given_folder(client):
    response = client.get("/api/data/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"generated_at", "expected_date", "data_root", "groups", "datasets", "summary"}
    brent = next(row for row in body["datasets"] if row["id"] == "raw/futures/brent")
    # Даты — строки ISO, пустые поля приходят как null, а не пропадают.
    assert (brent["first_date"], brent["last_date"]) == ("2026-09-01", "2026-09-11")
    assert brent["files"] == 1
    assert brent["note"] is None
    assert brent["facts"] == []
    assert body["generated_at"].endswith("+03:00")  # время проверки по Москве
    assert sum(body["summary"].values()) == len(body["datasets"])


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
