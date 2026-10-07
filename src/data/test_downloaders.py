"""Сбои загрузчиков не прячутся. Запуск: uv run pytest src/data/test_downloaders.py

Сеть подменена: сессия отдаёт заранее заданные ответы. Главное правило —
сбой источника становится ошибкой (и кодом выхода 1), а не «данных нет»:
иначе при --update файл молча считался бы актуальным, а у FUTOI оставалась
бы дыра, которую следующая докачка уже не заметит.
"""

# ruff: noqa: DTZ001 — даты в данных naive MSK, как в data/raw

import argparse
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest
import requests

from data import download_futoi as futoi
from data import download_futures_daily as daily
from data import download_futures_min as minutes
from data import download_reference as reference


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """Отдаёт ответы по очереди; последний повторяется. Исключение в списке — бросается."""

    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get(self, *_args, **_kwargs) -> FakeResponse:
        self.calls += 1
        response = self.responses[min(self.calls, len(self.responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    for module in (minutes, daily, futoi):
        monkeypatch.setattr(module, "REQUEST_DELAY", 0)


EMPTY_CANDLES = FakeResponse(200, {"candles": {"columns": ["begin"], "data": []}})


# ---------------------------------------------------------------- минутки и дневки
def test_unknown_contract_is_empty_not_error():
    """ISS отдаёт несуществующий контракт как 200 с пустым списком — это не ошибка."""
    assert minutes.fetch_minute_candles("QQZ9", session=FakeSession(EMPTY_CANDLES)) == []


@pytest.mark.parametrize("status", [503, 404])
def test_minutes_http_failure_raises(status):
    session = FakeSession(FakeResponse(status))
    with pytest.raises(requests.HTTPError, match=f"HTTP {status}"):
        minutes.fetch_minute_candles("SiZ6", session=session)
    assert session.calls == (minutes.MAX_RETRIES if status == 503 else 1)  # 503 — повторяем, 404 — сразу


def test_minutes_retry_then_success():
    session = FakeSession(FakeResponse(503), EMPTY_CANDLES)
    assert minutes.fetch_minute_candles("SiZ6", session=session) == []
    assert session.calls == 2


def test_daily_http_failure_raises():
    with pytest.raises(requests.HTTPError, match="HTTP 502"):
        daily.fetch_contract_history("SiZ6", session=FakeSession(FakeResponse(502)))


def failing_fetch(*_args, **_kwargs):
    raise requests.HTTPError("SECID SiZ6: HTTP 503 на start=0")


def test_minutes_update_failure_is_error_and_file_untouched(tmp_path, monkeypatch):
    target = tmp_path / "si" / "SiZ6.parquet"
    target.parent.mkdir()
    pl.DataFrame({"timestamp": [datetime(2026, 10, 1, 10)], "close": [1.0]}).write_parquet(target)
    before = target.read_bytes()
    monkeypatch.setattr(minutes, "fetch_minute_candles", failing_fetch)
    result = minutes.process_contract_minutes("si", "SiZ6", tmp_path, update=True)
    assert result["status"] == "error"  # раньше — «skipped», и в итогах «Ошибок: 0»
    assert "HTTP 503" in result["error"]
    assert target.read_bytes() == before


def test_daily_update_failure_is_error_and_file_untouched(tmp_path, monkeypatch):
    target = tmp_path / "si" / "SiZ6.parquet"
    target.parent.mkdir()
    pl.DataFrame({"TRADEDATE": [date(2026, 10, 1)], "OPENPOSITION": [100]}).write_parquet(target)
    before = target.read_bytes()
    monkeypatch.setattr(daily, "fetch_contract_history", failing_fetch)
    result = daily.process_single_contract("si", "SiZ6", tmp_path, update=True)
    assert result["status"] == "error"
    assert target.read_bytes() == before


# ---------------------------------------------------------------- FUTOI
def test_futoi_chunk_failure_raises():
    with pytest.raises(requests.HTTPError, match="HTTP 503"):
        futoi.fetch_futoi_chunk("Si", "2026-09-01", "2026-09-02", FakeSession(FakeResponse(503)), {})
    with pytest.raises(requests.ConnectionError):
        futoi.fetch_futoi_chunk("Si", "2026-09-01", "2026-09-02", FakeSession(requests.ConnectionError("нет сети")), {})


def test_futoi_free_tier_cutoff_is_not_error():
    cutoff = FakeResponse(200, {"futoi": {"columns": ["ERROR_MESSAGE"], "data": [["Free users …"]]}})
    assert futoi.fetch_futoi_chunk("Si", "2026-10-01", "2026-10-02", FakeSession(cutoff), {}) == ([], ["ERROR_MESSAGE"], True)


def test_futoi_one_failed_chunk_keeps_ticker_file_untouched(tmp_path, monkeypatch):
    """Один из кусков по два дня упал — тикер не сохраняется, а не сохраняется с дырой."""
    paths = futoi.Paths(root=tmp_path)
    paths.raw_futoi_5min.mkdir(parents=True)
    file_5min = paths.raw_futoi_5min / "Si.parquet"
    pl.DataFrame({"tradedate": [date(2026, 9, 1)], "tradetime": ["10:00:00"], "clgroup": ["FIZ"]}).write_parquet(file_5min)
    before = file_5min.read_bytes()

    def chunk(ticker, from_d, till_d, session, headers):
        if from_d == "2026-09-04":
            raise requests.HTTPError("HTTP 503")
        row = [1, 1, from_d, "10:00:00", ticker, "FIZ", 1, 1, 0, 1, 0, "", from_d]
        return [row], ["SESS_ID"], False

    monkeypatch.setattr(futoi, "fetch_futoi_chunk", chunk)
    monkeypatch.setattr(futoi, "get_trading_dates", lambda *_: ["2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"])
    args = argparse.Namespace(force=False, update=True, workers=2)
    with pytest.raises(requests.HTTPError):
        futoi.update_ticker(1, 1, "Si", paths, args, "2020-01-01", "2026-09-20")
    assert file_5min.read_bytes() == before
    assert not (paths.raw_futoi_daily / "Si.parquet").exists()


# ---------------------------------------------------------------- справочники
def test_reference_source_failure_is_remembered_and_file_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "FAILED", [])
    path = tmp_path / "rates_china.parquet"
    pl.DataFrame({"date": [date(2026, 9, 1)], "value": [1.0]}).write_parquet(path)
    before = path.read_bytes()

    def fetch(_start, _till):
        raise requests.ConnectionError("akshare не отвечает")

    reference.refresh_reference_file(
        "Ставки Китая", path, fetch, from_date="2020-01-01", till_date="2026-10-06", update=True
    )
    assert reference.FAILED == ["rates_china.parquet"]
    assert path.read_bytes() == before


def test_reference_no_new_data_is_not_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(reference, "FAILED", [])
    reference.refresh_reference_file(
        "Ставки Китая",
        tmp_path / "rates_china.parquet",
        lambda _start, _till: pl.DataFrame(),
        from_date="2026-10-01",
        till_date="2026-10-06",
        update=False,
    )
    assert reference.FAILED == []


def test_reference_retries_connection_reset():
    """ISS оборвал соединение («Connection reset by peer») — справочник повторяет запрос."""
    import socket
    import struct
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    seen: list[int] = []

    def serve():
        for attempt in range(2):
            conn, _ = server.accept()
            conn.recv(4096)
            seen.append(attempt)
            if attempt == 0:  # первый раз — обрыв (RST), как у ISS
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            else:
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    session = reference.retrying_session()
    session.adapters["http://"].max_retries.backoff_factor = 0  # без пауз в тесте
    response = session.get(f"http://127.0.0.1:{port}/history.json", timeout=5)
    assert (response.status_code, response.json(), seen) == (200, {}, [0, 1])
