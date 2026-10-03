"""Проверки чтения минуток. Запуск: uv run pytest src/backtest/test_minutes.py"""

# минутки в raw — naive timestamp MSK, как и бары в data/continuous_15m.
# ruff: noqa: DTZ001

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest.minutes import RawMinutes

BAR_START = datetime(2026, 9, 10, 10, 15)


def write_contract(
    directory: Path, contract: str, offsets: list[int], base: float = 100.0
) -> None:
    """Файл контракта в формате raw; минутка с отступом m стоит base + m."""
    pl.DataFrame(
        {
            "timestamp": [BAR_START + timedelta(minutes=m) for m in offsets],
            "open": [base + m for m in offsets],
            "high": [base + m + 1 for m in offsets],
            "low": [base + m - 1 for m in offsets],
            "close": [base + m for m in offsets],
            "volume": [1 for _ in offsets],
            "secid": [contract for _ in offsets],
        }
    ).write_parquet(directory / f"{contract}.parquet")


def minute_offsets(frame: pl.DataFrame) -> list[int]:
    return [
        int((timestamp - BAR_START).total_seconds() // 60)
        for timestamp in frame["timestamp"]
    ]


def test_bar_minutes_take_only_the_bar_interval(tmp_path):
    # Бар 10:15 — минутки 10:15…10:29; 10:14 и 10:30 — соседние бары.
    write_contract(tmp_path, "TSZ6", [-1, 0, 1, 14, 15])
    window = RawMinutes(tmp_path).bar_minutes("TSZ6", BAR_START)
    assert minute_offsets(window) == [0, 1, 14]
    assert window.columns == ["timestamp", "open", "high", "low", "close"]


def test_bar_minutes_are_sorted_by_time(tmp_path):
    write_contract(tmp_path, "TSZ6", [14, 0, 7])
    window = RawMinutes(tmp_path).bar_minutes("TSZ6", BAR_START)
    assert minute_offsets(window) == [0, 7, 14]


def test_contracts_are_not_mixed(tmp_path):
    # В памяти помещается один контракт: каждое чтение вытесняет предыдущий.
    write_contract(tmp_path, "TSZ6", [0], base=100.0)
    write_contract(tmp_path, "TSH7", [0], base=200.0)
    minutes = RawMinutes(tmp_path, cached_contracts=1)
    assert minutes.bar_minutes("TSZ6", BAR_START)["open"].to_list() == [100.0]
    assert minutes.bar_minutes("TSH7", BAR_START)["open"].to_list() == [200.0]
    assert minutes.bar_minutes("TSZ6", BAR_START)["open"].to_list() == [100.0]


def test_missing_contract_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="TSZ6"):
        RawMinutes(tmp_path).bar_minutes("TSZ6", BAR_START)


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Нет каталога"):
        RawMinutes(tmp_path / "brent")


def test_cache_needs_room_for_one_contract(tmp_path):
    with pytest.raises(ValueError, match="хотя бы один"):
        RawMinutes(tmp_path, cached_contracts=0)
