"""Минутные свечи контрактов для разбора неоднозначных 15-минутных баров.

Если 15-минутный бар задел и стоп, и цель, по самому бару не узнать, что было
раньше. Минутки того же контракта за этот интервал показывают порядок.
15-минутные бары собраны из этих же минуток (`build_continuous_futures.py`):
бар с меткой 10:00 — минутки 10:00…10:14 контракта из столбца `contract`.

Файлы raw только читаются: data/raw/futures/<рынок>/<контракт>.parquet.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FUTURES_DIR = PROJECT_ROOT / "data" / "raw" / "futures"
BAR_LENGTH = timedelta(minutes=15)
MINUTE_COLUMNS = ["timestamp", "open", "high", "low", "close"]


class RawMinutes:
    """Минутки контрактов одного рынка из каталога raw.

    Файл контракта читается один раз и держится в памяти. Сделки идут по
    времени, поэтому хватает нескольких последних контрактов.
    """

    def __init__(self, directory: Path, cached_contracts: int = 4) -> None:
        if not directory.is_dir():
            raise FileNotFoundError(f"Нет каталога минуток {directory}")
        if cached_contracts < 1:
            raise ValueError("В памяти должен помещаться хотя бы один контракт")
        self.directory = directory
        self.cached_contracts = cached_contracts
        self._cache: dict[str, pl.DataFrame] = {}

    @classmethod
    def for_market(cls, market: str) -> RawMinutes:
        """Минутки рынка по имени каталога в data/raw/futures (brent, gold...)."""
        return cls(RAW_FUTURES_DIR / market)

    def bar_minutes(self, contract: str, start: datetime) -> pl.DataFrame:
        """Минутки контракта внутри бара [start, start + 15 мин), по времени."""
        return self._contract(contract).filter(
            pl.col("timestamp").is_between(start, start + BAR_LENGTH, closed="left")
        )

    def _contract(self, contract: str) -> pl.DataFrame:
        if contract not in self._cache:
            path = self.directory / f"{contract}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Нет минуток контракта {contract}: {path}")
            if len(self._cache) >= self.cached_contracts:
                # Вытесняется контракт, прочитанный раньше всех.
                self._cache.pop(next(iter(self._cache)))
            frame = pl.read_parquet(path, columns=MINUTE_COLUMNS)
            self._cache[contract] = frame.sort("timestamp")
        return self._cache[contract]
