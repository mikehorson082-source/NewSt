"""Константы и пути уровня ядра (core).

Сетевые параметры, тайминги сессий MOEX, пути к слоям данных.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.instruments import (
    ASSETS,
    AssetSpec,
    Cadence,
    Sector,
    get_asset,
    list_assets,
)

# Базовые пути проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Параметры API MOEX ISS
ISS_BASE_URL = "https://iss.moex.com/iss"
DEFAULT_ENGINE = "futures"
DEFAULT_MARKET = "forts"
SHARES_ENGINE = "stock"
SHARES_MARKET = "shares"
SHARES_BOARD = "TQBR"
DEFAULT_BOARD = "TQBR"
INDEX_BOARD = "SNDX"


# Сетевые настройки
PAGE_SIZE = 100
REQUEST_DELAY = 0.1  # секунды
REQUEST_TIMEOUT = 30.0  # секунды
MAX_RETRIES = 5
CHECKPOINT_EVERY = 25_000

# Переход акций и облигаций на T+1; до этой даты расчёты шли в T+2
T1_SWITCH_DATE = pd.Timestamp("2023-07-31")

# Расписание торговых сессий
MAIN_SESSION_START = "09:50"
MAIN_SESSION_END = "18:50"
CONTINUOUS_START = "10:00"
CLOSING_AUCTION_START = "18:40"
EVENING_SESSION_START = "19:00"

# Порог определения простоя (2022 год, редомициляция)
LONG_GAP_CALENDAR_DAYS = 12


@dataclass(frozen=True)
class Paths:
    """Раскладка слоёв данных: raw -> clean -> reports; manual — ручные таблицы."""

    root: Path = Path("data")

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def raw_futures_min(self) -> Path:
        return self.raw / "futures"

    @property
    def raw_futures_daily(self) -> Path:
        return self.raw / "futures_daily"

    @property
    def raw_futoi(self) -> Path:
        return self.raw / "futoi"

    @property
    def raw_futoi_5min(self) -> Path:
        return self.raw_futoi / "5min"

    @property
    def raw_futoi_daily(self) -> Path:
        return self.raw_futoi / "daily"

    @property
    def clean(self) -> Path:
        return self.root / "clean"

    @property
    def manual(self) -> Path:
        return self.root / "manual"

    @property
    def reference(self) -> Path:
        return self.root / "reference"

    @property
    def reports(self) -> Path:
        return self.root / "reports"
