"""Справочник активов и спецификаций фьючерсов Московской биржи (FORTS).

Единый источник правды для всех модулей системы (data, features, dataset, backtest).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Sector(str, Enum):
    """Сектора рынка фьючерсов."""

    CURRENCY = "currency"
    COMMODITY = "commodity"
    INDEX = "index"
    EQUITY = "equity"
    FOREIGN_INDEX = "foreign_index"
    PERPETUAL = "perpetual"
    CRYPTO = "crypto"


class Cadence(str, Enum):
    """Периодичность экспирации контрактов."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    PERPETUAL = "perpetual"


MONTH_CODES_QUARTERLY = ["H", "M", "U", "Z"]  # Март, Июнь, Сентябрь, Декабрь
MONTH_CODES_ALL = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]  # Все 12 месяцев


@dataclass(frozen=True)
class AssetSpec:
    """Спецификация класса базового актива срочного рынка."""

    name: str  # Внутренний системный идентификатор (si, brent, gold...)
    short_name: str  # Человекочитаемое название
    asset_code: str  # Биржевой код базового актива MOEX (Si, BR, GOLD, RTS...)
    ticker_prefix: str  # Префикс кодов контрактов (Si, BR, GD, RI, CC...)
    sector: Sector  # Сектор рынка
    cadence: Cadence  # Ежемесячный или квартальный
    fixed_tickers: list[str] = field(default_factory=list)  # Для вечных фьючерсов

    def generate_tickers(self, years: list[int] | None = None) -> list[str]:
        """Генерирует список стандартных биржевых кодов контрактов для заданных годов."""
        if self.cadence == Cadence.PERPETUAL:
            return list(self.fixed_tickers)

        if years is None:
            years = [1, 2, 3, 4, 5, 6, 7]  # 2021-2027

        months = MONTH_CODES_ALL if self.cadence == Cadence.MONTHLY else MONTH_CODES_QUARTERLY
        return [f"{self.ticker_prefix}{m}{y}" for y in years for m in months]


# =====================================================================
# Реестр всех 25 ликвидных классов активов MOEX FORTS
# =====================================================================
ASSETS: dict[str, AssetSpec] = {
    # ------------------ Валюта ------------------
    "si": AssetSpec(
        name="si",
        short_name="Доллар США / Рубль",
        asset_code="Si",
        ticker_prefix="Si",
        sector=Sector.CURRENCY,
        cadence=Cadence.QUARTERLY,
    ),
    "cnyrub": AssetSpec(
        name="cnyrub",
        short_name="Китайский юань / Рубль",
        asset_code="CNY",
        ticker_prefix="CR",
        sector=Sector.CURRENCY,
        cadence=Cadence.QUARTERLY,
    ),
    "euro": AssetSpec(
        name="euro",
        short_name="Евро / Рубль",
        asset_code="Eu",
        ticker_prefix="Eu",
        sector=Sector.CURRENCY,
        cadence=Cadence.QUARTERLY,
    ),
    "eurusd": AssetSpec(
        name="eurusd",
        short_name="Евро / Доллар США",
        asset_code="ED",
        ticker_prefix="ED",
        sector=Sector.CURRENCY,
        cadence=Cadence.QUARTERLY,
    ),
    "perpetual": AssetSpec(
        name="perpetual",
        short_name="Вечные фьючерсы (USD, CNY, EUR, Золото, IMOEX)",
        asset_code="PERPETUAL",
        ticker_prefix="",
        sector=Sector.PERPETUAL,
        cadence=Cadence.PERPETUAL,
        fixed_tickers=["USDRUBF", "CNYRUBF", "EURRUBF", "GLDRUBF", "IMOEXF"],
    ),
    # ------------------ Товарный рынок (Сырье) ------------------
    "brent": AssetSpec(
        name="brent",
        short_name="Нефть Brent",
        asset_code="BR",
        ticker_prefix="BR",
        sector=Sector.COMMODITY,
        cadence=Cadence.MONTHLY,
    ),
    "ng": AssetSpec(
        name="ng",
        short_name="Природный газ",
        asset_code="NG",
        ticker_prefix="NG",
        sector=Sector.COMMODITY,
        cadence=Cadence.MONTHLY,
    ),
    "gold": AssetSpec(
        name="gold",
        short_name="Золото",
        asset_code="GOLD",
        ticker_prefix="GD",
        sector=Sector.COMMODITY,
        cadence=Cadence.QUARTERLY,
    ),
    "silver": AssetSpec(
        name="silver",
        short_name="Серебро",
        asset_code="SILV",
        ticker_prefix="SV",
        sector=Sector.COMMODITY,
        cadence=Cadence.QUARTERLY,
    ),
    "platinum": AssetSpec(
        name="platinum",
        short_name="Платина",
        asset_code="PLT",
        ticker_prefix="PT",
        sector=Sector.COMMODITY,
        cadence=Cadence.QUARTERLY,
    ),
    "palladium": AssetSpec(
        name="palladium",
        short_name="Палладий",
        asset_code="PLD",
        ticker_prefix="PD",
        sector=Sector.COMMODITY,
        cadence=Cadence.QUARTERLY,
    ),
    "copper": AssetSpec(
        name="copper",
        short_name="Медь",
        asset_code="CE",
        ticker_prefix="CE",
        sector=Sector.COMMODITY,
        cadence=Cadence.QUARTERLY,
    ),
    "cocoa": AssetSpec(
        name="cocoa",
        short_name="Какао",
        asset_code="COCOA",
        ticker_prefix="CC",
        sector=Sector.COMMODITY,
        cadence=Cadence.MONTHLY,
    ),
    # ------------------ Фондовые индексы ------------------
    "rts": AssetSpec(
        name="rts",
        short_name="Индекс РТС",
        asset_code="RTS",
        ticker_prefix="RI",
        sector=Sector.INDEX,
        cadence=Cadence.QUARTERLY,
    ),
    "mix": AssetSpec(
        name="mix",
        short_name="Индекс Мосбиржи (стандарт)",
        asset_code="MIX",
        ticker_prefix="MX",
        sector=Sector.INDEX,
        cadence=Cadence.QUARTERLY,
    ),
    "mix_mini": AssetSpec(
        name="mix_mini",
        short_name="Мини-Индекс Мосбиржи",
        asset_code="MXI",
        ticker_prefix="MM",
        sector=Sector.INDEX,
        cadence=Cadence.QUARTERLY,
    ),
    # ------------------ Голубые фишки РФ ------------------
    "sber": AssetSpec(
        name="sber",
        short_name="Сбербанк",
        asset_code="SBRF",
        ticker_prefix="SR",
        sector=Sector.EQUITY,
        cadence=Cadence.QUARTERLY,
    ),
    "gazp": AssetSpec(
        name="gazp",
        short_name="Газпром",
        asset_code="GAZR",
        ticker_prefix="GZ",
        sector=Sector.EQUITY,
        cadence=Cadence.QUARTERLY,
    ),
    "lkoh": AssetSpec(
        name="lkoh",
        short_name="Лукойл",
        asset_code="LKOH",
        ticker_prefix="LK",
        sector=Sector.EQUITY,
        cadence=Cadence.QUARTERLY,
    ),
    "yandex": AssetSpec(
        name="yandex",
        short_name="Яндекс",
        asset_code="YD",
        ticker_prefix="YD",
        sector=Sector.EQUITY,
        cadence=Cadence.QUARTERLY,
    ),
    "ozon": AssetSpec(
        name="ozon",
        short_name="Озон",
        asset_code="OZON",
        ticker_prefix="ON",
        sector=Sector.EQUITY,
        cadence=Cadence.QUARTERLY,
    ),
    "tbank": AssetSpec(
        name="tbank",
        short_name="Т-Банк",
        asset_code="TCS",
        ticker_prefix="TI",
        sector=Sector.EQUITY,
        cadence=Cadence.QUARTERLY,
    ),
    # ------------------ Зарубежные бенчмарки ------------------
    "spy": AssetSpec(
        name="spy",
        short_name="SPDR S&P 500 ETF",
        asset_code="SPYF",
        ticker_prefix="SF",
        sector=Sector.FOREIGN_INDEX,
        cadence=Cadence.QUARTERLY,
    ),
    "nasdaq": AssetSpec(
        name="nasdaq",
        short_name="Invesco QQQ (Nasdaq 100)",
        asset_code="NASD",
        ticker_prefix="NA",
        sector=Sector.FOREIGN_INDEX,
        cadence=Cadence.QUARTERLY,
    ),
    # ------------------ Криптоактивы ------------------
    "btc": AssetSpec(
        name="btc",
        short_name="Биткоин",
        asset_code="BTC",
        ticker_prefix="BT",
        sector=Sector.CRYPTO,
        cadence=Cadence.MONTHLY,
    ),
}


def get_asset(name: str) -> AssetSpec:
    """Возвращает спецификацию актива по внутреннему имени."""
    key = name.lower()
    if key not in ASSETS:
        raise KeyError(f"Неизвестный актив: {name}. Доступные: {list(ASSETS.keys())}")
    return ASSETS[key]


def list_assets(sector: Sector | str | None = None) -> list[AssetSpec]:
    """Возвращает список активов (опционально с фильтром по сектору)."""
    if sector is None:
        return list(ASSETS.values())
    sector_val = Sector(sector) if isinstance(sector, str) else sector
    return [a for a in ASSETS.values() if a.sector == sector_val]
