"""Каталог инвентаря: как читать справочники и что к какому инструменту относится.

Связи записаны явно, чтобы их было видно и легко поправить: справочник
попадает в карточку инструмента целиком (курс клиринга — у долларовых
фьючерсов) или срезом по ключу (ставки ГО по asset_code BR — у нефти).
Допустимое отставание наборов и его причины — в rules.json (rules.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.instruments import ASSETS, Sector

REBUILD_HINT = "uv run python src/data/build_continuous_futures.py"
CONTRACTS_HINT = "uv run python src/data/download_reference.py --update --only contracts"


@dataclass(frozen=True)
class ReferenceSpec:
    """Как читать файл справочника. Правило свежести — в rules.json по имени файла."""

    file: str
    description: str
    date_column: str | None  # колонка даты; None — без дат
    count_column: str | None = None  # что посчитать уникальным у файла целиком
    count_label: str | None = None


REFERENCE_FILES: tuple[ReferenceSpec, ...] = (
    ReferenceSpec("margin_rates.parquet", "Минимальные ставки ГО mr1–mr3", "date", "asset_code", "Рынков"),
    ReferenceSpec(
        "margin_snapshot.parquet",
        "Снимки фактического ГО (копятся при обновлении)",
        "snapshot_date",
        "snapshot_date",
        "Снимков",
    ),
    ReferenceSpec("rates_moex_clearing.parquet", "Курс клиринга MOEX USD/RUB", "date"),
    ReferenceSpec("rates_cbr_fx.parquet", "Официальные курсы ЦБ", "date", "currency", "Валют"),
    ReferenceSpec("cbr_key_rate.parquet", "Ключевая ставка ЦБ", "date"),
    ReferenceSpec("rates_usd.parquet", "Ставка SOFR", "date"),
    ReferenceSpec("rates_china.parquet", "Ставки Китая", "date", "series", "Рядов"),
    ReferenceSpec("spot_currency.parquet", "Спот валют MOEX", "date", "secid", "Инструментов"),
    ReferenceSpec("indices_daily.parquet", "Индексы MOEX", "date", "secid", "Индексов"),
    ReferenceSpec("benchmarks_world.parquet", "Мировые бенчмарки", "date", "symbol", "Инструментов"),
    ReferenceSpec("eia_ng_storage.parquet", "Запасы газа в США (EIA)", "release_date"),
    ReferenceSpec("futures_contracts.parquet", "Справочник контрактов", "first_trade_date", "asset_code", "Активов"),
    ReferenceSpec("calendar_events.parquet", "Календарь событий", "date"),
    ReferenceSpec("corp_dividends.parquet", "Дивиденды", "declared_date", "ticker", "Тикеров"),
    ReferenceSpec("corp_listing.parquet", "Листинг акций", None, "ticker", "Тикеров"),
    ReferenceSpec("corp_splits.parquet", "Сплиты акций", "date"),
    ReferenceSpec("ru_working_saturdays.json", "Рабочие субботы (торги идут)", None),
)
REFERENCE_BY_FILE = {spec.file: spec for spec in REFERENCE_FILES}


@dataclass(frozen=True)
class RefLink:
    """Справочник в карточке инструмента: целиком (column None) или срез column == value."""

    file: str
    column: str | None = None
    value: str | None = None
    fact_column: str | None = None  # что посчитать уникальным внутри среза
    fact_label: str | None = None


# Рубли за пункт у этих фьючерсов = цена в долларах × курс клиринга.
USD_QUOTED_SECTORS = {Sector.COMMODITY, Sector.FOREIGN_INDEX, Sector.CRYPTO}
USD_QUOTED_MARKETS = {"eurusd"}

# Что ещё нужно для справедливой цены (базиса): спот и ставки для валют,
# индекс для индексных фьючерсов, запасы газа для газа.
EXTRA_LINKS: dict[str, tuple[RefLink, ...]] = {
    "si": (
        RefLink("spot_currency.parquet", "secid", "USD000UTSTOM"),
        RefLink("rates_cbr_fx.parquet", "currency", "USD"),
        RefLink("cbr_key_rate.parquet"),
        RefLink("rates_usd.parquet"),
    ),
    "cnyrub": (
        RefLink("spot_currency.parquet", "secid", "CNYRUB_TOM"),
        RefLink("rates_cbr_fx.parquet", "currency", "CNY"),
        RefLink("cbr_key_rate.parquet"),
        RefLink("rates_china.parquet"),
    ),
    "euro": (
        RefLink("spot_currency.parquet", "secid", "EUR_RUB__TOM"),
        RefLink("rates_cbr_fx.parquet", "currency", "EUR"),
        RefLink("cbr_key_rate.parquet"),
    ),
    "mix": (RefLink("indices_daily.parquet", "secid", "IMOEX"),),
    "mix_mini": (RefLink("indices_daily.parquet", "secid", "IMOEX"),),
    "rts": (RefLink("indices_daily.parquet", "secid", "RTSI"),),
    "ng": (RefLink("eia_ng_storage.parquet"),),
}


def market_links(name: str) -> list[RefLink]:
    """Справочники для карточки фьючерса. Пустой срез строки не даст."""
    links: list[RefLink] = []
    spec = ASSETS.get(name)
    if spec is not None:
        code = spec.asset_code
        links += [
            RefLink("margin_rates.parquet", "asset_code", code),
            RefLink("margin_snapshot.parquet", "asset_code", code, "snapshot_date", "Снимков"),
            RefLink("futures_contracts.parquet", "asset_code", code, "ticker", "Контрактов"),
        ]
        if spec.sector in USD_QUOTED_SECTORS or name in USD_QUOTED_MARKETS:
            links.append(RefLink("rates_moex_clearing.parquet"))
    links.append(RefLink("benchmarks_world.parquet", "asset", name))
    links += EXTRA_LINKS.get(name, ())
    return links


# Базовые акции фьючерсов: дивиденды и сплиты двигают цену фьючерса.
# Старые тикеры — потому что архив Algopack (2020–2023) лежит под ними.
UNDERLYING_STOCKS: dict[str, tuple[str, ...]] = {
    "sber": ("SBER",),
    "gazp": ("GAZP",),
    "lkoh": ("LKOH",),
    "yandex": ("YDEX", "YNDX"),
    "ozon": ("OZON",),
    "tbank": ("T", "TCSG"),
}


def stock_links(ticker: str) -> list[RefLink]:
    """Справочники для карточки акции."""
    return [
        RefLink("corp_dividends.parquet", "ticker", ticker),
        RefLink("corp_splits.parquet", "ticker", ticker),
    ]


def futoi_market(ticker: str) -> str | None:
    """Рынок реестра по тикеру FUTOI: тикеры вечных фьючерсов или префикс контрактов (Si = SI)."""
    for name, spec in ASSETS.items():
        if ticker in spec.fixed_tickers:
            return name
        prefixes = {prefix.upper() for prefix in (spec.ticker_prefix, *spec.extra_prefixes) if prefix}
        if ticker.upper() in prefixes:
            return name
    return None


SECTOR_TITLES = {
    Sector.CURRENCY: "валюта",
    Sector.COMMODITY: "сырьё",
    Sector.INDEX: "индекс",
    Sector.EQUITY: "акция",
    Sector.FOREIGN_INDEX: "иностранный индекс",
    Sector.PERPETUAL: "вечные",
    Sector.CRYPTO: "крипто",
}

ALGOPACK_DATASETS = {"tradestats": "сделки", "orderstats": "заявки", "obstats": "стакан"}
ALGOPACK_MARKETS = {"eq": "Акции", "fo": "Фьючерсы", "fx": "Валюта"}

CONTINUOUS_ALL = "continuous_futures_15m.parquet"
CONTINUOUS_SERVICE = {
    CONTINUOUS_ALL: ("Все рынки одним файлом", "date"),
    "daily_contract_selection.parquet": ("Выбранный контракт по дням", "date"),
}
