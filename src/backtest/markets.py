"""Паспорта рынков плана st01 и параметры, которые меняются по датам.

Руками записаны только факты контракта из его спецификации на бирже:
код базового актива, валюта котировки, стоимость пункта в этой валюте
и ставка биржевого сбора (tariffs/tariffs.md, тариф ВТБ «Изумруд»: брокер 0).

Всё, что меняется со временем, считается из данных:
- шаг цены — из последнего снимка ГО (самый ликвидный контракт рынка);
- рубли за пункт по свече — стоимость пункта × курс клиринга, который
  рассчитывает эту свечу (у рублёвых контрактов курс = 1);
- доля ГО по дате — минимальная ставка mr1 на дату × коэффициент ГО
  самого ликвидного контракта из последнего снимка;
- обычный объём свечи — медиана объёма в то же время суток за прошлые
  20 дней (`usual_volume`), для лимита ликвидности в `sizing.py`;
- σ свечи — стандартное отклонение 15-минутной доходности NAV за прошлые
  20 дней (`bar_sigma`), для влияния на рынок в `costs.impact_points`;
- отметки расписания — последняя свеча дня и последняя свеча перед
  выходными или праздником (`is_day_end`, `is_before_break`), для выхода
  по расписанию в `position.py` (`Signal.hold`).

Факты контракта сверяются со снимком биржи: если стоимость шага в снимке
не сходится с записанной, паспорт не загружается.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from backtest.specs import ContractSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTINUOUS_DIR = PROJECT_ROOT / "data" / "continuous_15m"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
BAR_LENGTH = timedelta(minutes=15)

# Клиринг, который рассчитывает свечу: дневной в 14:00, вечерний в 19:00
# (с 2024 года вечерний 18:50–19:05, раньше 18:45–19:00). Свеча относится
# к первому клирингу, который наступает не раньше её конца. Вечерняя
# сессия и сессии выходных рассчитываются дневным клирингом следующего
# торгового дня.
CLEARING_TIME = {"tc": time(14, 0), "mc": time(19, 0)}
# С 23.03.2026 — единая торговая сессия: дневного клиринга нет, клиринг один,
# в 23:50, и переоценивает позиции по расчётной цене, которую биржа
# определяет в 19:00. Поэтому граница по ценам осталась в 19:00: свечи дня
# до 19:00 рассчитываются курсом этого дня (mc, фиксируется в 19:00), а
# вечерняя сессия и сессии выходных — курсом следующего торгового дня.
# Дневной курс tc биржа по-прежнему публикует, но с июня 2024 года это
# вчерашний вечерний курс, и клиринга по нему больше нет.
SINGLE_SESSION_START = date(2026, 3, 23)

# Биржевой сбор тейкера, доля от стоимости контракта (tariffs/tariffs.md).
CURRENCY_FEE = 0.0000462
INDEX_FEE = 0.0000660
COMMODITY_FEE = 0.0001320
STOCK_FEE = 0.0001980

# Допуск сверки стоимости шага со снимком биржи.
POINT_VALUE_TOLERANCE = 0.005
# Обычный объём свечи — медиана за 20 прошлых дней в то же время суток;
# меньше 5 дней истории — неизвестен.
USUAL_VOLUME_DAYS = 20
USUAL_VOLUME_MIN_DAYS = 5
# σ свечи — по 15-минутным доходностям NAV за 20 прошлых календарных дней;
# меньше 100 свечей истории — неизвестна.
BAR_SIGMA_WINDOW = "20d"
BAR_SIGMA_MIN_BARS = 100
SATURDAY = 6
TICK_TOLERANCE = 1e-6
# Хвост ряда, который биржа ещё не рассчитала, не длиннее недели: самый
# долгий перерыв между клирингами — Новый год, вечер 30.12 → 14:00 05.01,
# меньше 6 дней. Длиннее — справочник курсов устарел.
MAX_UNSETTLED_TAIL = timedelta(days=7)


@dataclass(frozen=True)
class Market:
    """Факты контракта, которые не меняются со временем."""

    name: str  # каталог в data/raw/futures и ряд data/continuous_15m
    asset_code: str  # код базового актива биржи (BR, GOLD, CNY, MIX)
    quote_currency: str  # валюта котировки: RUB или USD
    quote_point_value: float  # стоимость 1 пункта цены в валюте котировки
    exchange_fee_rate: float
    broker_fee_rate: float = 0.0
    slippage_ticks: float = 1.0

    def __post_init__(self) -> None:
        if self.quote_currency not in ("RUB", "USD"):
            raise ValueError(f"{self.name}: валюта котировки должна быть RUB или USD")
        if self.quote_point_value <= 0:
            raise ValueError(f"{self.name}: стоимость пункта должна быть > 0")


# Brent: 10 баррелей, цена в $ за баррель — $10 за пункт.
# Золото: 1 тройская унция, цена в $ — $1 за пункт.
# CNY: 1000 юаней, цена в ₽ за юань — 1000 ₽ за пункт.
# MIX: цена в пунктах индекса × 100, 1 ₽ за пункт.
MARKETS = {
    "cnyrub": Market("cnyrub", "CNY", "RUB", 1000.0, CURRENCY_FEE),
    "mix": Market("mix", "MIX", "RUB", 1.0, INDEX_FEE),
    "brent": Market("brent", "BR", "USD", 10.0, COMMODITY_FEE),
    "gold": Market("gold", "GOLD", "USD", 1.0, COMMODITY_FEE),
}

# Рынки экзамена (plan.md, 5.4): стратегия их не видела. Отдельно от
# `MARKETS`, чтобы прогоны «по всем рынкам» их не трогали.
# Si, Eu: 1000 долларов / евро, цена в ₽ за 1000 — 1 ₽ за пункт.
# Серебро: 10 тройских унций, цена в $ — $10 за пункт.
# Платина, палладий: 1 тройская унция, цена в $ — $1 за пункт.
# Медь: 10 кг, цена в $ за тонну — $0.01 за пункт.
EXAM_MARKETS = {
    "si": Market("si", "Si", "RUB", 1.0, CURRENCY_FEE),
    "euro": Market("euro", "Eu", "RUB", 1.0, CURRENCY_FEE),
    "silver": Market("silver", "SILV", "USD", 10.0, COMMODITY_FEE),
    "platinum": Market("platinum", "PLT", "USD", 1.0, COMMODITY_FEE),
    "palladium": Market("palladium", "PLD", "USD", 1.0, COMMODITY_FEE),
    "copper": Market("copper", "COPPER", "USD", 0.01, COMMODITY_FEE),
}

# Остальные рынки (plan.md, 5.4б): стратегия их тоже не видела.
# RTS: цена в пунктах индекса, $0.02 за пункт.
# Акции (SBRF, GAZR, LKOH, YDEX): цена в ₽ за лот — 1 ₽ за пункт.
# SPYF: цена в $, $1 за пункт. NASD: $0.01 за пункт.
# ED (EUR/USD): 1000 евро, цена в $ за евро — $1000 за пункт.
# NG: 100 MMBtu, цена в $ — $100 за пункт.
# COCOA: в снимке ГО шаг 0.1 стоит ровно 1 ₽ — 10 ₽ за пункт.
SCREEN_MARKETS = {
    "rts": Market("rts", "RTS", "USD", 0.02, INDEX_FEE),
    "sber": Market("sber", "SBRF", "RUB", 1.0, STOCK_FEE),
    "gazp": Market("gazp", "GAZR", "RUB", 1.0, STOCK_FEE),
    "lkoh": Market("lkoh", "LKOH", "RUB", 1.0, STOCK_FEE),
    "yandex": Market("yandex", "YDEX", "RUB", 1.0, STOCK_FEE),
    "spy": Market("spy", "SPYF", "USD", 1.0, INDEX_FEE),
    "nasdaq": Market("nasdaq", "NASD", "USD", 0.01, INDEX_FEE),
    "eurusd": Market("eurusd", "ED", "USD", 1000.0, CURRENCY_FEE),
    "ng": Market("ng", "NG", "USD", 100.0, COMMODITY_FEE),
    "cocoa": Market("cocoa", "COCOA", "RUB", 10.0, COMMODITY_FEE),
}
PASSPORTS = MARKETS | EXAM_MARKETS | SCREEN_MARKETS


@dataclass(frozen=True)
class MarketParams:
    """Параметры рынка из последнего снимка ГО биржи."""

    market: Market
    tick_size: float
    margin_factor: float
    snapshot_date: date
    snapshot_contract: str

    def contract_spec(self) -> ContractSpec:
        """Постоянные правила торговли для симулятора.

        Рубли за пункт и ГО симулятор берёт из баров `load_market`.
        """
        return ContractSpec(
            asset=self.market.name,
            tick_size=self.tick_size,
            exchange_fee_rate=self.market.exchange_fee_rate,
            broker_fee_rate=self.market.broker_fee_rate,
            slippage_ticks=self.market.slippage_ticks,
        )


def snapshot_params(
    market: Market,
    snapshot: pl.DataFrame,
    clearing_rates: pl.DataFrame | None = None,
) -> MarketParams:
    """Шаг цены и коэффициент ГО по самому ликвидному контракту снимка.

    Самый ликвидный — с наибольшим открытым интересом: у дальних контрактов
    ГО выше (Brent 03.10.2026: BRX6 21.7% стоимости, BRZ6 24.9%).
    Заодно сверяет стоимость пункта со стоимостью шага в снимке.
    """
    latest = snapshot.filter(
        (pl.col("asset_code") == market.asset_code)
        & pl.col("margin_factor").is_not_null()
        & pl.col("min_step").is_not_null()
        & pl.col("step_price").is_not_null()
    )
    if latest.is_empty():
        raise ValueError(
            f"{market.name}: в снимке ГО нет контрактов {market.asset_code}. "
            "Обновите справочники: download_reference.py --only margins"
        )
    latest = latest.filter(pl.col("snapshot_date") == pl.col("snapshot_date").max())
    row = latest.sort("open_interest", descending=True, nulls_last=True).row(
        0, named=True
    )

    rub_per_point = row["step_price"] / row["min_step"]
    rate = 1.0
    if market.quote_currency == "USD":
        if clearing_rates is None:
            raise ValueError(f"{market.name}: для сверки нужен курс клиринга USD/RUB")
        moment = row["im_time"] or datetime.combine(row["snapshot_date"], time(23, 59))
        known = clearing_rates.filter(pl.col("fixing_time") <= moment)
        if known.is_empty():
            raise ValueError(f"{market.name}: нет курса клиринга на дату снимка")
        rate = known.sort("fixing_time")["rate"][-1]
    expected = market.quote_point_value * rate
    if abs(rub_per_point / expected - 1.0) > POINT_VALUE_TOLERANCE:
        raise ValueError(
            f"{market.name}: стоимость пункта {market.quote_point_value} "
            f"{market.quote_currency} не сходится со снимком биржи "
            f"{row['secid']}: {rub_per_point:.4f} ₽ за пункт при курсе {rate}"
        )

    return MarketParams(
        market=market,
        tick_size=row["min_step"],
        margin_factor=row["margin_factor"],
        snapshot_date=row["snapshot_date"],
        snapshot_contract=row["secid"],
    )


def check_tick(bars: pl.DataFrame, tick_size: float, name: str = "") -> None:
    """Все реальные цены ряда лежат на сетке шага цены."""
    off_grid = bars.filter(
        pl.any_horizontal(
            [
                ((pl.col(c) / tick_size) - (pl.col(c) / tick_size).round()).abs()
                > TICK_TOLERANCE
                for c in ("open", "high", "low", "close")
            ]
        )
    )
    if not off_grid.is_empty():
        first = off_grid.row(0, named=True)
        raise ValueError(
            f"{name}: {off_grid.height} свечей с ценами вне шага {tick_size}, "
            f"первая {first['timestamp']}"
        )


def clearing_schedule(clearing_rates: pl.DataFrame) -> pl.DataFrame:
    """Курсы с моментом, до которого клиринг рассчитывает цены.

    С `SINGLE_SESSION_START` дневного клиринга нет: остаётся только курс mc.
    """
    return (
        clearing_rates.filter(
            (pl.col("clearing") == "mc") | (pl.col("date") < SINGLE_SESSION_START)
        )
        .with_columns(
            pl.when(pl.col("clearing") == "tc")
            .then(pl.col("date").dt.combine(CLEARING_TIME["tc"]))
            .otherwise(pl.col("date").dt.combine(CLEARING_TIME["mc"]))
            .alias("clearing_time")
        )
        .select("clearing_time", "fixing_time", "rate")
        .sort("clearing_time")
    )


def settled_bars(
    bars: pl.DataFrame, clearing_rates: pl.DataFrame, name: str = ""
) -> pl.DataFrame:
    """Свечи, которые биржа уже рассчитала: конец не позже последнего клиринга.

    Вечернюю сессию и сессии выходных рассчитывает клиринг следующего
    торгового дня. Пока его курс не зафиксирован или не скачан, рублей за
    пункт у этих свечей нет: в ряд они не входят, об этом выдаётся
    предупреждение.
    Пример: данные обновлены в субботу, ряд кончается пятничной свечой 23:45,
    последний курс — пятница 19:00. 20 свечей вечерней сессии отброшены,
    ряд кончается свечой 18:45. В понедельник после 19:00 (до 23.03.2026 —
    после 14:00) и обновления справочников они вернутся.

    Хвост длиннее `MAX_UNSETTLED_TAIL` — это уже не «биржа ещё не рассчитала»,
    а устаревший справочник: ошибка.
    """
    last_clearing = clearing_schedule(clearing_rates)["clearing_time"].max()
    bar_end = pl.col("timestamp") + BAR_LENGTH
    settled = bars.filter(bar_end <= last_clearing)
    dropped = bars.height - settled.height
    if dropped == 0:
        return bars
    tail = bars.filter(bar_end > last_clearing)["timestamp"]
    if tail.max() + BAR_LENGTH - last_clearing > MAX_UNSETTLED_TAIL:
        raise ValueError(
            f"{name}: нет курса клиринга для {dropped} свечей "
            f"({tail.min()} — {tail.max()}), последний клиринг {last_clearing}. "
            "Обновите справочники: download_reference.py --only clearing_fx"
        )
    warnings.warn(
        f"{name}: у {dropped} свечей с {tail.min():%d.%m.%Y %H:%M} ещё нет курса "
        f"клиринга (последний в справочнике — {last_clearing:%d.%m.%Y %H:%M}), "
        "в ряд не вошли. Появятся после следующего клиринга биржи и "
        "download_reference.py --only clearing_fx",
        stacklevel=2,
    )
    return settled


def add_point_value(
    bars: pl.DataFrame,
    market: Market,
    clearing_rates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Рубли за пункт по каждой свече.

    point_value — по курсу клиринга, который рассчитывает свечу: так биржа
    начисляет вариационную маржу. Для прибыли сделки.
    point_value_known — по последнему курсу, известному на закрытии свечи.
    Для размера позиции в момент сигнала: будущий курс ещё не известен.
    """
    if market.quote_currency == "RUB":
        return bars.with_columns(
            pl.lit(market.quote_point_value).alias("point_value"),
            pl.lit(market.quote_point_value).alias("point_value_known"),
        )
    if clearing_rates is None:
        raise ValueError(f"{market.name}: нужен курс клиринга USD/RUB")

    schedule = clearing_schedule(clearing_rates)
    with_end = bars.sort("timestamp").with_columns(
        (pl.col("timestamp") + BAR_LENGTH).alias("_bar_end")
    )
    settled = with_end.join_asof(
        schedule.select(
            pl.col("clearing_time").alias("_bar_end"),
            pl.col("clearing_time"),
            pl.col("rate").alias("_settle_rate"),
        ),
        on="_bar_end",
        strategy="forward",
    ).join_asof(
        schedule.sort("fixing_time").select(
            pl.col("fixing_time").alias("_bar_end"),
            pl.col("rate").alias("_known_rate"),
        ),
        on="_bar_end",
        strategy="backward",
    )
    missing = settled.filter(
        pl.col("_settle_rate").is_null() | pl.col("_known_rate").is_null()
    )
    if not missing.is_empty():
        raise ValueError(
            f"{market.name}: нет курса клиринга для {missing.height} свечей "
            f"({missing['timestamp'].min()} — {missing['timestamp'].max()}). "
            "Обновите справочники: download_reference.py --only clearing_fx"
        )
    return settled.with_columns(
        (pl.col("_settle_rate") * market.quote_point_value).alias("point_value"),
        (pl.col("_known_rate") * market.quote_point_value).alias("point_value_known"),
    ).drop("_bar_end", "clearing_time", "_settle_rate", "_known_rate")


def add_margin_rate(
    bars: pl.DataFrame,
    market: Market,
    margin_rates: pl.DataFrame,
    margin_factor: float,
) -> pl.DataFrame:
    """Доля ГО от стоимости контракта по дате свечи: mr1 на дату × коэффициент."""
    rates = (
        margin_rates.filter(pl.col("asset_code") == market.asset_code)
        .select(pl.col("date").alias("_rate_date"), "mr1")
        .sort("_rate_date")
    )
    joined = (
        bars.sort("timestamp")
        .with_columns(pl.col("timestamp").dt.date().alias("_rate_date"))
        .join_asof(rates, on="_rate_date", strategy="backward")
    )
    missing = joined.filter(pl.col("mr1").is_null())
    if not missing.is_empty():
        raise ValueError(
            f"{market.name}: нет ставки ГО для {missing.height} свечей "
            f"(с {missing['timestamp'].min()}). "
            "Обновите справочники: download_reference.py --only margins"
        )
    return joined.with_columns(
        (pl.col("mr1") * margin_factor).alias("margin_rate")
    ).drop("_rate_date", "mr1")


def add_usual_volume(
    bars: pl.DataFrame,
    days: int = USUAL_VOLUME_DAYS,
    min_days: int = USUAL_VOLUME_MIN_DAYS,
) -> pl.DataFrame:
    """Обычный объём свечи: медиана объёма в то же время суток за прошлые дни.

    Будни и выходные считаются отдельно. Сам день свечи в медиану не входит,
    поэтому значение известно заранее: «последние 20 дней в 08:45 торговали
    1–3 контракта». Меньше `min_days` прошлых дней — значение пустое.
    Дни без сделок в это время свечи не имеют и в медиану не попадают:
    у самых тонких свечей обычный объём немного завышен.
    """
    slot = [
        pl.col("timestamp").dt.time(),
        pl.col("timestamp").dt.weekday() >= SATURDAY,
    ]
    return bars.sort("timestamp").with_columns(
        pl.col("volume")
        .cast(pl.Float64)
        .shift(1)
        .rolling_median(window_size=days, min_samples=min_days)
        .over(slot)
        .alias("usual_volume")
    )


def add_bar_sigma(
    bars: pl.DataFrame,
    window: str = BAR_SIGMA_WINDOW,
    min_bars: int = BAR_SIGMA_MIN_BARS,
) -> pl.DataFrame:
    """Обычное движение 15-минутной свечи в долях цены: σ доходности NAV.

    Окно — прошлые `window` календарных дней, сама свеча в него не входит:
    значение известно до её начала. Доходность — по NAV, поэтому переносы
    контрактов не дают ложных скачков.
    """
    ordered = bars.sort("timestamp")
    returns = ordered.select(
        "timestamp", (pl.col("nav") / pl.col("nav").shift(1)).log().alias("_r")
    )
    sigma = returns.rolling("timestamp", period=window, closed="left").agg(
        pl.when(pl.col("_r").drop_nulls().len() >= min_bars)
        .then(pl.col("_r").std())
        .otherwise(None)
        .alias("bar_sigma")
    )
    return ordered.with_columns(sigma["bar_sigma"])


def add_session_marks(bars: pl.DataFrame) -> pl.DataFrame:
    """Отметки расписания: последняя свеча дня и последняя перед перерывом.

    is_day_end — следующая свеча ряда уже другого календарного дня.
    is_before_break — последняя свеча дня, после которой не обычная ночь.
    Обычная ночь — с буднего дня на следующий календарный день, тоже будний
    (пн→вт … чт→пт): 7–15 часов без торгов. Всё остальное — перерыв: пятница,
    канун праздника (30.12.2025 → 05.01.2026 — 129 часов), а также сессии
    выходных: позицию, открытую в субботу, отметка закроет в субботу же.

    Отметки — расписание биржи: оно известно заранее, поэтому это не
    подглядывание. Исключение — день, когда вечерняя сессия сорвалась: ряд
    «знает», что свеча 18:45 последняя, а трейдер в тот день не знал.
    У последней свечи ряда отметок нет: что будет после неё, неизвестно.
    """
    day = pl.col("timestamp").dt.date()
    next_day = day.shift(-1)
    is_day_end = (next_day != day).fill_null(False)
    ordinary_night = (
        (day.dt.weekday() < SATURDAY)
        & (next_day.dt.weekday() < SATURDAY)
        & ((next_day - day).dt.total_days() == 1)
    )
    return bars.sort("timestamp").with_columns(
        is_day_end.alias("is_day_end"),
        (is_day_end & ~ordinary_night).alias("is_before_break"),
    )


def load_market(name: str) -> tuple[MarketParams, pl.DataFrame]:
    """Паспорт рынка и его 15-минутный ряд с point_value, margin_rate,
    usual_volume, bar_sigma и отметками расписания.

    У долларовых рынков ряд кончается последней свечой, которую биржа уже
    рассчитала (`settled_bars`).
    """
    if name not in PASSPORTS:
        raise ValueError(f"Нет паспорта рынка {name}. Есть: {', '.join(PASSPORTS)}")
    market = PASSPORTS[name]

    clearing_rates = None
    if market.quote_currency == "USD":
        clearing_rates = pl.read_parquet(
            REFERENCE_DIR / "rates_moex_clearing.parquet"
        ).filter(pl.col("pair") == "USD/RUB")
    snapshot = pl.read_parquet(REFERENCE_DIR / "margin_snapshot.parquet")
    params = snapshot_params(market, snapshot, clearing_rates)

    bars = pl.read_parquet(CONTINUOUS_DIR / f"{name}_15m.parquet")
    check_tick(bars, params.tick_size, name)
    if clearing_rates is not None:
        bars = settled_bars(bars, clearing_rates, name)
    bars = add_point_value(bars, market, clearing_rates)
    margin_rates = pl.read_parquet(REFERENCE_DIR / "margin_rates.parquet")
    bars = add_margin_rate(bars, market, margin_rates, params.margin_factor)
    return params, add_session_marks(add_bar_sigma(add_usual_volume(bars)))
