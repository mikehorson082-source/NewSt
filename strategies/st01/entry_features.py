"""Признаки и метки сигналов пробоя для фильтра входа (plan.md, шаг 6, 6.1–6.2).

Строка набора — сигнал замороженных правил 5.3 (`frozen_breakout_signals`),
в том числе тот, что в портфеле получил бы `busy`: если модель пропустит
сигнал в 11:00, сделкой может стать сигнал в 11:15, и его тоже надо уметь
оценить.

Метка — результат сделки по сигналу в R после издержек, будто сделка открыта
в одиночку: один контракт, тот же симулятор и минутки, что у портфеля. R тот
же, что `PortfolioTrade.r_multiple`: издержки на контракт от размера позиции
не зависят. Пример: стоп в 0.75 дня, сделка дошла до цели — около +2.9R.

Признаки — только то, что известно на закрытии сигнальной свечи, и только по
своему рынку. Расстояния — в дневных движениях (σ свечи × √60 × цена), как
стоп правил: так рынки и годы сравнимы. «По направлению сделки» — со знаком
стороны: плюс значит «в сторону сделки». Пример: шорт, цена за прошлый день
упала на 1.2 дневных движения — `move_1d` = +1.2.

| Признак             | Что это                                                          |
| ------------------- | ---------------------------------------------------------------- |
| `side`              | +1 лонг, −1 шорт                                                 |
| `breakout_strength` | насколько закрытие вышло за канал                                |
| `channel_width`     | ширина канала 240 свечей («сжатая пружина» — узкий канал)        |
| `vol_rank`          | место σ свечи среди значений рынка за прошлый год (0…1)          |
| `trend`             | закрытие против средней 200 дневных закрытий, по направлению     |
| `move_1d`           | движение за прошлые 60 свечей (около дня), по направлению        |
| `move_5d`           | движение за прошлые 300 свечей (около 5 дней), по направлению    |
| `hours_to_break`    | часов от сигнала до выхода перед выходными или праздником        |
| `entry_hour`        | час свечи входа (категория)                                      |
| `weekday`           | день недели сигнала, 1 — понедельник (категория)                 |
| `volume_ratio`      | ln(объём сигнальной свечи ÷ обычный объём этого времени суток)   |
| `repeat_signals`    | сколько из прошлых 240 свечей закрылись за каналом в ту же сторону |
| `cost_share`        | издержки входа и выхода — доля 1R                                |

Пропуски: `vol_rank` — пока нет года истории σ (до 05.2023), `trend` — пока
нет 200 дней, `hours_to_break` — в конце данных, когда перерыва ещё не видно.
"""

from __future__ import annotations

import math
from datetime import datetime

import polars as pl
from rules import (
    BARS_PER_DAY,
    FROZEN_LOOKBACK,
    frozen_breakout_signals,
    long_trend,
    volatility_rank,
)

from backtest.portfolio import MarketData
from backtest.position import Signal, simulate_position
from backtest.sizing import SizingRules, size_position

TREND_DAYS = 200  # «200-дневная средняя», как фильтр е на 5.2
DAY_BARS = BARS_PER_DAY
WEEK_BARS = 5 * BARS_PER_DAY
BAR_MINUTES = 15

FEATURES = (
    "side",
    "breakout_strength",
    "channel_width",
    "vol_rank",
    "trend",
    "move_1d",
    "move_5d",
    "hours_to_break",
    "entry_hour",
    "weekday",
    "volume_ratio",
    "repeat_signals",
    "cost_share",
)
CATEGORICAL = ("entry_hour", "weekday")
SIGNAL_COLUMNS = ("timestamp", "side", "stop", "target", "hold", "breakeven_r", "trail")
# Признаки, кроме тех, что уже есть среди столбцов сигнала (side).
EXTRA_FEATURES = tuple(f for f in FEATURES if f not in SIGNAL_COLUMNS)

# Метке нужен только риск контракта, он от капитала не зависит. Лимит
# ликвидности выключен: метка — результат сделки, а не её размер.
LABEL_CAPITAL = 1e12
LABEL_SIZING = SizingRules(max_volume_share=None)


def bar_features(bars: pl.DataFrame, lookback: int = FROZEN_LOOKBACK) -> pl.DataFrame:
    """Заготовки признаков на каждой свече, пока без стороны сделки.

    Канал — как в `breakout_signals`: прошлые `lookback` свечей без
    сигнальной, по непрерывным ценам. Конец перерыва — закрытие ближайшей
    свечи с `is_before_break` (это расписание биржи, известное заранее).
    """
    adjustment = pl.col("price_adjustment")
    close_adj = pl.col("close") + adjustment
    upper = (pl.col("high") + adjustment).shift(1).rolling_max(lookback)
    lower = (pl.col("low") + adjustment).shift(1).rolling_min(lookback)
    bar = pl.duration(minutes=BAR_MINUTES)
    frame = (
        volatility_rank(bars)
        .with_columns(
            close_adj.alias("_close_adj"),
            upper.alias("_upper"),
            lower.alias("_lower"),
            (pl.col("bar_sigma") * math.sqrt(BARS_PER_DAY) * pl.col("close")).alias(
                "_day_move"
            ),
        )
        .with_columns(
            (pl.col("_close_adj") > pl.col("_upper")).cast(pl.Int64).alias("_above"),
            (pl.col("_close_adj") < pl.col("_lower")).cast(pl.Int64).alias("_below"),
        )
        .with_columns(
            pl.col("_above")
            .shift(1)
            .rolling_sum(lookback, min_samples=1)
            .alias("_above_count"),
            pl.col("_below")
            .shift(1)
            .rolling_sum(lookback, min_samples=1)
            .alias("_below_count"),
            pl.col("_close_adj").shift(DAY_BARS).alias("_close_1d"),
            pl.col("_close_adj").shift(WEEK_BARS).alias("_close_5d"),
            pl.when(pl.col("is_before_break"))
            .then(pl.col("timestamp") + bar)
            .fill_null(strategy="backward")
            .alias("_break_end"),
            pl.col("timestamp").shift(-1).alias("_entry_time"),
            # Логарифм: свеча бывает в 100 раз больше обычной, и без него
            # такие выбросы задавили бы линейную модель. 0 — обычный объём.
            pl.when((pl.col("usual_volume") > 0) & (pl.col("volume") > 0))
            .then((pl.col("volume") / pl.col("usual_volume")).log())
            .alias("volume_ratio"),
        )
    )
    trend = long_trend(frame, TREND_DAYS).rename({"date": "_trend_date"})
    return (
        frame.with_columns(pl.col("timestamp").dt.date().alias("_trend_date"))
        .join(trend, on="_trend_date", how="left")
        .sort("timestamp")
    )


def signal_features(signals: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """Признаки сигналов: столбцы сигнала + `FEATURES` без `cost_share`."""
    side = pl.col("side")
    move = pl.col("_day_move")
    close_adj = pl.col("_close_adj")
    boundary = pl.when(side > 0).then(pl.col("_upper")).otherwise(pl.col("_lower"))
    bar = pl.duration(minutes=BAR_MINUTES)
    return (
        signals.select(*SIGNAL_COLUMNS)
        .join(bar_features(bars), on="timestamp", how="left")
        .with_columns(
            (side * (close_adj - boundary) / move).alias("breakout_strength"),
            ((pl.col("_upper") - pl.col("_lower")) / move).alias("channel_width"),
            (side * (close_adj - pl.col("trend_mean")) / move).alias("trend"),
            (side * (close_adj - pl.col("_close_1d")) / move).alias("move_1d"),
            (side * (close_adj - pl.col("_close_5d")) / move).alias("move_5d"),
            (
                (pl.col("_break_end") - (pl.col("timestamp") + bar)).dt.total_minutes()
                / 60
            ).alias("hours_to_break"),
            pl.col("_entry_time").dt.hour().cast(pl.Int64).alias("entry_hour"),
            pl.col("timestamp").dt.weekday().cast(pl.Int64).alias("weekday"),
            pl.when(side > 0)
            .then(pl.col("_above_count"))
            .otherwise(pl.col("_below_count"))
            .cast(pl.Float64)
            .alias("repeat_signals"),
        )
        .select(*SIGNAL_COLUMNS, *(f for f in EXTRA_FEATURES if f != "cost_share"))
    )


def label_signals(data: MarketData, signals: pl.DataFrame) -> pl.DataFrame:
    """Сделка по каждому сигналу в одиночку: R после издержек и время выхода.

    Сигнал на последней свече ряда пропускается: входить не на чем.
    `exit_reason` = "end" — сделку закрыл конец данных, в обучение она не идёт.
    """
    bars = data.bars
    index = {stamp: i for i, stamp in enumerate(bars["timestamp"])}
    rows: list[dict[str, object]] = []
    for row in signals.iter_rows(named=True):
        bar = index[row["timestamp"]]
        if bar == bars.height - 1:
            continue
        signal = Signal(
            side=row["side"],
            stop=row["stop"],
            target=row["target"],
            hold=row["hold"],
            breakeven_r=row.get("breakeven_r"),
            trail=row.get("trail"),
        )
        size = size_position(data.spec, bars, bar, signal, LABEL_CAPITAL, LABEL_SIZING)
        trade = simulate_position(data.spec, bars, bar, signal, minutes=data.minutes)
        rows.append(
            {
                "timestamp": row["timestamp"],
                "r": trade.net_pnl / size.risk_per_contract,
                "exit_time": trade.exit_time,
                "exit_reason": trade.exit_reason,
                "cost_share": size.cost_share,
            }
        )
    schema = {
        "timestamp": pl.Datetime("us"),
        "r": pl.Float64,
        "exit_time": pl.Datetime("us"),
        "exit_reason": pl.String,
        "cost_share": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def market_dataset(name: str, data: MarketData) -> pl.DataFrame:
    """Сигналы одного рынка: столбцы сигнала, признаки, метка."""
    signals = frozen_breakout_signals(name, data.bars)
    labels = label_signals(data, signals)
    features = signal_features(signals, data.bars)
    timestamp_type = data.bars.schema["timestamp"]
    labels = labels.with_columns(
        pl.col("timestamp").cast(timestamp_type),
        pl.col("exit_time").cast(timestamp_type),
    )
    return (
        features.join(labels, on="timestamp", how="inner")
        .with_columns(pl.lit(name).alias("market"))
        .select(
            "market", *SIGNAL_COLUMNS, *EXTRA_FEATURES, "r", "exit_time", "exit_reason"
        )
        .sort("timestamp")
    )


def build_dataset(markets: dict[str, MarketData]) -> pl.DataFrame:
    """Набор всех рынков одной таблицей, по времени."""
    return pl.concat(
        [market_dataset(name, data) for name, data in markets.items()]
    ).sort("timestamp", "market")


def training_rows(dataset: pl.DataFrame, start: datetime) -> pl.DataFrame:
    """Сигналы, чья сделка закрылась до `start`: зазор walk-forward (6.3).

    Сигнал 28.12.2023 входит в обучение для 2024 года, только если его сделка
    закрылась до 01.01.2024: иначе метка «заглядывает» в проверочный год.
    Сделки, закрытые концом данных, не входят никогда.
    """
    return dataset.filter(
        (pl.col("exit_time") < start) & (pl.col("exit_reason") != "end")
    )
