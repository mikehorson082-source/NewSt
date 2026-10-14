"""Две простые стратегии st01 без ML: пробой канала и возврат к средней.

Функция получает 15-минутный ряд одного рынка из `backtest.markets.load_market`
и возвращает сигналы для `backtest.portfolio.run_portfolio`: timestamp (свеча,
по закрытию которой посчитан сигнал), side, stop, target, hold. Столбец market
добавляет тот, кто собирает сигналы рынков вместе.

Параметры заданы заранее. Варианты пробоя для подбора (plan.md, 5.2) —
необязательные аргументы `breakout_signals`; по умолчанию — правило шага 4.

Пробой канала (тренд). Канал — максимум и минимум цены за прошлые 120 свечей
(около двух торговых дней). Свеча закрылась выше канала — лонг, ниже — шорт.
Стоп — одно обычное дневное движение от закрытия, цель — три таких движения.
Ночи позиция держит, перед выходными и праздниками закрывается.
Пример: Brent два дня ходил между 69 и 71, свеча закрылась на 71.20. Обычное
движение свечи 0.22%, за день — 0.22% × √60 ≈ 1.7%, это $1.21. Лонг:
стоп 71.20 − 1.21 = 69.99, цель 71.20 + 3 × 1.21 = 74.83.

Возврат к средней. Средняя и обычный разброс (стандартное отклонение) цены за
последние 60 свечей (около торгового дня). Цена ушла от средней дальше двух
разбросов — сделка против движения. Цель — средняя, стоп — на таком же
расстоянии с другой стороны. Сигнал один на каждый уход: пока цена не
вернулась внутрь, повторных нет. Позиция живёт только внутри дня.
Пример: средняя Brent за день 70.00, разброс 0.40, свеча закрылась на 69.10 —
на 0.90 ниже средней, это больше 2 × 0.40. Лонг: цель 70.00, стоп 68.20.

Общее для обеих:
- сигнал только в будни и не на последней свече дня (вход был бы уже после
  ночи, по другой цене);
- канал, средняя и разброс считаются по непрерывным ценам (цена +
  `price_adjustment`): перенос контракта не рисует ложный пробой. Стоп и цель
  отдаются в реальных ценах сигнальной свечи, как ждёт симулятор;
- в расчёт входит только прошлое: канал — свечи до сигнальной, σ свечи
  (`bar_sigma`) — дни до неё, средняя — по закрытие сигнальной включительно.

Варианты пробоя (plan.md, 5.2):
- выход: без цели (`reward=None`), безубыток (`breakeven_r`), скользящий стоп
  на `trail_days` дневных движений от лучшего закрытия;
- вход: время свечи входа в окне `entry_hours`; без входов в день, который
  кончается перерывом (`skip_break_days`); только по направлению долгого тренда
  (`trend_days`): закрытие выше средней дневных закрытий за прошлые
  `trend_days` торговых дней — только лонги, ниже — только шорты.
"""

from __future__ import annotations

import math
from datetime import time

import polars as pl

from backtest.position import HOLD_DAY, HOLD_OVERNIGHT, LONG, SHORT

# σ 15-минутной свечи → σ торгового дня: в дне около 60 свечей (08:45–23:45).
BARS_PER_DAY = 60
SATURDAY = 6

BREAKOUT_LOOKBACK = 120  # свечей в канале, около двух торговых дней
BREAKOUT_STOP_DAYS = 1.0  # стоп в обычных дневных движениях
BREAKOUT_REWARD = 3.0  # цель в расстояниях до стопа

REVERSION_LOOKBACK = 60  # свечей в средней, около торгового дня
REVERSION_ENTRY_SD = 2.0  # уход от средней в разбросах
# Стоп не ближе одного обычного движения свечи: на стоячем рынке разброс почти
# нулевой, и «уход на два разброса» — это один шаг цены.
REVERSION_MIN_STOP_SIGMAS = 1.0

SIGNAL_COLUMNS = ("timestamp", "side", "stop", "target", "hold")
# Подвижный стоп пробоя (`Signal.breakeven_r`, `Signal.trail`).
MOVING_STOP_COLUMNS = ("breakeven_r", "trail")
REQUIRED_COLUMNS = {
    "timestamp",
    "high",
    "low",
    "close",
    "price_adjustment",
    "bar_sigma",
    "is_day_end",
}


def _ordered(bars: pl.DataFrame) -> pl.DataFrame:
    missing = REQUIRED_COLUMNS - set(bars.columns)
    if missing:
        raise ValueError(f"В барах нет столбцов: {', '.join(sorted(missing))}")
    return bars.sort("timestamp")


def _signals(
    frame: pl.DataFrame,
    distance: pl.Expr,
    reward: pl.Expr | None,
    hold: str,
    extra: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Сигналы из столбца side: стоп на `distance` от закрытия, цель на `reward`.

    `reward=None` — без цели. Остаются только свечи, где сигнал можно
    исполнить: будни, не последняя свеча дня, уровни посчитаны и больше нуля.
    """
    side = pl.col("side")
    close = pl.col("close")
    target = pl.lit(None, dtype=pl.Float64) if reward is None else close + side * reward
    valid_target = pl.lit(True) if reward is None else (reward > 0) & (target > 0)
    return (
        frame.with_columns(
            (close - side * distance).alias("stop"),
            target.alias("target"),
            pl.lit(hold).alias("hold"),
        )
        .filter(
            side.is_not_null()
            & (pl.col("timestamp").dt.weekday() < SATURDAY)
            & ~pl.col("is_day_end")
            & (distance > 0)
            & (pl.col("stop") > 0)
            & valid_target
        )
        .select(*SIGNAL_COLUMNS, *extra)
    )


def _in_hours(moment: pl.Expr, hours: tuple[time | None, time | None]) -> pl.Expr:
    """Время свечи в окне [начало, конец); None — без границы."""
    start, end = hours
    inside = pl.lit(True)
    if start is not None:
        inside = inside & (moment.dt.time() >= start)
    if end is not None:
        inside = inside & (moment.dt.time() < end)
    return inside


def break_days(bars: pl.DataFrame) -> pl.Series:
    """Дни, которые кончаются перерывом: пятница, канун праздника.

    Это расписание, известное заранее: отметка `is_before_break` у последней
    свечи дня (`markets.add_session_marks`).
    """
    return (
        bars.filter(pl.col("is_before_break"))
        .select(pl.col("timestamp").dt.date().alias("date"))
        .unique()
        .sort("date")["date"]
    )


def long_trend(bars: pl.DataFrame, days: int) -> pl.DataFrame:
    """Средняя дневных закрытий за `days` прошлых торговых дней, по дате.

    Дневное закрытие — закрытие последней будничной свечи дня в непрерывных
    ценах. В среднюю входят только завершённые дни до этой даты: закрытие
    сегодняшнего дня в момент сигнала ещё неизвестно. Меньше `days` дней
    истории — средней нет.
    """
    return (
        bars.filter(pl.col("timestamp").dt.weekday() < SATURDAY)
        .sort("timestamp")
        .group_by(pl.col("timestamp").dt.date().alias("date"), maintain_order=True)
        .agg((pl.col("close") + pl.col("price_adjustment")).last().alias("_day_close"))
        .with_columns(
            pl.col("_day_close").rolling_mean(days).shift(1).alias("trend_mean")
        )
        .select("date", "trend_mean")
    )


def breakout_signals(
    bars: pl.DataFrame,
    lookback: int = BREAKOUT_LOOKBACK,
    stop_days: float = BREAKOUT_STOP_DAYS,
    reward: float | None = BREAKOUT_REWARD,
    breakeven_r: float | None = None,
    trail_days: float | None = None,
    entry_hours: tuple[time | None, time | None] | None = None,
    skip_break_days: bool = False,
    trend_days: int | None = None,
) -> pl.DataFrame:
    """Пробой канала: закрытие выше максимума или ниже минимума прошлых свечей.

    Кроме столбцов сигнала отдаёт `breakeven_r` и `trail` (пусто, если не
    заданы). `trail` — в пунктах цены: `trail_days` дневных движений на
    свече сигнала.
    """
    adjustment = pl.col("price_adjustment")
    close_adj = pl.col("close") + adjustment
    # shift(1): сама сигнальная свеча в канал не входит.
    upper = (pl.col("high") + adjustment).shift(1).rolling_max(lookback)
    lower = (pl.col("low") + adjustment).shift(1).rolling_min(lookback)
    day_move = pl.col("bar_sigma") * math.sqrt(BARS_PER_DAY) * pl.col("close")

    frame = _ordered(bars).with_columns(
        pl.when(close_adj > upper)
        .then(LONG)
        .when(close_adj < lower)
        .then(SHORT)
        .alias("side"),
        (stop_days * day_move).alias("_distance"),
        pl.lit(breakeven_r, dtype=pl.Float64).alias("breakeven_r"),
        (
            pl.lit(None, dtype=pl.Float64)
            if trail_days is None
            else trail_days * day_move
        ).alias("trail"),
        # Время входа — открытие следующей свечи: это расписание биржи.
        pl.col("timestamp").shift(-1).alias("_entry_time"),
    )
    if entry_hours is not None:
        frame = frame.filter(_in_hours(pl.col("_entry_time"), entry_hours))
    if skip_break_days:
        if "is_before_break" not in frame.columns:
            raise ValueError(
                "Для skip_break_days в барах нужен столбец is_before_break"
            )
        frame = frame.filter(
            ~pl.col("timestamp").dt.date().is_in(break_days(frame).implode())
        )
    if trend_days is not None:
        frame = (
            frame.join(
                long_trend(frame, trend_days),
                left_on=pl.col("timestamp").dt.date(),
                right_on="date",
                how="left",
            )
            .filter(
                ((pl.col("side") == LONG) & (close_adj > pl.col("trend_mean")))
                | ((pl.col("side") == SHORT) & (close_adj < pl.col("trend_mean")))
            )
            .sort("timestamp")
        )
    distance = pl.col("_distance")
    reward_distance = None if reward is None else reward * distance
    return _signals(
        frame, distance, reward_distance, HOLD_OVERNIGHT, MOVING_STOP_COLUMNS
    )


def reversion_signals(
    bars: pl.DataFrame,
    lookback: int = REVERSION_LOOKBACK,
    entry_sd: float = REVERSION_ENTRY_SD,
    min_stop_sigmas: float = REVERSION_MIN_STOP_SIGMAS,
) -> pl.DataFrame:
    """Возврат к средней: цена ушла от средней дальше `entry_sd` разбросов."""
    close_adj = pl.col("close") + pl.col("price_adjustment")
    frame = (
        _ordered(bars)
        .with_columns(
            close_adj.rolling_mean(lookback).alias("_mean"),
            close_adj.rolling_std(lookback).alias("_sd"),
        )
        .with_columns(((close_adj - pl.col("_mean")) / pl.col("_sd")).alias("_z"))
    )
    z = pl.col("_z")
    before = z.shift(1)
    # Расстояние до средней — оно же расстояние до цели и до стопа.
    distance = (pl.col("_mean") - close_adj).abs()
    frame = frame.with_columns(
        # Сигнал — только на свече, где цена вышла за границу: на прошлой
        # свече она была внутри.
        pl.when(z.is_finite() & before.is_finite() & (z < -entry_sd))
        .then(pl.when(before >= -entry_sd).then(LONG))
        .when(z.is_finite() & before.is_finite() & (z > entry_sd))
        .then(pl.when(before <= entry_sd).then(SHORT))
        .alias("side"),
        distance.alias("_distance"),
    ).filter(
        pl.col("_distance") >= min_stop_sigmas * pl.col("bar_sigma") * pl.col("close")
    )
    distance = pl.col("_distance")
    return _signals(frame, distance, distance, HOLD_DAY)
