"""Две простые стратегии st01 без ML: пробой канала и возврат к средней.

Функция получает 15-минутный ряд одного рынка из `backtest.markets.load_market`
и возвращает сигналы для `backtest.portfolio.run_portfolio`: timestamp (свеча,
по закрытию которой посчитан сигнал), side, stop, target, hold. Столбец market
добавляет тот, кто собирает сигналы рынков вместе.

Параметры заданы заранее и по результатам не подбираются: подбор — это шаг 5
(walk-forward), и только на прошлом.

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
"""

from __future__ import annotations

import math

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
    frame: pl.DataFrame, distance: pl.Expr, reward: pl.Expr, hold: str
) -> pl.DataFrame:
    """Сигналы из столбца side: стоп на `distance` от закрытия, цель на `reward`.

    Остаются только свечи, где сигнал можно исполнить: будни, не последняя
    свеча дня, уровни посчитаны и больше нуля.
    """
    side = pl.col("side")
    close = pl.col("close")
    return (
        frame.with_columns(
            (close - side * distance).alias("stop"),
            (close + side * reward).alias("target"),
            pl.lit(hold).alias("hold"),
        )
        .filter(
            side.is_not_null()
            & (pl.col("timestamp").dt.weekday() < SATURDAY)
            & ~pl.col("is_day_end")
            & (distance > 0)
            & (reward > 0)
            & (pl.col("stop") > 0)
            & (pl.col("target") > 0)
        )
        .select(SIGNAL_COLUMNS)
    )


def breakout_signals(
    bars: pl.DataFrame,
    lookback: int = BREAKOUT_LOOKBACK,
    stop_days: float = BREAKOUT_STOP_DAYS,
    reward: float = BREAKOUT_REWARD,
) -> pl.DataFrame:
    """Пробой канала: закрытие выше максимума или ниже минимума прошлых свечей."""
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
    )
    distance = pl.col("_distance")
    return _signals(frame, distance, reward * distance, HOLD_OVERNIGHT)


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
