"""ADX, VHF и сжатие полос Боллинджера: есть ли у пробоя «плохая треть» сделок.

Вопрос пользователя (21.10.2026): добавить ли в стратегию или модель ADX, VHF
и Bollinger Bands Squeeze. Правила 5.3 заморожены, поэтому это проверка без
модели, как диагностика «д» на 5.2.

Урок шага 6 (journal.md, раздел 17): фильтр помогает, только если находит
группу сигналов, которая в среднем не зарабатывает. Даже худшая треть по
прогнозу модели была в плюсе (+0.14R), и пропуск её срезал прибыль.

Индикаторы — на 15-минутных свечах, по непрерывным ценам (перенос контракта
не рисует ложного движения), на закрытии сигнальной свечи:
- ADX (Уайлдер) — сила направленного движения, без знака. Около 10 — цена
  мечется, 40 и выше — сильный тренд в любую сторону;
- VHF — (максимум − минимум закрытий) ÷ сумма |изменений закрытия| за то же
  окно. 1 — цена шла в одну сторону, около 0 — «пила»;
- сжатие полос Боллинджера — ширина полос (20 свечей, ±2σ) делится на цену;
  её место среди значений за прошлые полгода: 0 — полосы уже, чем когда-либо
  за полгода (определение сжатия у самого Боллинджера).
Классические настройки (ADX 14, VHF 28) — центральные, по ним решение.
Длинные (ADX 60 — около дня, VHF 240 — окно канала) — для сведения.

Сделки — портфель paper mode (cnyrub, si, euro, gold, правила 5.5–5.6), вся
история, 1 шаг проскальзывания. Трети — по границам каждого рынка: границы
знают всю историю, для правила это подглядывание, для проверки допустимо.

Ориентир (записан до запуска): индикатор — кандидат в блок шага 7, если вместе
по четырём рынкам хотя бы у одной его трети при классической настройке R после
издержек ≤ 0. Все трети в плюсе — вопрос закрыт: фильтр по нему срезал бы
прибыль. Треть ≤ 0 только у длинной настройки — записываем, но это уже поиск
после просмотра.

Ещё печатает ранговую корреляцию (Спирмен) индикаторов с признаками шага 6:
если индикатор почти повторяет признак, модель его уже видела.

Ничего не записывает. Запуск (около минуты):
  uv run python strategies/st01/check_trend_indicators.py
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from check_breakout import closed
from entry_features import signal_features
from rules import frozen_breakout_signals
from run_portfolio import RISK_FRACTIONS, portfolio

from backtest.portfolio import MarketData, load_markets

ADX_PERIOD = 14  # Уайлдер
VHF_PERIOD = 28  # классический VHF
LONG_ADX_PERIOD = 60  # около торгового дня
LONG_VHF_PERIOD = 240  # окно канала замороженных правил
BB_PERIOD = 20
BB_SIGMAS = 2.0
SQUEEZE_DAYS = 182  # полгода

CLASSIC = {
    "adx": f"ADX {ADX_PERIOD}",
    "vhf": f"VHF {VHF_PERIOD}",
    "squeeze": f"сжатие BB {BB_PERIOD}",
}
LONG = {
    "adx_long": f"ADX {LONG_ADX_PERIOD}",
    "vhf_long": f"VHF {LONG_VHF_PERIOD}",
}
INDICATORS = {**CLASSIC, **LONG}
THIRDS = ("низ", "середина", "верх")
STEP6_FEATURES = (
    "channel_width",
    "vol_rank",
    "breakout_strength",
    "move_1d",
    "move_5d",
    "repeat_signals",
)


# -------------------------------------------------------------- индикаторы


def _prices() -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Максимум, минимум и закрытие в непрерывных ценах."""
    adjustment = pl.col("price_adjustment")
    return (
        pl.col("high") + adjustment,
        pl.col("low") + adjustment,
        pl.col("close") + adjustment,
    )


def _wilder(value: pl.Expr, period: int) -> pl.Expr:
    """Сглаживание Уайлдера: экспоненциальное с весом 1/period."""
    return value.ewm_mean(alpha=1 / period, adjust=False, min_samples=period)


def adx(period: int) -> pl.Expr:
    """ADX Уайлдера по свечам.

    Движение вверх — насколько максимум свечи выше прошлого, вниз — насколько
    минимум ниже прошлого; засчитывается большее из двух. Их сглаженные доли
    в истинном диапазоне — +DI и −DI; DX = |+DI − −DI| ÷ (+DI + −DI) × 100;
    ADX — сглаженный DX. Пример: цена каждую свечу на шаг выше прошлой —
    движение только вверх, DX = 100, ADX растёт к 100.
    """
    high, low, close = _prices()
    previous = close.shift(1)
    true_range = pl.max_horizontal(
        high - low, (high - previous).abs(), (low - previous).abs()
    )
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_move = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus_move = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    atr = _wilder(true_range, period)
    plus_di = pl.when(atr > 0).then(100 * _wilder(plus_move, period) / atr)
    minus_di = pl.when(atr > 0).then(100 * _wilder(minus_move, period) / atr)
    total = plus_di + minus_di
    # Ни вверх, ни вниз за всю историю (стоячий ряд) — направления нет, DX 0.
    dx = (
        pl.when(total > 0)
        .then(100 * (plus_di - minus_di).abs() / total)
        .otherwise(pl.when(total.is_not_null()).then(0.0))
    )
    return _wilder(dx, period)


def vhf(period: int) -> pl.Expr:
    """VHF: размах закрытий ÷ путь закрытий за `period` изменений.

    Размах — по тем же `period` + 1 закрытиям, что и путь, поэтому VHF от 0
    до 1. Пример: 28 свечей подряд вверх на шаг — размах 28, путь 28, VHF 1;
    вверх-вниз на шаг — размах 1, путь 28, VHF 0.04.
    """
    _, _, close = _prices()
    path = close.diff().abs().rolling_sum(period)
    span = close.rolling_max(period + 1) - close.rolling_min(period + 1)
    return pl.when(path > 0).then(span / path)


def bollinger_width(period: int = BB_PERIOD, sigmas: float = BB_SIGMAS) -> pl.Expr:
    """Ширина полос Боллинджера (2 × sigmas × σ закрытий) в долях цены."""
    _, _, close = _prices()
    return 2 * sigmas * close.rolling_std(period, ddof=0) / pl.col("close")


def past_rank(timestamps: np.ndarray, values: np.ndarray, days: int) -> np.ndarray:
    """Место каждого значения среди значений за прошлые `days` дней.

    Доля значений на свечах [время − `days`, время), которые меньше значения
    свечи; сама свеча в сравнение не входит. 0 — меньше всего за период.
    Пока у ряда нет `days` дней истории — NaN. Пример: за полгода 6 000
    значений ширины полос, 60 из них уже нынешней — место 0.01, сжатие.
    """
    window = np.timedelta64(days, "D")
    ranks = np.full(len(values), np.nan)
    if not len(values):
        return ranks
    starts = np.searchsorted(timestamps, timestamps - window, side="left")
    for i in np.flatnonzero(timestamps - timestamps[0] >= window):
        if np.isnan(values[i]):
            continue
        past = values[starts[i] : i]
        past = past[~np.isnan(past)]
        if len(past):
            ranks[i] = np.count_nonzero(past < values[i]) / len(past)
    return ranks


def indicator_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """Индикаторы на каждой свече: `timestamp` и столбцы `INDICATORS`."""
    frame = bars.sort("timestamp").with_columns(
        adx(ADX_PERIOD).alias("adx"),
        vhf(VHF_PERIOD).alias("vhf"),
        bollinger_width().alias("_width"),
        adx(LONG_ADX_PERIOD).alias("adx_long"),
        vhf(LONG_VHF_PERIOD).alias("vhf_long"),
    )
    squeeze = past_rank(
        frame["timestamp"].to_numpy(),
        frame["_width"].fill_null(np.nan).to_numpy(),
        SQUEEZE_DAYS,
    )
    return frame.with_columns(pl.Series("squeeze", squeeze).fill_nan(None)).select(
        "timestamp", *INDICATORS
    )


# ------------------------------------------------------------------ сделки


def trades_with_indicators(markets: dict[str, MarketData]) -> pl.DataFrame:
    """Закрытые сделки портфеля с индикаторами и признаками шага 6 на сигнале."""
    _, report = portfolio(markets)
    trades = closed(report.trades)
    frames = []
    for name, data in markets.items():
        signals = frozen_breakout_signals(name, data.bars)
        frames.append(
            signal_features(signals, data.bars)
            .select("timestamp", *STEP6_FEATURES)
            .join(indicator_bars(data.bars), on="timestamp", how="left")
            .with_columns(pl.lit(name).alias("market"))
        )
    features = pl.concat(frames).select(
        "market",
        pl.col("timestamp").cast(trades.schema["signal_time"]).alias("signal_time"),
        *INDICATORS,
        *STEP6_FEATURES,
    )
    return trades.join(features, on=["market", "signal_time"], how="left").select(
        "market",
        "signal_time",
        "r_multiple",
        (pl.col("pnl_before_costs") / pl.col("risk")).alias("r_before"),
        *INDICATORS,
        *STEP6_FEATURES,
    )


def with_thirds(trades: pl.DataFrame, column: str) -> pl.DataFrame:
    """Сделки с известным `column` и третью по нему — по границам своего рынка.

    Пример: у cnyrub 300 сделок с ADX — 100 с самым низким ADX в «низ», у
    золота 150 — 50 в «низ», хотя ADX золота обычно выше.
    """
    known = trades.filter(pl.col(column).is_not_null())
    rank = pl.col(column).rank("ordinal").over("market") - 1
    third = (rank * len(THIRDS) // pl.len().over("market")).cast(pl.Int64)
    return known.with_columns(
        third.replace_strict(dict(enumerate(THIRDS))).alias("треть")
    )


def _mean_se(values: pl.Series) -> tuple[float, float]:
    return values.mean(), values.std() / math.sqrt(values.len())


def thirds_table(trades: pl.DataFrame, indicators: dict[str, str]) -> pl.DataFrame:
    """По каждому индикатору и трети: сделок, R после издержек ± погрешность, R до."""
    rows: list[dict[str, Any]] = []
    for column, label in indicators.items():
        grouped = with_thirds(trades, column)
        for third in THIRDS:
            part = grouped.filter(pl.col("треть") == third)
            mean, se = _mean_se(part["r_multiple"])
            rows.append(
                {
                    "индикатор": label,
                    "треть": third,
                    "значения": f"{part[column].min():.2f}…{part[column].max():.2f}",
                    "сделок": part.height,
                    "R после издержек": round(mean, 3),
                    "±": round(se, 3),
                    "R до издержек": round(part["r_before"].mean(), 3),
                }
            )
    return pl.DataFrame(rows)


def by_market_table(trades: pl.DataFrame, indicators: dict[str, str]) -> pl.DataFrame:
    """R после издержек по третям на каждом рынке (для сведения)."""
    frames = [
        with_thirds(trades, column)
        .group_by("market", "треть")
        .agg(pl.col("r_multiple").mean().round(3))
        .with_columns(pl.lit(label).alias("индикатор"))
        for column, label in indicators.items()
    ]
    order = {third: i for i, third in enumerate(THIRDS)}
    return (
        pl.concat(frames)
        .pivot(on="market", index=["индикатор", "треть"], values="r_multiple")
        .sort(
            pl.col("индикатор").replace_strict(
                {label: i for i, label in enumerate(indicators.values())}
            ),
            pl.col("треть").replace_strict(order),
        )
        .select("индикатор", "треть", *RISK_FRACTIONS)
    )


def similarity_table(
    trades: pl.DataFrame, indicators: dict[str, str] = INDICATORS
) -> pl.DataFrame:
    """Спирмен индикаторов с признаками шага 6 по сделкам."""
    rows = []
    for column, label in indicators.items():
        row: dict[str, Any] = {"индикатор": label}
        for feature in STEP6_FEATURES:
            pair = trades.select(column, feature).drop_nulls()
            row[feature] = round(
                pair.select(pl.corr(column, feature, method="spearman")).item(), 2
            )
        rows.append(row)
    return pl.DataFrame(rows)


def verdict(table: pl.DataFrame, indicators: dict[str, str]) -> list[str]:
    """Строки решения: по каждому индикатору — худшая треть и кандидат ли он."""
    lines = []
    for label in indicators.values():
        rows = table.filter(pl.col("индикатор") == label).sort("R после издержек")
        worst = rows.row(0, named=True)
        status = (
            "кандидат в блок шага 7"
            if worst["R после издержек"] <= 0
            else "все трети в плюсе — фильтру нечего выбрасывать"
        )
        lines.append(
            f"  {label}: худшая треть — «{worst['треть']}», "
            f"{worst['R после издержек']:+.3f} ± {worst['±']:.3f}R → {status}"
        )
    return lines


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    trades = trades_with_indicators(load_markets(RISK_FRACTIONS))
    mean, se = _mean_se(trades["r_multiple"])
    print(
        f"\nПортфель paper mode ({', '.join(RISK_FRACTIONS)}), вся история: "
        f"{trades.height} закрытых сделок, {mean:+.3f} ± {se:.3f}R на сделку. "
        f"Сжатие BB — с полугода истории: {trades['squeeze'].is_not_null().sum()} сделок."
    )

    classic = thirds_table(trades, CLASSIC)
    print("\nКлассические настройки, вместе по четырём рынкам (по ним решение):")
    print(classic)
    print("\nРешение (ориентир: треть с R после издержек ≤ 0):")
    print("\n".join(verdict(classic, CLASSIC)))

    long = thirds_table(trades, LONG)
    print("\nДлинные настройки — для сведения:")
    print(long)
    print("\n".join(verdict(long, LONG)))

    print("\nПо рынкам, R после издержек (для сведения):")
    print(by_market_table(trades, CLASSIC))

    print(
        "\nПохожесть на признаки шага 6 — ранговая корреляция Спирмена по сделкам "
        "(около ±1 — почти тот же признак, около 0 — новое):"
    )
    print(similarity_table(trades))


if __name__ == "__main__":
    main()
