"""Портфель на реальных данных: срабатывают ли его правила.

Случайные сигналы на всех четырёх рынках (в среднем один на 50 свечей
каждого рынка, seed фиксирован): сторона случайная, стоп 0.5%, цель 1%,
не дольше 32 свечей (8 часов). Результат по деньгам смысла не имеет —
сигналы случайные. Смотрим другое: правило, которое ни разу не сработало,
не проверено. Ничего не записывает.
Запуск: uv run python st01/check_portfolio.py
"""

from __future__ import annotations

import random

import polars as pl

from backtest.portfolio import load_markets, run_portfolio
from backtest.position import LONG, SHORT

CAPITAL = 1_000_000.0
SIGNAL_PROBABILITY = 0.02
STOP = 0.005
TARGET = 0.01
MAX_BARS = 32
SEED = 7


def random_signals(markets) -> pl.DataFrame:
    rng = random.Random(SEED)
    rows = []
    for name, data in markets.items():
        bars = data.bars.select("timestamp", "close")
        for timestamp, close in bars.iter_rows():
            if rng.random() >= SIGNAL_PROBABILITY:
                continue
            side = rng.choice((LONG, SHORT))
            rows.append(
                {
                    "market": name,
                    "timestamp": timestamp,
                    "side": side,
                    "stop": close * (1 - side * STOP),
                    "target": close * (1 + side * TARGET),
                    "max_bars": MAX_BARS,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_thousands_separator(" ")

    markets = load_markets()
    signals = random_signals(markets)
    result = run_portfolio(signals, markets, CAPITAL)
    trades = result.trade_log()

    print(f"Сигналов {signals.height}, сделок {trades.height}.")
    print("\n1. Что стало с сигналами (taken, busy, lost_tie, zero_size...):")
    print(
        result.signals.pivot(
            on="status", index="market", values="timestamp", aggregate_function="len"
        ).fill_null(0)
    )

    print(
        "\nПочему 0 контрактов (zero_size): volume — свеча входа обычно почти пустая."
    )
    print(
        result.signals.filter(pl.col("status") == "zero_size")
        .group_by("market", "limit")
        .len()
        .sort("market", "limit")
    )

    print("\n2. Как найден выход. minutes — порядок стопа и цели взят из минуток:")
    print(
        trades.group_by("exit_reason", "exit_resolution")
        .len()
        .sort("exit_reason", "exit_resolution")
    )

    print("\n3. Размер следует за капиталом: риск сделки ÷ капитал перед ней.")
    print(
        trades.group_by("market", "limit")
        .agg(
            pl.len().alias("сделок"),
            pl.col("contracts").min().alias("контр. мин"),
            pl.col("contracts").max().alias("контр. макс"),
            (pl.col("risk") / pl.col("capital_before") * 100)
            .median()
            .round(3)
            .alias("риск % медиана"),
            pl.col("cost_share").median().mul(100).round(1).alias("издержки % риска"),
        )
        .sort("market", "limit")
    )
    capital = trades["capital_before"]
    print(
        f"Капитал перед сделками: от {capital.min():,.0f} до {capital.max():,.0f} ₽, "
        f"в конце {result.final_capital:,.0f} ₽.".replace(",", " ")
    )

    print(
        "\n4. Доля сделки в объёме свечи входа (выше 10% — 1 шаг проскальзывания мало):"
    )
    print(
        trades.group_by("market")
        .agg(
            pl.col("volume_share").median().mul(100).round(1).alias("медиана %"),
            pl.col("volume_share").quantile(0.95).mul(100).round(1).alias("95% %"),
            pl.col("volume_share").max().mul(100).round(1).alias("макс %"),
            (pl.col("volume_share") > 0.1).sum().alias("сделок > 10%"),
        )
        .sort("market")
    )

    print(
        "\n5. Десять сделок с наибольшей долей: объём свечи входа против обычного "
        "(обычный известен в момент сигнала, фактический — нет):"
    )
    print(
        trades.sort("volume_share", descending=True, nulls_last=True)
        .head(10)
        .select(
            "market",
            pl.col("entry_time").dt.strftime("%Y-%m-%d %a %H:%M").alias("вход"),
            "contracts",
            "limit",
            pl.col("usual_volume").alias("обычный объём"),
            pl.col("entry_volume").alias("объём свечи"),
            pl.col("volume_share").mul(100).round(1).alias("доля %"),
        )
    )


if __name__ == "__main__":
    main()
