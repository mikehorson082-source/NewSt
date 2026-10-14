"""Сколько стоит честная цена стопа: выход по минуткам против выхода по уровню.

Свеча задела стоп — раньше выход считался ровно по уровню. Теперь симулятор
смотрит минутки этой свечи: если минутка открылась сразу за стопом, выход по
её открытию. Пример: стоп 99.00, минутка открылась на 98.90 — выход по 98.90.
То же для цели: минутка открылась за целью — выход по её открытию.

Пробой канала, как в run_baseline.py, каждый рынок в одиночку, два прогона:
с минутками (как теперь) и без них (выход по уровню, как раньше). Без минуток
свеча, задевшая и стоп, и цель, закрывается по стопу — у пробоя таких почти
нет (стоп по пессимизму 0.0%), поэтому разница — именно цена исполнения.

Ничего не записывает.
Запуск: uv run python strategies/st01/check_stop_fills.py
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
from rules import breakout_signals
from run_baseline import CAPITAL, all_signals

from backtest.portfolio import MarketData, load_markets, run_portfolio


def solo_trades(name: str, data: MarketData) -> pl.DataFrame:
    market = {name: data}
    signals = all_signals(breakout_signals, market)
    trades = run_portfolio(signals, market, CAPITAL).trade_log()
    return trades.filter(pl.col("exit_reason") != "end")


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    summary = []
    worst = []
    for name, data in load_markets().items():
        by_minutes = solo_trades(name, data)
        by_level = solo_trades(name, replace(data, minutes=None))
        paired = by_minutes.join(
            by_level.select(
                "entry_time",
                pl.col("exit_time").alias("exit_time_level"),
                pl.col("exit_price").alias("exit_price_level"),
                pl.col("r_multiple").alias("r_level"),
            ),
            on="entry_time",
        ).with_columns((pl.col("r_multiple") - pl.col("r_level")).alias("разница, R"))
        changed = paired.filter(pl.col("разница, R").abs() > 1e-9)
        summary.append(
            {
                "рынок": name,
                "сделок": by_minutes.height,
                "expectancy по минуткам, R": round(by_minutes["r_multiple"].mean(), 3),
                "по уровню, R": round(by_level["r_multiple"].mean(), 3),
                "сделок с другой ценой": changed.height,
                "из них хуже": changed.filter(pl.col("разница, R") < 0).height,
                "средняя разница на сделку, R": round(paired["разница, R"].mean(), 4),
                "не нашли пару": by_minutes.height - paired.height,
            }
        )
        worst.append(
            changed.sort("разница, R")
            .head(3)
            .select(
                pl.lit(name).alias("рынок"),
                "side",
                pl.col("exit_time").dt.strftime("%Y-%m-%d %a %H:%M").alias("выход"),
                "exit_reason",
                pl.col("exit_price_level").alias("цена по уровню"),
                pl.col("exit_price").alias("цена по минуткам"),
                pl.col("разница, R").round(3),
            )
        )

    print("1. Пробой, каждый рынок в одиночку: выход по минуткам против уровня.")
    print(
        "   Отрицательная разница — по минуткам вышло хуже (минутка открылась за стопом)."
    )
    print(pl.DataFrame(summary))
    print("\n2. Три самые большие разницы на каждом рынке:")
    print(pl.concat(worst))


if __name__ == "__main__":
    main()
