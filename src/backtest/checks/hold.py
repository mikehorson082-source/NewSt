"""Выход по расписанию на реальных данных: убирает ли он гэпы через стоп.

Те же случайные сигналы, что в backtest.checks.portfolio (seed 7, стоп 0.5%,
цель 1%, не дольше 32 свечей), три раза — с разным `hold`:
- any — как раньше: позиция живёт через ночи, выходные и праздники;
- overnight — обычные ночи держит, перед выходными и праздниками выходит;
- day — только внутри дня.

Смотрим хвост, а не доходность: сколько сделок ушло против нас дальше 1.5R
при стопе в 1R (гэп через стоп) и какая худшая. Сигналы случайные, поэтому expectancy около нуля в
любом варианте, а сами сделки в вариантах разные (после раннего выхода
портфель берёт другой сигнал) — разница expectancy в сотые доли R здесь шум.
Ничего не записывает.
Запуск: uv run python -m backtest.checks.hold
"""

from __future__ import annotations

import polars as pl

from backtest.checks.portfolio import CAPITAL, random_signals
from backtest.portfolio import load_markets, run_portfolio
from backtest.position import HOLD_ANY, HOLD_DAY, HOLD_EXITS, HOLD_OVERNIGHT

HOLDS = (HOLD_ANY, HOLD_OVERNIGHT, HOLD_DAY)
GAP_R = 1.5
REASONS = ("stop", "target", "time", *(reason for _, reason in HOLD_EXITS.values()))


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets()
    signals = random_signals(markets)

    print("1. Отметки расписания в рядах: сколько раз за год позицию закроют.")
    print("   day_end — последняя свеча дня, break — перед выходными и праздниками.")
    for name, data in markets.items():
        marks = (
            data.bars.group_by(pl.col("timestamp").dt.year().alias("год"))
            .agg(
                pl.col("is_day_end").sum().alias("day_end"),
                pl.col("is_before_break").sum().alias("break"),
            )
            .sort("год")
        )
        print(name)
        print(marks)

    summary = []
    worst = []
    for hold in HOLDS:
        result = run_portfolio(
            signals.with_columns(pl.lit(hold).alias("hold")), markets, CAPITAL
        )
        trades = result.trade_log().filter(pl.col("exit_reason") != "end")
        reasons = dict(trades.group_by("exit_reason").len().iter_rows())
        summary.append(
            {
                "hold": hold,
                "сделок": trades.height,
                "expectancy, R": round(trades["r_multiple"].mean(), 3),
                "худшая, R": round(trades["r_multiple"].min(), 2),
                f"гэпов > {GAP_R}R": trades.filter(pl.col("mae_r") < -GAP_R).height,
                **{reason: reasons.get(reason, 0) for reason in REASONS},
            }
        )
        worst.append(
            trades.sort("r_multiple")
            .head(3)
            .select(
                pl.lit(hold).alias("hold"),
                "market",
                "side",
                pl.col("entry_time").dt.strftime("%Y-%m-%d %a %H:%M").alias("вход"),
                pl.col("exit_time").dt.strftime("%Y-%m-%d %a %H:%M").alias("выход"),
                "exit_reason",
                pl.col("r_multiple").round(2).alias("итог, R"),
            )
        )

    print("\n2. Те же сигналы при трёх правилах. Справа — как закрылись сделки.")
    print(pl.DataFrame(summary))

    print("\n3. Три худшие сделки каждого варианта:")
    print(pl.concat(worst))


if __name__ == "__main__":
    main()
