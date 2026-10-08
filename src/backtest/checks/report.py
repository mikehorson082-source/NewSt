"""Отчёт по портфелю на случайных сигналах: метрики, годы и графики.

Те же случайные сигналы, что в backtest.checks.portfolio (seed 7, стоп 0.5%,
цель 1%). Стратегии в них нет, и отчёт должен сам это показать: Sharpe
ниже нуля, вероятность «Sharpe > 0» маленькая, запас по издержкам меньше
1 раза или пустой (до издержек тоже убыток). Печатает таблицу метрик за весь
период и по годам, рисует src/backtest/checks/reports/random_signals.png.
Запуск: uv run python -m backtest.checks.report
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from backtest.charts import save_report_chart
from backtest.checks.portfolio import CAPITAL, random_signals
from backtest.portfolio import load_markets, run_portfolio
from backtest.report import LABELS, build_report, format_value, market_timeline
from backtest.sensitivity import by_session

CHART = Path(__file__).resolve().parent / "reports" / "random_signals.png"
GAP_R = 1.5


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets()
    result = run_portfolio(random_signals(markets), markets, CAPITAL)
    report = build_report(
        result, market_timeline(data.bars for data in markets.values())
    )

    table = {"метрика": list(LABELS.values())}
    table["весь период"] = [format_value(k, report.summary[k]) for k in LABELS]
    for row in report.by_year.iter_rows(named=True):
        table[str(row["year"])] = [format_value(k, row[k]) for k in LABELS]
    print("Случайные сигналы на четырёх рынках. Сделки года — по дате выхода.")
    print(pl.DataFrame(table))

    print(
        f"\nСделки, ушедшие против нас дальше {GAP_R}R при стопе в 1R "
        "(гэп через стоп: риск до стопа его не учитывает):"
    )
    gaps = report.trades.filter(pl.col("mae_r") < -GAP_R)
    print(gaps.group_by("market").len().sort("market"))
    print(
        gaps.sort("mae_r")
        .head(10)
        .select(
            "market",
            "side",
            pl.col("entry_time").dt.strftime("%Y-%m-%d %a %H:%M").alias("вход"),
            pl.col("exit_time").dt.strftime("%Y-%m-%d %a %H:%M").alias("выход"),
            "exit_reason",
            "bars_held",
            pl.col("mae_r").round(2).alias("MAE, R"),
            pl.col("r_multiple").round(2).alias("итог, R"),
        )
    )

    print(
        "\nСделки по сессии входа (спред везде 1 шаг — настоящего спреда нет, "
        "смотрим результат, объём свечи и гэпы):"
    )
    print(
        by_session(report.trades).select(
            pl.col("session").alias("сессия"),
            pl.col("trades").alias("сделок"),
            (pl.col("win_rate") * 100).round(1).alias("выигрышных %"),
            pl.col("expectancy_r").round(3).alias("expectancy, R"),
            pl.col("costs_per_trade_r").round(3).alias("издержки, R"),
            (pl.col("volume_share_median") * 100).round(2).alias("доля в свече %"),
            pl.col("over_10pct").alias("> 10% свечи"),
            pl.col("gaps_over_1_5r").alias("гэпов > 1.5R"),
        )
    )

    path = save_report_chart(report, CHART, "Случайные сигналы, 4 рынка")
    print(f"\nГрафики: {path}")


if __name__ == "__main__":
    main()
