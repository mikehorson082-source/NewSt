"""Шаг 4: две простые стратегии на четырёх рынках через общий портфель.

Каждая стратегия прогоняется отдельно: пробой канала и возврат к средней
часто смотрят в разные стороны на одной и той же свече, а позиция в портфеле
одна. Правила и параметры — в rules.py, заданы заранее.

По умолчанию берётся вся история; `--before YYYY-MM-DD` обрезает данные.

Печатает по каждой стратегии: что стало с сигналами, метрики за период и по
годам, сделки по рынкам, причинам выхода и сессиям, результат при двух шагах
проскальзывания. Рисует strategies/st01/reports/baseline_<стратегия>.png.
Запуск: uv run python strategies/st01/run_baseline.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl
from rules import breakout_signals, reversion_signals

from backtest.charts import save_report_chart
from backtest.portfolio import MarketData, load_markets, run_portfolio
from backtest.report import LABELS, build_report, format_value, market_timeline
from backtest.sensitivity import by_session, with_costs

REPORTS = Path(__file__).resolve().parent / "reports"
CAPITAL = 1_000_000.0
STRESS_SLIPPAGE_TICKS = 2.0
GAP_R = 1.5

STRATEGIES = {
    "breakout": ("Пробой канала", breakout_signals),
    "reversion": ("Возврат к средней", reversion_signals),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Две простые стратегии st01.")
    parser.add_argument(
        "--before",
        type=date.fromisoformat,
        help="Данные до этой даты, YYYY-MM-DD. По умолчанию вся история.",
    )
    parser.add_argument(
        "--only",
        choices=list(STRATEGIES),
        help="Только одна стратегия.",
    )
    return parser.parse_args()


def cut(markets: dict[str, MarketData], before: date | None) -> dict[str, MarketData]:
    """Те же рынки, свечи только до даты `before`; None — вся история."""
    if before is None:
        return markets
    return {
        name: replace(
            data, bars=data.bars.filter(pl.col("timestamp").dt.date() < before)
        )
        for name, data in markets.items()
    }


def all_signals(rule, markets: dict[str, MarketData]) -> pl.DataFrame:
    """Сигналы правила по всем рынкам одной таблицей."""
    return pl.concat(
        [
            rule(data.bars).with_columns(pl.lit(name).alias("market"))
            for name, data in markets.items()
        ]
    )


def metrics_table(report) -> pl.DataFrame:
    table = {"метрика": list(LABELS.values())}
    table["весь период"] = [format_value(k, report.summary[k]) for k in LABELS]
    for row in report.by_year.iter_rows(named=True):
        table[str(row["year"])] = [format_value(k, row[k]) for k in LABELS]
    return pl.DataFrame(table)


def trades_by(trades: pl.DataFrame, column: str) -> pl.DataFrame:
    """Сделки по рынку или причине выхода; закрытые концом данных не входят."""
    return (
        trades.filter(pl.col("exit_reason") != "end")
        .group_by(column)
        .agg(
            pl.len().alias("сделок"),
            ((pl.col("net_pnl") > 0).mean() * 100).round(1).alias("выигрышных %"),
            pl.col("r_multiple").mean().round(3).alias("expectancy, R"),
            pl.col("r_multiple").sum().round(1).alias("сумма, R"),
            pl.col("net_pnl").sum().round(0).alias("итог, ₽"),
            pl.col("bars_held").median().alias("свечей, медиана"),
            (pl.col("mae_r") < -GAP_R).sum().alias(f"гэпов > {GAP_R}R"),
        )
        .sort(column)
    )


def run(key: str, markets: dict[str, MarketData], timeline: pl.Series) -> None:
    title, rule = STRATEGIES[key]
    signals = all_signals(rule, markets)
    result = run_portfolio(signals, markets, CAPITAL)
    report = build_report(result, timeline)
    trades = report.trades

    print(f"\n{'=' * 78}\n{title}: сигналов {signals.height}, сделок {trades.height}")
    print(f"{'=' * 78}")

    print("\n1. Что стало с сигналами (taken — исполнен, busy — позиция занята):")
    print(
        result.signals.pivot(
            on="status", index="market", values="timestamp", aggregate_function="len"
        ).fill_null(0)
    )

    print("\n2. Метрики. Сделки года — по дате выхода.")
    print(metrics_table(report))

    print("\n3. Сделки по рынкам:")
    print(trades_by(trades, "market"))

    print("\n4. Сделки по причине выхода:")
    print(trades_by(trades, "exit_reason"))

    print("\n5. Сделки по сессии входа:")
    print(
        by_session(trades).select(
            pl.col("session").alias("сессия"),
            pl.col("trades").alias("сделок"),
            (pl.col("win_rate") * 100).round(1).alias("выигрышных %"),
            pl.col("expectancy_r").round(3).alias("expectancy, R"),
            pl.col("costs_per_trade_r").round(3).alias("издержки, R"),
            (pl.col("volume_share_median") * 100).round(2).alias("доля в свече %"),
        )
    )

    stressed = run_portfolio(
        signals, with_costs(markets, STRESS_SLIPPAGE_TICKS, 1.0), CAPITAL
    )
    stress = build_report(stressed, timeline).summary
    print(
        f"\n6. То же при {STRESS_SLIPPAGE_TICKS:g} шагах проскальзывания: "
        f"expectancy {format_value('expectancy_r', stress['expectancy_r'])}R "
        f"(при 1 шаге {format_value('expectancy_r', report.summary['expectancy_r'])}R), "
        f"доходность {format_value('total_return', stress['total_return'])}."
    )

    path = save_report_chart(report, REPORTS / f"baseline_{key}.png", title)
    print(f"\nГрафики: {path}")


def main() -> None:
    args = parse_args()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = cut(load_markets(), args.before)
    timeline = market_timeline(data.bars for data in markets.values())
    capital = f"{CAPITAL:,.0f}".replace(",", " ")
    print(
        f"Данные: {timeline.min():%d.%m.%Y} — {timeline.max():%d.%m.%Y}, "
        f"капитал {capital} ₽."
    )
    for key in [args.only] if args.only else STRATEGIES:
        run(key, markets, timeline)


if __name__ == "__main__":
    main()
