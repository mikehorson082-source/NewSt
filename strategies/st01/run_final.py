"""Пробой по замороженным правилам (plan.md, 5.3) — итоговый прогон и экзамены.

Каждый рынок в одиночку, вся история, правила из `rules.frozen_breakout_signals`.
По каждому рынку: метрики за период и по годам, лонги и шорты, причины
выхода, тот же прогон при 2 шагах проскальзывания, бутстрап по неделям и
сравнение с «купил и держи»; полный отчёт из шести панелей. В конце — сводка
по рынкам.

Итоговый прогон 5.3 (рынки подбора):
  uv run python strategies/st01/run_final.py
  → strategies/st01/reports/final_<рынок>.png
Экзамен 5.4 (рынки, которых стратегия не видела; нужны паспорта в markets.py):
  uv run python strategies/st01/run_final.py --exam si euro silver ...
  → strategies/st01/reports/exam_<рынок>.png
Экзамен запускается один раз, без правок после него. Файлы прошлых запусков
не затираются: новый получает суффикс _2, _3…
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import polars as pl
from check_breakout import bootstrap_row, daily_correlation, trades_by
from rules import frozen_breakout_signals
from run_baseline import CAPITAL, metrics_table
from tune_breakout import REPORTS, free_path

from backtest.baseline import buy_and_hold_result
from backtest.charts import save_report_chart
from backtest.portfolio import MarketData, load_markets, run_portfolio
from backtest.report import Report, build_report, format_value, market_timeline

TUNING_MARKETS = ("cnyrub", "mix", "brent", "gold")
STRESS_SLIPPAGE_TICKS = 2.0


def solo(name: str, data: MarketData) -> Report:
    signals = frozen_breakout_signals(name, data.bars).with_columns(
        pl.lit(name).alias("market")
    )
    result = run_portfolio(signals, {name: data}, CAPITAL)
    return build_report(result, market_timeline([data.bars]))


def market_report(name: str, data: MarketData, prefix: str) -> dict[str, object]:
    report = solo(name, data)
    stressed = solo(
        name,
        replace(data, spec=replace(data.spec, slippage_ticks=STRESS_SLIPPAGE_TICKS)),
    ).summary
    holding = build_report(
        buy_and_hold_result(data.spec, data.bars, CAPITAL),
        market_timeline([data.bars]),
    )
    boot = bootstrap_row(name, report.trades)
    s = report.summary

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print("Метрики. Сделки года — по дате выхода.")
    print(metrics_table(report))
    print("\nЛонги и шорты:")
    print(trades_by(report.trades, "сторона"))
    print("\nПричины выхода:")
    print(trades_by(report.trades, "exit_reason"))
    print(
        f"\nПри {STRESS_SLIPPAGE_TICKS:g} шагах проскальзывания: expectancy "
        f"{format_value('expectancy_r', stressed['expectancy_r'])}R, Sharpe "
        f"{format_value('sharpe', stressed['sharpe'])}."
    )
    print(
        f"«Купил и держи»: Sharpe {format_value('sharpe', holding.summary['sharpe'])}, "
        f"CAGR {format_value('cagr', holding.summary['cagr'])}, "
        f"Max DD {format_value('max_drawdown', holding.summary['max_drawdown'])}; "
        f"корреляция дневных доходностей с пробоем "
        f"{daily_correlation(report, holding):.2f}."
    )
    title = f"Пробой, замороженные правила: {name}"
    path = save_report_chart(report, free_path(REPORTS / f"{prefix}_{name}.png"), title)
    print(f"График: {path}")

    return {
        "рынок": name,
        "сделок": s["trades"],
        "expectancy, R": format_value("expectancy_r", s["expectancy_r"]),
        "при 2 шагах, R": format_value("expectancy_r", stressed["expectancy_r"]),
        "Sharpe": format_value("sharpe", s["sharpe"]),
        "±": format_value("sharpe_se", s["sharpe_se"]),
        "Max DD": format_value("max_drawdown", s["max_drawdown"]),
        "бутстрап: историй ≤ 0, %": boot["доля историй ≤ 0, %"],
        "Sharpe «держать»": format_value("sharpe", holding.summary["sharpe"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пробой по замороженным правилам.")
    parser.add_argument(
        "--exam",
        nargs="+",
        metavar="РЫНОК",
        help="Экзамен на рынках, которых стратегия не видела (exam_<рынок>.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    names = args.exam or list(TUNING_MARKETS)
    prefix = "exam" if args.exam else "final"
    rows = [
        market_report(name, data, prefix) for name, data in load_markets(names).items()
    ]
    print(f"\n{'=' * 78}\nСводка ({prefix}):")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
