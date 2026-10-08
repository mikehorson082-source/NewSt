"""«Купил и держи» на реальных данных: симулятор против NAV склейки.

Для каждого рынка — один контракт лонг с первой свечи до последней.
1. Без издержек: на каждой свече прибыль симулятора, приведённая к
   доходности, должна совпасть с доходностью NAV. Ошибка больше 1e-9 —
   симулятор и склейка считают по-разному.
2. С издержками: сколько стоит один перенос в б.п. — это число можно
   подставить в сборку NAV (`build_continuous_futures.py --roll-cost-bps`),
   которая сейчас считает переносы бесплатными.
3. Что дало бы просто держать рынок: рост NAV, CAGR, худшая просадка.
4. Базовая линия для стратегий: 1 000 000 ₽ в рынок с плечом 1× через тот
   же симулятор и отчёт, что у стратегий. Таблица сравнения рынков,
   доходность по годам и графики src/backtest/checks/reports/
   buy_and_hold_<рынок>.png.
Запуск: uv run python -m backtest.checks.buy_and_hold
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from backtest.baseline import (
    buy_and_hold,
    buy_and_hold_result,
    free_spec,
    nav_comparison,
    nav_summary,
    roll_costs_bps,
)
from backtest.charts import save_report_chart
from backtest.markets import MARKETS, load_market
from backtest.report import LABELS, build_report, format_value, market_timeline

TOLERANCE = 1e-9
CAPITAL = 1_000_000.0
REPORTS = Path(__file__).resolve().parent / "reports"
# Метрики для сравнения рынков; у «купил и держи» одна сделка, закрытая концом
# данных, — процент выигрышных и Profit Factor для неё не определены.
COMPARE = (
    "end_capital",
    "total_return",
    "cagr",
    "max_drawdown",
    "longest_drawdown_days",
    "sharpe",
    "sharpe_se",
    "psr",
    "sortino",
    "calmar",
    "costs",
)


def rub(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    rows = []
    worst_bars = []
    reports = {}
    for name in MARKETS:
        params, bars = load_market(name)
        spec = params.contract_spec()

        free = buy_and_hold(free_spec(spec), bars)
        comparison = nav_comparison(free, bars)
        worst = comparison.sort("difference", descending=True).head(1)
        worst_bars.append(worst.with_columns(pl.lit(name).alias("market")))

        paid = buy_and_hold(spec, bars)
        rolls = roll_costs_bps(spec, bars)
        summary = nav_summary(bars)
        result = buy_and_hold_result(spec, bars, CAPITAL)
        reports[name] = (result, build_report(result, market_timeline([bars])))
        rows.append(
            {
                "рынок": name,
                "свечей сверено": comparison.height,
                "из них переносов": int(comparison["is_first_bar_after_roll"].sum()),
                "макс. расхождение": float(comparison["difference"].max()),
                "сходится": bool(comparison["difference"].max() < TOLERANCE),
                "переносов": rolls.height,
                "перенос, б.п. (медиана)": round(rolls["cost_bps"].median(), 2),
                "перенос, б.п. (макс)": round(rolls["cost_bps"].max(), 2),
                "1 контракт до издержек, ₽": rub(paid.pnl_before_costs),
                "переносы, ₽": rub(paid.roll_costs),
                "NAV: рост": f"{summary['total_return'] * 100:.1f}%",
                "NAV: CAGR": f"{summary['cagr'] * 100:.1f}%",
                "NAV: Max DD": f"{summary['max_drawdown'] * 100:.1f}%",
            }
        )

    table = pl.DataFrame(rows)
    print("1. Симулятор против NAV на каждой свече (без издержек):")
    print(
        table.select(
            "рынок",
            "свечей сверено",
            "из них переносов",
            "макс. расхождение",
            "сходится",
        )
    )
    print("\nСвеча с наибольшим расхождением на каждом рынке:")
    print(pl.concat(worst_bars).select("market", pl.exclude("market")))

    print("\n2. Стоимость переноса (проскальзывание двух ног и два сбора):")
    print(
        table.select(
            "рынок",
            "переносов",
            "перенос, б.п. (медиана)",
            "перенос, б.п. (макс)",
            "1 контракт до издержек, ₽",
            "переносы, ₽",
        )
    )
    print(
        "Медиану можно подставить в сборку NAV: "
        "build_continuous_futures.py --roll-cost-bps <число>."
    )

    print("\n3. Что дало бы просто держать рынок (NAV, без издержек переноса):")
    print(table.select("рынок", "NAV: рост", "NAV: CAGR", "NAV: Max DD"))

    capital = format_value("end_capital", CAPITAL)
    print(
        f"\n4. Базовая линия: {capital} ₽ в рынок с плечом 1×, тот же симулятор "
        "и отчёт, что у стратегий (переносы платные)."
    )
    compare = {"метрика": [LABELS[key] for key in COMPARE]}
    contracts = {"метрика": ["Контрактов", "1R = номинал на входе, ₽"]}
    for name, (result, report) in reports.items():
        compare[name] = [format_value(key, report.summary[key]) for key in COMPARE]
        size = result.trades[0].size
        contracts[name] = [str(size.contracts), format_value("costs", size.risk)]
    print(pl.DataFrame(contracts))
    print(pl.DataFrame(compare))

    print("\nДоходность по годам:")
    years = None
    for name, (_, report) in reports.items():
        column = report.by_year.select(
            "year",
            pl.col("total_return")
            .map_elements(
                lambda v: format_value("total_return", v), return_dtype=pl.String
            )
            .alias(name),
        )
        years = (
            column
            if years is None
            else years.join(column, on="year", how="full", coalesce=True)
        )
    print(years.sort("year").rename({"year": "год"}))

    for name, (_, report) in reports.items():
        path = save_report_chart(
            report,
            REPORTS / f"buy_and_hold_{name}.png",
            f"Купил и держи: {name}, плечо 1×",
        )
        print(f"График: {path}")


if __name__ == "__main__":
    main()
