"""Общий портфель пробоя (plan.md, 5.5–5.6): cnyrub, si, euro и gold вместе.

Состав: в 5.5 и на экзамене 5.6 вместо золота был газ (ng); после экзамена
газ убран, золото добавлено с риском 1% (решение пользователя 19.10.2026).

Замороженные правила 5.3, позиция на каждом рынке (`run_multi_portfolio`):
рублёвые пары — группа с риском 2%, по 0.67% на пару; золото — 1%; сумма ГО
всех позиций — не больше 50% капитала, не хватает — новая позиция
уменьшается до остатка. По умолчанию данные до 01.01.2026 — так считался
5.5, пока 2026 год был отложенным экзаменом.

Печатает: метрики за период и по годам, что стало с сигналами, вклад
рынков, сколько позиций открыто одновременно, тот же прогон при 2 шагах
проскальзывания, бутстрап по неделям, корреляцию дневного результата
рынков. График — полный отчёт из шести панелей.

Запуск:
  uv run python strategies/st01/run_portfolio.py [--full]
  → strategies/st01/reports/final_portfolio.png
  `--full` — вся история, включая 2026 год (после экзамена 5.6 он открыт).
Отложенный экзамен 2026 года (plan.md, 5.6), тоже один раз:
  uv run python strategies/st01/run_portfolio.py --exam
  → strategies/st01/reports/exam_portfolio.png
  Сигналы с 01.01.2026, капитал 1 000 000 ₽ на эту дату; история до 2026 года
  — только для канала, σ и обычного объёма. Сдал — expectancy портфеля после
  издержек больше нуля. Без cnyrub — для сведения: его 2026 год видели на подборе.
Файлы прошлых запусков не затираются: новый получает суффикс _2, _3…
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

import polars as pl
from check_breakout import bootstrap_expectancy, closed
from rules import frozen_breakout_signals
from run_baseline import CAPITAL, cut, metrics_table, trades_by
from tune_breakout import REPORTS, free_path

from backtest.charts import save_report_chart
from backtest.portfolio import (
    PORTFOLIO_MARGIN,
    MarketData,
    MultiRules,
    PortfolioResult,
    load_markets,
    run_multi_portfolio,
)
from backtest.report import (
    Report,
    build_report,
    daily_equity,
    equity_curve,
    format_value,
    market_timeline,
)

ROUBLE_GROUP_RISK = 0.02
ROUBLE_PAIRS = ("cnyrub", "si", "euro")
RISK_FRACTIONS = {
    **{name: ROUBLE_GROUP_RISK / len(ROUBLE_PAIRS) for name in ROUBLE_PAIRS},
    "gold": 0.01,  # до 19.10.2026 — "ng": 0.01
}
RULES = MultiRules(risk_fractions=RISK_FRACTIONS, max_margin_fraction=0.5)
EXAM_BEFORE = date(2026, 1, 1)
EXAM_START = date(2026, 1, 1)  # первый день сигналов экзамена 5.6
SEEN_IN_TUNING = "cnyrub"
STRESS_SLIPPAGE_TICKS = 2.0


def portfolio(
    markets: dict[str, MarketData], start: date | None = None
) -> tuple[PortfolioResult, Report]:
    """Прогон портфеля; `start` — сигналы и кривая капитала только с этого дня.

    Индикаторы считаются по всей истории: канал 240 свечей и σ за 20 дней
    на первой свече 2026 года уже знают конец 2025-го, как знал бы трейдер.
    """
    signals = pl.concat(
        [
            frozen_breakout_signals(name, data.bars).with_columns(
                pl.lit(name).alias("market")
            )
            for name, data in markets.items()
        ],
        how="diagonal_relaxed",
    )
    timeline = market_timeline(data.bars for data in markets.values())
    if start is not None:
        signals = signals.filter(pl.col("timestamp").dt.date() >= start)
        timeline = timeline.filter(timeline.dt.date() >= start)
    result = run_multi_portfolio(signals, markets, CAPITAL, RULES)
    return result, build_report(result, timeline)


def signal_table(result: PortfolioResult) -> pl.DataFrame:
    """Что стало с сигналами по рынкам; урезанные ГО портфеля — отдельно."""
    log = result.signals.with_columns(
        pl.when((pl.col("status") == "taken") & (pl.col("limit") == PORTFOLIO_MARGIN))
        .then(pl.lit("taken, урезан ГО"))
        .when(pl.col("status") == "zero_size")
        .then(pl.concat_str(pl.lit("zero_size: "), pl.col("limit")))
        .otherwise(pl.col("status"))
        .alias("итог")
    )
    return log.pivot(
        on="market", index="итог", values="timestamp", aggregate_function="len"
    ).fill_null(0)


def positions_table(report: Report) -> pl.DataFrame:
    """Доля времени (свечей расписания) с 0, 1, 2… открытыми позициями."""
    return (
        report.curve.group_by("positions")
        .agg(pl.len().alias("свечей"))
        .sort("positions")
        .with_columns(
            (pl.col("свечей") / pl.col("свечей").sum() * 100)
            .round(1)
            .alias("% времени")
        )
    )


def rouble_same_side(result: PortfolioResult) -> pl.DataFrame:
    """Сколько рублёвых позиций в одну сторону открыто в момент каждого входа."""
    trades = [item.trade for item in result.trades if item.trade.asset in ROUBLE_PAIRS]
    counts = [
        sum(
            1
            for other in trades
            if other.side == trade.side
            and other.entry_time <= trade.entry_time <= other.exit_time
        )
        for trade in trades
    ]
    return (
        pl.DataFrame({"позиций в одну сторону": counts})
        .group_by("позиций в одну сторону")
        .agg(pl.len().alias("входов"))
        .sort("позиций в одну сторону")
    )


def daily_by_market(result: PortfolioResult, timeline: pl.Series) -> pl.DataFrame:
    """Дневной результат каждого рынка в рублях — для корреляции."""
    columns = {}
    for name in RISK_FRACTIONS:
        part = replace(
            result, trades=[t for t in result.trades if t.trade.asset == name]
        )
        daily = daily_equity(equity_curve(part, timeline), CAPITAL)
        columns[name] = daily["equity"].diff().fill_null(0.0)
    return pl.DataFrame(columns)


def stressed(markets: dict[str, MarketData]) -> dict[str, MarketData]:
    return {
        name: replace(
            data, spec=replace(data.spec, slippage_ticks=STRESS_SLIPPAGE_TICKS)
        )
        for name, data in markets.items()
    }


def exam() -> None:
    """Отложенный экзамен 2026 года (plan.md, 5.6)."""
    markets = load_markets(RISK_FRACTIONS)
    result, report = portfolio(markets, EXAM_START)
    s = report.summary
    print(f"\nЭкзамен 2026 года: сигналы с {EXAM_START:%d.%m.%Y} до конца данных.")
    print("\nМетрики. Сделки года — по дате выхода.")
    print(metrics_table(report))
    print("\nЧто стало с сигналами:")
    print(signal_table(result))
    print("\nВклад рынков (сделки, закрытые концом данных, не входят):")
    print(trades_by(report.trades, "market"))

    verdict = "СДАЛ" if (s["expectancy_r"] or 0) > 0 else "НЕ СДАЛ"
    print(
        f"\nИтог по условию 5.6 (expectancy портфеля после издержек > 0): "
        f"{format_value('expectancy_r', s['expectancy_r'])}R на {s['trades']} "
        f"сделках — {verdict}."
    )

    print("\nДля сведения, без решения:")
    hard = portfolio(stressed(markets), EXAM_START)[1].summary
    print(
        f"- при {STRESS_SLIPPAGE_TICKS:g} шагах: expectancy "
        f"{format_value('expectancy_r', hard['expectancy_r'])}R, Sharpe "
        f"{format_value('sharpe', hard['sharpe'])};"
    )
    boot = bootstrap_expectancy(report.trades)
    print(
        f"- бутстрап по неделям: 90% — {boot['low']:.3f}…{boot['high']:.3f}R, "
        f"историй ≤ 0: {boot['not_positive'] * 100:.2f}%;"
    )
    unseen = {name: data for name, data in markets.items() if name != SEEN_IN_TUNING}
    fresh = portfolio(unseen, EXAM_START)[1].summary
    print(
        f"- без {SEEN_IN_TUNING} ({', '.join(unseen)}): expectancy "
        f"{format_value('expectancy_r', fresh['expectancy_r'])}R на "
        f"{fresh['trades']} сделках, Sharpe {format_value('sharpe', fresh['sharpe'])}."
    )

    title = (
        f"Пробой, экзамен 2026: {', '.join(RISK_FRACTIONS)}; "
        f"expectancy {format_value('expectancy_r', s['expectancy_r'])}R"
    )
    path = save_report_chart(report, free_path(REPORTS / "exam_portfolio.png"), title)
    print(f"\nГрафик: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--before",
        type=date.fromisoformat,
        default=EXAM_BEFORE,
        help="Данные только до этой даты, YYYY-MM-DD (по умолчанию 2026-01-01).",
    )
    parser.add_argument(
        "--exam",
        action="store_true",
        help="Отложенный экзамен 2026 года (plan.md, 5.6): сигналы с 01.01.2026.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Вся история, включая 2026 год (открыт после экзамена 5.6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    if args.exam:
        exam()
        return

    before = None if args.full else args.before
    markets = cut(load_markets(RISK_FRACTIONS), before)
    result, report = portfolio(markets)
    s = report.summary
    risks = ", ".join(f"{name} {share:.2%}" for name, share in RISK_FRACTIONS.items())
    print(
        f"\nОбщий портфель, {'вся история' if before is None else f'до {before:%d.%m.%Y}'}. "
        f"Риск на сделку: {risks}; "
        f"ГО всех позиций — до {RULES.max_margin_fraction:.0%} капитала."
    )
    print("\nМетрики. Сделки года — по дате выхода.")
    print(metrics_table(report))

    print("\nЧто стало с сигналами:")
    print(signal_table(result))
    print("\nВклад рынков (сделки, закрытые концом данных, не входят):")
    print(trades_by(report.trades, "market"))
    print("\nЛонги и шорты:")
    print(trades_by(closed(report.trades), "сторона"))
    print("\nСколько позиций открыто одновременно:")
    print(positions_table(report))
    print("\nРублёвые позиции в одну сторону в момент входа (включая новую):")
    print(rouble_same_side(result))

    hard = portfolio(stressed(markets))[1].summary
    print(
        f"\nПри {STRESS_SLIPPAGE_TICKS:g} шагах проскальзывания: expectancy "
        f"{format_value('expectancy_r', hard['expectancy_r'])}R, Sharpe "
        f"{format_value('sharpe', hard['sharpe'])}, CAGR "
        f"{format_value('cagr', hard['cagr'])}."
    )

    boot = bootstrap_expectancy(report.trades)
    print(
        f"Бутстрап по неделям: expectancy {boot['expectancy']:.3f}R, 90% — "
        f"{boot['low']:.3f}…{boot['high']:.3f}R, историй ≤ 0: "
        f"{boot['not_positive'] * 100:.2f}%."
    )

    timeline = market_timeline(data.bars for data in markets.values())
    print("\nКорреляция дневного результата рынков в портфеле:")
    print(daily_by_market(result, timeline).corr().with_columns(pl.all().round(2)))

    title = (
        f"Пробой, общий портфель: {', '.join(RISK_FRACTIONS)}; "
        f"Sharpe {format_value('sharpe', s['sharpe'])}"
    )
    path = save_report_chart(report, free_path(REPORTS / "final_portfolio.png"), title)
    print(f"\nГрафик: {path}")


if __name__ == "__main__":
    main()
