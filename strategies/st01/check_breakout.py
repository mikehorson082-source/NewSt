"""Проверка пробоя канала на прочность: откуда взялся плюс.

Первый прогон (run_baseline.py) дал +0.14R на сделку. Прежде чем нести
стратегию в walk-forward, четыре вопроса — без подбора параметров:

1. Лонги и шорты отдельно. Золото все эти годы росло: если зарабатывают
   только лонги по растущему рынку, это не пробой, а «держал золото».
2. Каждый рынок в одиночку. В общем портфеле позиция одна, и 95% сигналов
   получили `busy`: какой рынок занял место — во многом случайность.
   В одиночку у рынка нет очереди, и сделок больше.
3. Против «купил и держи». Стратегия должна быть лучше, чем просто держать
   рынок, и не повторять его: корреляция дневных доходностей около нуля.
4. Бутстрап. Недели сделок перемешиваются с возвращением 10 000 раз: в каких
   пределах лежит expectancy и как часто она оказывается не выше нуля.
   Перемешиваются недели, а не сделки: сделки одной недели похожи друг на
   друга (один и тот же тренд), а через выходные позиция не переходит.

Ничего не записывает.
Запуск: uv run python strategies/st01/check_breakout.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rules import breakout_signals
from run_baseline import CAPITAL, all_signals

from backtest.baseline import buy_and_hold_result
from backtest.portfolio import MarketData, load_markets, run_portfolio
from backtest.position import LONG
from backtest.report import Report, build_report, format_value, market_timeline

BOOTSTRAP_RUNS = 10_000
BOOTSTRAP_SEED = 1
# Границы интервала: между ними лежит 90% перемешанных историй.
LOW, HIGH = 5, 95
COMPARE = ("cagr", "max_drawdown", "sharpe", "calmar")


def closed(trades: pl.DataFrame) -> pl.DataFrame:
    """Сделки, которые закрыл рынок или расписание, а не конец данных."""
    return trades.filter(pl.col("exit_reason") != "end").with_columns(
        pl.when(pl.col("side") == LONG)
        .then(pl.lit("лонг"))
        .otherwise(pl.lit("шорт"))
        .alias("сторона")
    )


def trades_by(trades: pl.DataFrame, *columns: str) -> pl.DataFrame:
    """Число сделок и результат в R по группам."""
    return (
        closed(trades)
        .group_by(*columns)
        .agg(
            pl.len().alias("сделок"),
            ((pl.col("net_pnl") > 0).mean() * 100).round(1).alias("выигрышных %"),
            pl.col("r_multiple").mean().round(3).alias("expectancy, R"),
            pl.col("r_multiple").sum().round(1).alias("сумма, R"),
        )
        .sort(*columns)
    )


def bootstrap_expectancy(
    trades: pl.DataFrame, runs: int = BOOTSTRAP_RUNS, seed: int = BOOTSTRAP_SEED
) -> dict[str, float | int]:
    """Разброс expectancy в R, если недели сделок выпали бы в другом наборе.

    Неделя — по дате выхода. Из недель с возвращением набирается столько же
    недель, сколько было, и считается expectancy: сумма R ÷ число сделок.
    Пример: 200 недель, из них 30 дали почти всю прибыль. В перемешанной
    истории таких недель может выпасть 20 или 40 — интервал покажет, насколько
    результат зависит от этого везения.
    """
    weeks = (
        closed(trades)
        .group_by(
            pl.col("exit_time").dt.iso_year().alias("_year"),
            pl.col("exit_time").dt.week().alias("_week"),
        )
        .agg(pl.col("r_multiple").sum().alias("r"), pl.len().alias("n"))
        .sort("_year", "_week")
    )
    total_r = weeks["r"].to_numpy()
    count = weeks["n"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, weeks.height, size=(runs, weeks.height))
    samples = total_r[picks].sum(axis=1) / count[picks].sum(axis=1)
    return {
        "weeks": weeks.height,
        "trades": int(count.sum()),
        "expectancy": float(total_r.sum() / count.sum()),
        "low": float(np.percentile(samples, LOW)),
        "high": float(np.percentile(samples, HIGH)),
        "not_positive": float((samples <= 0).mean()),
    }


def bootstrap_row(name: str, trades: pl.DataFrame) -> dict[str, object]:
    stats = bootstrap_expectancy(trades)
    return {
        "прогон": name,
        "недель": stats["weeks"],
        "сделок": stats["trades"],
        "expectancy, R": round(stats["expectancy"], 3),
        f"нижняя граница ({LOW}%)": round(stats["low"], 3),
        f"верхняя граница ({HIGH}%)": round(stats["high"], 3),
        "доля историй ≤ 0, %": round(stats["not_positive"] * 100, 1),
    }


def daily_correlation(strategy: Report, holding: Report) -> float | None:
    """Корреляция дневных доходностей стратегии и «купил и держи»."""
    joined = strategy.daily.select("date", pl.col("return").alias("strategy")).join(
        holding.daily.select("date", pl.col("return").alias("holding")), on="date"
    )
    return joined.select(pl.corr("strategy", "holding")).item()


def solo(name: str, data: MarketData) -> Report:
    """Пробой на одном рынке: позиция одна, но очереди из других рынков нет."""
    market = {name: data}
    result = run_portfolio(all_signals(breakout_signals, market), market, CAPITAL)
    return build_report(result, market_timeline([data.bars]))


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets()
    timeline = market_timeline(data.bars for data in markets.values())
    portfolio = build_report(
        run_portfolio(all_signals(breakout_signals, markets), markets, CAPITAL),
        timeline,
    )
    alone = {name: solo(name, data) for name, data in markets.items()}
    holding = {
        name: build_report(
            buy_and_hold_result(data.spec, data.bars, CAPITAL),
            market_timeline([data.bars]),
        )
        for name, data in markets.items()
    }

    print("1. Лонги и шорты в общем портфеле (тот же прогон, что в run_baseline).")
    print("   Плохо, если весь плюс — в лонгах по рынку, который и так рос.")
    print(trades_by(portfolio.trades, "сторона"))
    print(trades_by(portfolio.trades, "market", "сторона"))

    print("\n2. Каждый рынок в одиночку: без очереди за единственной позицией.")
    rows = []
    for name, report in alone.items():
        s = report.summary
        rows.append(
            {
                "рынок": name,
                "сделок": s["trades"],
                "expectancy, R": format_value("expectancy_r", s["expectancy_r"]),
                "доходность": format_value("total_return", s["total_return"]),
                "CAGR": format_value("cagr", s["cagr"]),
                "Max DD": format_value("max_drawdown", s["max_drawdown"]),
                "Sharpe": format_value("sharpe", s["sharpe"]),
                "± погрешность": format_value("sharpe_se", s["sharpe_se"]),
                "в позиции": format_value("exposure", s["exposure"]),
            }
        )
    print(pl.DataFrame(rows))

    print("\n   Лонги и шорты каждого рынка в одиночку:")
    sides = pl.concat(
        [
            trades_by(report.trades, "сторона").select(
                pl.lit(name).alias("рынок"), pl.all()
            )
            for name, report in alone.items()
        ]
    )
    print(sides)

    print("\n   Expectancy в R по годам (сделки — по году выхода):")
    years = None
    for name, report in alone.items():
        column = report.by_year.select(
            pl.col("year").alias("год"), pl.col("expectancy_r").round(2).alias(name)
        )
        years = (
            column
            if years is None
            else years.join(column, on="год", how="full", coalesce=True)
        )
    print(years.sort("год"))

    print(
        "\n3. Против «купил и держи» (весь капитал в рынок, плечо 1×, переносы "
        "платные)."
    )
    print("   Размер позиций разный, поэтому сравниваем Sharpe и Calmar, а не рубли.")
    print("   Корреляция около 0 — стратегия не повторяет рынок.")
    compare = []
    for name in markets:
        strategy, hold = alone[name].summary, holding[name].summary
        row: dict[str, object] = {"рынок": name}
        for key, label in zip(
            COMPARE, ("CAGR", "Max DD", "Sharpe", "Calmar"), strict=True
        ):
            row[f"{label}: пробой"] = format_value(key, strategy[key])
            row[f"{label}: держать"] = format_value(key, hold[key])
        correlation = daily_correlation(alone[name], holding[name])
        row["корреляция"] = None if correlation is None else round(correlation, 2)
        compare.append(row)
    print(pl.DataFrame(compare))

    print(
        f"\n4. Бутстрап: {BOOTSTRAP_RUNS} перемешанных историй. Между границами "
        f"лежит {HIGH - LOW}% историй."
    )
    print("   Хорошо, если нижняя граница выше нуля и доля историй ≤ 0 мала.")
    boot = [bootstrap_row("портфель, 4 рынка", portfolio.trades)]
    boot += [
        bootstrap_row(f"{name} в одиночку", report.trades)
        for name, report in alone.items()
    ]
    print(pl.DataFrame(boot))


if __name__ == "__main__":
    main()
