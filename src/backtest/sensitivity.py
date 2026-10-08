"""Сетка издержек: насколько можно ошибиться в издержках, прежде чем ответ сменится.

Те же сигналы прогоняются через весь портфель заново при другом
проскальзывании и сборе (ML4T 16 `14_cost_sensitivity`). Именно заново: при
больших издержках размер позиции меньше (риск контракта включает издержки),
а выбор «дешевле по издержкам» между рынками может смениться. Этим сетка
отличается от «запаса по издержкам» в отчёте, где сделки остаются те же.

Пример вывода: стратегия зарабатывает при 1 шаге и теряет при 2 — весь её
заработок висит на одном шаге проскальзывания, торговать её опасно.

Влияние на рынок η (`ContractSpec.impact`) в обычном отчёте выключено;
в сетке его можно включить как стресс: η = 0.5 и 1 (`IMPACTS`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from itertools import pairwise

import polars as pl

from backtest.portfolio import MarketData, run_portfolio
from backtest.report import build_report

SLIPPAGE_TICKS = (0.0, 1.0, 2.0, 3.0, 5.0)
FEE_MULTIPLIERS = (0.0, 1.0, 2.0)
IMPACTS = (0.0, 0.5, 1.0)
METRICS = (
    "trades",
    "end_capital",
    "total_return",
    "sharpe",
    "profit_factor",
    "expectancy_r",
    "pnl_before_costs",
    "costs",
    "net_pnl",
)


def with_costs(
    markets: dict[str, MarketData],
    slippage_ticks: float,
    fee_multiplier: float,
    impact: float = 0.0,
) -> dict[str, MarketData]:
    """Те же рынки с другим проскальзыванием (в шагах), сбором (× к тарифу)
    и влиянием на рынок η."""
    return {
        name: replace(
            data,
            spec=replace(
                data.spec,
                slippage_ticks=slippage_ticks,
                exchange_fee_rate=data.spec.exchange_fee_rate * fee_multiplier,
                broker_fee_rate=data.spec.broker_fee_rate * fee_multiplier,
                impact=impact,
            ),
        )
        for name, data in markets.items()
    }


def cost_grid(
    signals: pl.DataFrame,
    markets: dict[str, MarketData],
    timeline: pl.Series,
    capital: float = 1_000_000.0,
    slippage_ticks: Iterable[float] = SLIPPAGE_TICKS,
    fee_multipliers: Iterable[float] = FEE_MULTIPLIERS,
    impacts: Iterable[float] = (0.0,),
) -> pl.DataFrame:
    """По строке на каждое сочетание (η, сбор, проскальзывание): метрики
    полного прогона."""
    rows = []
    for impact in impacts:
        for fee in fee_multipliers:
            for ticks in slippage_ticks:
                costs = with_costs(markets, ticks, fee, impact)
                result = run_portfolio(signals, costs, capital)
                summary = build_report(result, timeline).summary
                rows.append(
                    {
                        "impact": float(impact),
                        "slippage_ticks": float(ticks),
                        "fee_multiplier": float(fee),
                        "impact_costs": float(
                            sum(item.trade.impact for item in result.trades)
                        ),
                        **{key: summary[key] for key in METRICS},
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def break_even(
    grid: pl.DataFrame, fee_multiplier: float = 1.0, value: str = "expectancy_r"
) -> float | None:
    """Проскальзывание в шагах, при котором результат после издержек = 0.

    По умолчанию — expectancy в R на сделку: результат в рублях упирается в
    «дно», когда капитал кончается, а R на сделку от капитала не зависит.
    Линейная интерполяция между точками сетки при заданном сборе. None —
    результат не меняет знак: убыток даже без проскальзывания или прибыль
    на всей сетке.
    """
    line = grid.filter(pl.col("fee_multiplier") == fee_multiplier)
    if "impact" in line.columns:
        line = line.filter(pl.col("impact") == 0.0)
    line = line.sort("slippage_ticks")
    points = list(zip(line["slippage_ticks"], line[value], strict=True))
    for (x0, y0), (x1, y1) in pairwise(points):
        if y0 >= 0 > y1:
            return x0 + (x1 - x0) * y0 / (y0 - y1)
    return None


# Сессии FORTS по времени свечи входа (МСК): утренняя до 10:00, основная
# 10:00–19:00, вечерняя с 19:00; субботние и воскресные — отдельно.
SESSIONS = ("утро", "день", "вечер", "выходные")
SATURDAY = 6


def session_of(timestamp: pl.Expr) -> pl.Expr:
    """Сессия свечи по её времени."""
    return (
        pl.when(timestamp.dt.weekday() >= SATURDAY)
        .then(pl.lit("выходные"))
        .when(timestamp.dt.hour() < 10)
        .then(pl.lit("утро"))
        .when(timestamp.dt.hour() < 19)
        .then(pl.lit("день"))
        .otherwise(pl.lit("вечер"))
    )


def by_session(trades: pl.DataFrame) -> pl.DataFrame:
    """Сделки по сессии входа: хуже ли вечерние и выходные (ML4T 18.3).

    Спред в бэктесте везде 1 шаг — измерить его без данных стакана нельзя.
    Таблица показывает то, что видно и без спреда: результат в R, долю
    издержек, долю сделки в объёме свечи входа и гэпы через стоп. Сделки,
    закрытые концом данных, не входят. `trades` — `PortfolioResult.trade_log()`.
    """
    closed = trades.filter(pl.col("exit_reason") != "end")
    table = (
        closed.with_columns(session_of(pl.col("entry_time")).alias("session"))
        .group_by("session")
        .agg(
            pl.len().alias("trades"),
            (pl.col("net_pnl") > 0).mean().alias("win_rate"),
            pl.col("r_multiple").mean().alias("expectancy_r"),
            (pl.col("costs") / pl.col("risk")).mean().alias("costs_per_trade_r"),
            pl.col("volume_share").median().alias("volume_share_median"),
            (pl.col("volume_share") > 0.1).sum().alias("over_10pct"),
            (pl.col("mae_r") < -1.5).sum().alias("gaps_over_1_5r"),
        )
    )
    order = pl.DataFrame({"session": list(SESSIONS)}).with_row_index("_order")
    return table.join(order, on="session").sort("_order").drop("_order")
