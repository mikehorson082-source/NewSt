"""Сетка издержек на случайных сигналах: работает ли сама сетка.

Те же случайные сигналы, что в backtest.checks.portfolio, прогоняются
через весь портфель при проскальзывании 0, 1, 2, 3, 5 шагов и сборе
×0, ×1, ×2 — 15 полных прогонов, плюс 2 прогона с влиянием на рынок
η = 0.5 и 1 при 1 шаге и сборе ×1 (стресс: η не оценён, задан) —
несколько минут. Стратегии в сигналах нет:
без издержек результат должен быть около нуля, с любыми издержками —
убыток, и он должен расти с издержками. На шаге 4 тот же скрипт покажет,
сколько шагов проскальзывания выдерживает настоящая стратегия.
Рисует src/backtest/checks/reports/cost_grid_random.png.
Запуск: uv run python -m backtest.checks.costs
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from backtest.charts import save_cost_grid_chart
from backtest.checks.portfolio import CAPITAL, random_signals
from backtest.portfolio import load_markets
from backtest.report import format_value, market_timeline
from backtest.sensitivity import FEE_MULTIPLIERS, IMPACTS, break_even, cost_grid

CHART = Path(__file__).resolve().parent / "reports" / "cost_grid_random.png"


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets()
    timeline = market_timeline(data.bars for data in markets.values())
    print("17 полных прогонов портфеля, несколько минут...")
    grid = cost_grid(random_signals(markets), markets, timeline, CAPITAL)

    print("\nСлучайные сигналы: результат при разных издержках.")
    print(
        grid.select(
            pl.col("slippage_ticks").alias("шагов"),
            pl.col("fee_multiplier").alias("сбор ×"),
            pl.col("trades").alias("сделок"),
            *[
                pl.col(key)
                .map_elements(
                    lambda v, k=key: format_value(k, v), return_dtype=pl.String
                )
                .alias(label)
                for key, label in (
                    ("pnl_before_costs", "до издержек, ₽"),
                    ("costs", "издержки, ₽"),
                    ("net_pnl", "после издержек, ₽"),
                    ("total_return", "доходность"),
                    ("sharpe", "Sharpe"),
                    ("expectancy_r", "expectancy, R"),
                )
            ],
        )
    )
    for fee in FEE_MULTIPLIERS:
        point = break_even(grid, fee)
        text = f"{point:.2f} шага" if point is not None else "нет (знак не меняется)"
        print(f"Безубыточное проскальзывание при сборе ×{fee:g}: {text}")

    stress = cost_grid(
        random_signals(markets),
        markets,
        timeline,
        CAPITAL,
        slippage_ticks=(1.0,),
        fee_multipliers=(1.0,),
        impacts=[impact for impact in IMPACTS if impact > 0],
    )
    base = grid.filter(
        (pl.col("slippage_ticks") == 1) & (pl.col("fee_multiplier") == 1)
    )
    print(
        "\nВлияние на рынок (1 шаг, сбор ×1): η × σ × цена × √(контракты ÷ объём свечи)."
    )
    print(
        pl.concat([base, stress], how="diagonal_relaxed").select(
            pl.col("impact").alias("η"),
            pl.col("trades").alias("сделок"),
            pl.col("impact_costs").round(0).alias("влияние, ₽"),
            pl.col("costs").round(0).alias("все издержки, ₽"),
            pl.col("expectancy_r").round(3).alias("expectancy, R"),
            pl.col("sharpe").round(2).alias("Sharpe"),
        )
    )

    path = save_cost_grid_chart(
        grid, CHART, "Случайные сигналы: expectancy после издержек"
    )
    print(f"\nГрафик: {path}")


if __name__ == "__main__":
    main()
