"""Паспорта четырёх рынков плана st01 на реальных данных.

Печатает шаг цены и коэффициент ГО из снимка биржи, а по годам — рубли
за пункт и ГО одного контракта. Ничего не записывает.
Запуск: uv run python -m backtest.checks.markets
"""

from __future__ import annotations

import polars as pl

from backtest.markets import MARKETS, load_market


def main() -> None:
    pl.Config.set_tbl_rows(20)
    for name in MARKETS:
        params, bars = load_market(name)
        market = params.market
        print(
            f"\n{name} ({market.asset_code}, {market.quote_currency}): "
            f"шаг {params.tick_size}, коэффициент ГО {params.margin_factor:.2f} "
            f"по {params.snapshot_contract} на {params.snapshot_date}, "
            f"сбор {market.exchange_fee_rate * 100:.5f}%"
        )
        by_year = (
            bars.with_columns(
                (pl.col("close") * pl.col("point_value")).alias("contract_value"),
            )
            .with_columns(
                (pl.col("contract_value") * pl.col("margin_rate")).alias("margin_rub")
            )
            .group_by(pl.col("timestamp").dt.year().alias("год"))
            .agg(
                pl.col("point_value").min().round(1).alias("₽ за пункт мин"),
                pl.col("point_value").max().round(1).alias("₽ за пункт макс"),
                (pl.col("margin_rate").min() * 100).round(1).alias("ГО % мин"),
                (pl.col("margin_rate").max() * 100).round(1).alias("ГО % макс"),
                pl.col("margin_rub").median().round(0).alias("ГО ₽ медиана"),
            )
            .sort("год")
        )
        print(by_year)


if __name__ == "__main__":
    main()
