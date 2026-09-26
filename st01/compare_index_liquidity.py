"""Сравнение ликвидности MIX, MXI и IMOEXF.

Скрипт только читает data/raw. Для MIX и MXI на каждый день выбирается
контракт с максимальным биржевым оборотом этого дня. Это диагностический
выбор для сравнения ликвидности, а не правило роллирования в бэктесте.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "futures"
OUTPUT_DIR = PROJECT_ROOT / "st01"
DEFAULT_FROM_DATE = date(2022, 5, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнение дневной ликвидности MIX, MXI и IMOEXF."
    )
    parser.add_argument(
        "--from-date",
        type=date.fromisoformat,
        default=DEFAULT_FROM_DATE,
        help="Начало общего периода в формате YYYY-MM-DD.",
    )
    return parser.parse_args()


def aggregate_contract(path: Path, asset: str) -> pl.DataFrame:
    df = pl.read_parquet(
        path,
        columns=["timestamp", "volume", "value"],
    )
    return (
        df.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.len().alias("trading_minutes"),
            pl.col("volume").sum().alias("contracts_traded"),
            pl.col("value").sum().alias("turnover_rub"),
        )
        .with_columns(
            pl.lit(asset).alias("asset"),
            pl.lit(path.stem).alias("contract"),
        )
        .select(
            "asset",
            "date",
            "contract",
            "trading_minutes",
            "contracts_traded",
            "turnover_rub",
        )
    )


def dominant_daily_contract(asset: str, directory: Path) -> pl.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"В {directory} нет Parquet-файлов")

    daily = pl.concat(
        [aggregate_contract(path, asset) for path in files],
        how="vertical_relaxed",
    )

    # Максимальный оборот лучше отражает основной торгуемый контракт, чем число
    # строк: одна минутная свеча может содержать как одну, так и тысячи сделок.
    return (
        daily.sort(
            ["date", "turnover_rub", "contracts_traded"],
            descending=[False, True, True],
        )
        .unique(subset="date", keep="first", maintain_order=True)
        .sort("date")
    )


def perpetual_daily(asset: str, path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return aggregate_contract(path, asset).sort("date")


def percentile(column: str, probability: float, alias: str) -> pl.Expr:
    return pl.col(column).quantile(probability, interpolation="nearest").alias(alias)


def build_summary(daily: pl.DataFrame, observed_dates: pl.DataFrame) -> pl.DataFrame:
    observed_count = observed_dates.height
    return (
        daily.group_by("asset")
        .agg(
            pl.col("date").min().alias("history_start"),
            pl.col("date").max().alias("history_end"),
            pl.len().alias("trading_days"),
            pl.col("trading_minutes").median().alias("median_minutes_day"),
            percentile("trading_minutes", 0.10, "p10_minutes_day"),
            pl.col("contracts_traded").median().alias("median_contracts_day"),
            percentile("contracts_traded", 0.10, "p10_contracts_day"),
            pl.col("turnover_rub").median().alias("median_turnover_rub_day"),
            percentile("turnover_rub", 0.10, "p10_turnover_rub_day"),
            (pl.col("trading_minutes") < 100).sum().alias("days_under_100_minutes"),
            (pl.col("trading_minutes") < 500).sum().alias("days_under_500_minutes"),
            pl.col("contract").n_unique().alias("dominant_contracts"),
        )
        .with_columns(
            (pl.lit(observed_count) - pl.col("trading_days")).alias(
                "missing_observed_market_days"
            )
        )
        .sort("asset")
    )


def build_yearly_summary(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("asset", "year")
        .agg(
            pl.len().alias("trading_days"),
            pl.col("trading_minutes").median().alias("median_minutes_day"),
            pl.col("contracts_traded").median().alias("median_contracts_day"),
            pl.col("turnover_rub").median().alias("median_turnover_rub_day"),
        )
        .sort("year", "asset")
    )


def main() -> None:
    args = parse_args()

    datasets = [
        dominant_daily_contract("MIX", RAW_DIR / "mix"),
        dominant_daily_contract("MXI", RAW_DIR / "mix_mini"),
        perpetual_daily("IMOEXF", RAW_DIR / "perpetual" / "IMOEXF.parquet"),
    ]
    daily = (
        pl.concat(datasets, how="vertical_relaxed")
        .filter(pl.col("date") >= args.from_date)
        .sort("date", "asset")
    )

    # Объединение дат трех рынков используется как приближенный календарь:
    # если хотя бы один индекс торговался, отсутствие другого считается пропуском.
    observed_dates = daily.select("date").unique().sort("date")
    summary = build_summary(daily, observed_dates)
    yearly = build_yearly_summary(daily)

    daily_path = OUTPUT_DIR / "index_liquidity_daily.csv"
    summary_path = OUTPUT_DIR / "index_liquidity_summary.csv"
    yearly_path = OUTPUT_DIR / "index_liquidity_yearly.csv"

    daily.write_csv(daily_path)
    summary.write_csv(summary_path)
    yearly.write_csv(yearly_path)

    print("\nОбщая сводка:")
    print(summary)
    print("\nПо годам:")
    print(yearly)
    print(f"\nДневные данные: {daily_path}")
    print(f"Общая сводка: {summary_path}")
    print(f"Сводка по годам: {yearly_path}")


if __name__ == "__main__":
    main()
