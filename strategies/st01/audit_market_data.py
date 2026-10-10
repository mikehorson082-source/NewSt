"""Аудит минутных данных фьючерсов для Multi-asset Advisor.

Скрипт только читает data/raw и сохраняет результаты аудита в st01.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "futures"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "st01"
DEFAULT_ASSETS = ("mix", "mix_mini", "cnyrub", "brent", "gold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка покрытия и качества минутных данных фьючерсов."
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=list(DEFAULT_ASSETS),
        help="Каталоги рынков в data/raw/futures.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Каталог с минутными данными.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог для CSV-отчетов.",
    )
    return parser.parse_args()


def audit_contract(path: Path, asset: str) -> dict[str, object]:
    try:
        df = pl.read_parquet(
            path,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
    except Exception as exc:
        return {
            "asset": asset,
            "contract": path.stem,
            "file": str(path),
            "status": "read_error",
            "error": str(exc),
        }

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        return {
            "asset": asset,
            "contract": path.stem,
            "file": str(path),
            "status": "missing_columns",
            "error": ",".join(sorted(missing)),
        }

    df = df.sort("timestamp")
    timestamps = df.get_column("timestamp")
    price_columns = ["open", "high", "low", "close"]

    duplicate_rows = int(timestamps.is_duplicated().sum())
    null_timestamps = int(timestamps.null_count())
    null_prices = int(sum(df.get_column(col).null_count() for col in price_columns))
    nonpositive_prices = int(
        df.select(
            pl.any_horizontal([pl.col(col) <= 0 for col in price_columns]).sum()
        ).item()
    )
    invalid_ohlc = int(
        df.select(
            (
                (pl.col("high") < pl.max_horizontal("open", "close", "low"))
                | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
            ).sum()
        ).item()
    )
    zero_volume_rows = int(df.select((pl.col("volume") <= 0).sum()).item())

    # Разрывы считаются только внутри одной календарной даты. Регулярные клиринговые
    # паузы тоже попадут в этот показатель, поэтому это диагностика, а не ошибка.
    gap_stats = (
        df.select("timestamp")
        .with_columns(
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("timestamp").diff().dt.total_minutes().alias("gap_minutes"),
        )
        .filter(
            (pl.col("date") == pl.col("date").shift(1))
            & (pl.col("gap_minutes") > 1)
        )
    )

    first_timestamp = timestamps.min()
    last_timestamp = timestamps.max()

    return {
        "asset": asset,
        "contract": path.stem,
        "file": str(path.relative_to(PROJECT_ROOT)),
        "status": "ok",
        "rows": df.height,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "duplicate_rows": duplicate_rows,
        "null_timestamps": null_timestamps,
        "null_prices": null_prices,
        "nonpositive_prices": nonpositive_prices,
        "invalid_ohlc_rows": invalid_ohlc,
        "zero_volume_rows": zero_volume_rows,
        "intraday_gaps_gt_1m": gap_stats.height,
        "max_intraday_gap_minutes": gap_stats.get_column("gap_minutes").max()
        if gap_stats.height
        else 0,
        "error": None,
    }


def build_summary(details: pl.DataFrame) -> pl.DataFrame:
    valid = details.filter(pl.col("status") == "ok")
    if valid.is_empty():
        return pl.DataFrame()

    return (
        valid.group_by("asset")
        .agg(
            pl.len().alias("contracts"),
            pl.col("rows").sum().alias("rows"),
            pl.col("first_timestamp").min().alias("history_start"),
            pl.col("last_timestamp").max().alias("history_end"),
            pl.col("duplicate_rows").sum().alias("duplicate_rows"),
            pl.col("null_prices").sum().alias("null_prices"),
            pl.col("nonpositive_prices").sum().alias("nonpositive_prices"),
            pl.col("invalid_ohlc_rows").sum().alias("invalid_ohlc_rows"),
            pl.col("zero_volume_rows").sum().alias("zero_volume_rows"),
            pl.col("intraday_gaps_gt_1m").sum().alias("intraday_gaps_gt_1m"),
        )
        .sort("asset")
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for asset in args.assets:
        asset_dir = data_dir / asset
        files = sorted(asset_dir.glob("*.parquet"))
        if not files:
            records.append(
                {
                    "asset": asset,
                    "contract": None,
                    "file": str(asset_dir),
                    "status": "no_files",
                    "error": "Parquet-файлы не найдены",
                }
            )
            continue

        print(f"{asset}: проверка {len(files)} контрактов")
        records.extend(audit_contract(path, asset) for path in files)

    details = pl.DataFrame(records, infer_schema_length=None)
    summary = build_summary(details)

    details_path = output_dir / "market_data_audit_contracts.csv"
    summary_path = output_dir / "market_data_audit_summary.csv"
    details.write_csv(details_path)
    summary.write_csv(summary_path)

    print("\nСводка:")
    print(summary)
    print(f"\nДетали: {details_path}")
    print(f"Сводка:  {summary_path}")


if __name__ == "__main__":
    main()
