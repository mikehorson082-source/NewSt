"""Строит непрерывные 15-минутные ряды фьючерсов для стратегии st01.

Правило роллирования:
1. В каждый торговый день решение принимается по объему предыдущего торгового дня.
2. Сравниваются текущий контракт и ближайший следующий контракт.
3. Переход выполняется, когда вчерашний объем следующего контракта стал больше.
4. Возврат к более ранней экспирации запрещен.

Скрипт только читает data/raw. Результаты сохраняются в st01/data/continuous_15m.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import date
from pathlib import Path
import re
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "futures"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "st01" / "data" / "continuous_15m"
DEFAULT_FROM_DATE = date(2022, 5, 1)

ASSET_DIRECTORIES = {
    "MXI": "mix_mini",
    "CNYRUB": "cnyrub",
    "BRENT": "brent",
    "GOLD": "gold",
}

MONTH_NUMBER = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Склейка контрактов MXI, CNY/RUB, Brent и Gold в 15-минутные ряды."
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=sorted(ASSET_DIRECTORIES),
        default=list(ASSET_DIRECTORIES),
        help="Рынки для обработки.",
    )
    parser.add_argument(
        "--from-date",
        type=date.fromisoformat,
        default=DEFAULT_FROM_DATE,
        help="Начало ряда в формате YYYY-MM-DD.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог производных данных.",
    )
    return parser.parse_args()


def contract_maturity(contract: str) -> tuple[int, int]:
    """Возвращает (год, месяц) из биржевого кода вида MMZ5, CRH6, BRF4."""
    match = re.fullmatch(r".*?([FGHJKMNQUVXZ])(\d)", contract.upper())
    if match is None:
        raise ValueError(f"Не удалось определить экспирацию контракта {contract}")
    month_code, year_digit = match.groups()
    return 2020 + int(year_digit), MONTH_NUMBER[month_code]


def aggregate_daily(path: Path, asset: str) -> pl.DataFrame:
    """Дневная ликвидность и граничные цены одного контракта."""
    return (
        pl.read_parquet(
            path,
            columns=["timestamp", "open", "close", "volume"],
        )
        .sort("timestamp")
        .with_columns(pl.col("timestamp").dt.date().alias("calendar_date"))
        .group_by("calendar_date", maintain_order=True)
        .agg(
            pl.col("open").first().alias("first_open"),
            pl.col("close").last().alias("last_close"),
            pl.col("volume").sum().alias("daily_volume"),
            pl.len().alias("trading_minutes"),
        )
        .with_columns(
            pl.lit(asset).alias("asset"),
            pl.lit(path.stem).alias("contract"),
        )
        .select(
            "asset",
            "calendar_date",
            "contract",
            "first_open",
            "last_close",
            "daily_volume",
            "trading_minutes",
        )
    )


def assign_trading_dates(daily: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Относит выходные сессии к следующему наблюдаемому буднему дню."""
    calendar_dates = sorted(daily.get_column("calendar_date").unique().to_list())
    weekday_dates = [day for day in calendar_dates if day.weekday() < 5]
    if not weekday_dates:
        raise ValueError("Не удалось построить торговый календарь")

    mapping: list[dict[str, date]] = []
    for calendar_date in calendar_dates:
        if calendar_date.weekday() < 5:
            trading_date = calendar_date
        else:
            position = bisect_left(weekday_dates, calendar_date)
            if position >= len(weekday_dates):
                # Незавершенную выходную сессию в конце набора нельзя привязать
                # к еще отсутствующему следующему торговому дню.
                continue
            trading_date = weekday_dates[position]
        mapping.append(
            {"calendar_date": calendar_date, "date": trading_date}
        )

    date_map = pl.DataFrame(mapping)
    combined = (
        daily.join(date_map, on="calendar_date", how="inner")
        .sort("calendar_date")
        .group_by("asset", "contract", "date", maintain_order=True)
        .agg(
            pl.col("first_open").first().alias("first_open"),
            pl.col("last_close").last().alias("last_close"),
            pl.col("daily_volume").sum().alias("daily_volume"),
            pl.col("trading_minutes").sum().alias("trading_minutes"),
        )
        .sort("date", "contract")
    )
    return combined, date_map


def load_daily_market(
    asset: str,
    directory: Path,
) -> tuple[pl.DataFrame, dict[str, Path], pl.DataFrame]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"В {directory} нет Parquet-файлов")

    paths = {path.stem: path for path in files}
    calendar_daily = pl.concat(
        [aggregate_daily(path, asset) for path in files],
        how="vertical_relaxed",
    ).sort("calendar_date", "contract")
    daily, date_map = assign_trading_dates(calendar_daily)
    return daily, paths, date_map


def rows_by_date(daily: pl.DataFrame) -> dict[date, dict[str, dict[str, Any]]]:
    result: dict[date, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in daily.iter_rows(named=True):
        result[row["date"]][row["contract"]] = row
    return dict(result)


def choose_initial_contract(rows: dict[str, dict[str, Any]]) -> str:
    return max(
        rows,
        key=lambda contract: (
            float(rows[contract]["daily_volume"] or 0),
            -contract_maturity(contract)[0],
            -contract_maturity(contract)[1],
        ),
    )


def nearest_later_contract(
    current: str,
    known_yesterday: set[str],
    available_today: set[str],
) -> str | None:
    current_maturity = contract_maturity(current)
    candidates = [
        contract
        for contract in known_yesterday.intersection(available_today)
        if contract_maturity(contract) > current_maturity
    ]
    return min(candidates, key=contract_maturity) if candidates else None


def build_selection(
    asset: str,
    daily: pl.DataFrame,
    from_date: date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Строит выбранный контракт на каждый день и журнал переходов."""
    all_rows = rows_by_date(daily)
    dates = sorted(day for day in all_rows if day >= from_date)
    if not dates:
        raise ValueError(f"{asset}: нет данных начиная с {from_date}")

    current = choose_initial_contract(all_rows[dates[0]])
    current_adjustment = 0.0
    selection: list[dict[str, Any]] = []
    rolls: list[dict[str, Any]] = []

    selection.append(
        {
            "asset": asset,
            "date": dates[0],
            "contract": current,
            "selection_reason": "initial_max_volume",
            "decision_date": None,
            "daily_volume": all_rows[dates[0]][current]["daily_volume"],
            "price_adjustment": current_adjustment,
            "is_roll_day": False,
        }
    )

    for index in range(1, len(dates)):
        today = dates[index]
        yesterday = dates[index - 1]
        today_rows = all_rows[today]
        yesterday_rows = all_rows[yesterday]
        reason = "keep_current"
        next_contract = nearest_later_contract(
            current,
            set(yesterday_rows),
            set(today_rows),
        )

        roll_to: str | None = None
        if current not in today_rows:
            later_today = [
                contract
                for contract in today_rows
                if contract_maturity(contract) > contract_maturity(current)
            ]
            if not later_today:
                raise ValueError(
                    f"{asset} {today}: текущий {current} отсутствует, "
                    "следующий контракт не найден"
                )
            roll_to = min(later_today, key=contract_maturity)
            reason = "current_unavailable"
        elif next_contract is not None:
            current_volume = float(
                yesterday_rows.get(current, {}).get("daily_volume") or 0
            )
            next_volume = float(yesterday_rows[next_contract]["daily_volume"] or 0)
            if next_volume > current_volume:
                roll_to = next_contract
                reason = "next_volume_exceeded_current"

        if roll_to is not None:
            old_row = yesterday_rows.get(current)
            new_row = yesterday_rows.get(roll_to)
            if old_row is None or new_row is None:
                raise ValueError(
                    f"{asset} {today}: для ценовой поправки нет закрытия "
                    f"{current} или {roll_to} за {yesterday}"
                )

            old_close = float(old_row["last_close"])
            new_close = float(new_row["last_close"])
            adjustment_delta = old_close - new_close
            current_adjustment += adjustment_delta

            rolls.append(
                {
                    "asset": asset,
                    "roll_date": today,
                    "decision_date": yesterday,
                    "from_contract": current,
                    "to_contract": roll_to,
                    "old_volume": old_row["daily_volume"],
                    "new_volume": new_row["daily_volume"],
                    "old_close": old_close,
                    "new_close": new_close,
                    "adjustment_delta": adjustment_delta,
                    "cumulative_adjustment": current_adjustment,
                    "reason": reason,
                }
            )
            current = roll_to

        if current not in today_rows:
            raise ValueError(f"{asset} {today}: выбранный {current} не торговался")

        selection.append(
            {
                "asset": asset,
                "date": today,
                "contract": current,
                "selection_reason": reason,
                "decision_date": yesterday,
                "daily_volume": today_rows[current]["daily_volume"],
                "price_adjustment": current_adjustment,
                "is_roll_day": roll_to is not None,
            }
        )

    selection_df = pl.DataFrame(selection, infer_schema_length=None).sort("date")
    rolls_df = pl.DataFrame(rolls, infer_schema_length=None)
    return selection_df, rolls_df


def load_selected_minutes(
    asset: str,
    paths: dict[str, Path],
    selection: pl.DataFrame,
    date_map: pl.DataFrame,
) -> pl.DataFrame:
    selected_contracts = selection.get_column("contract").unique().to_list()
    frames: list[pl.DataFrame] = []

    for contract in selected_contracts:
        path = paths[contract]
        frame = (
            pl.read_parquet(
                path,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "value",
                ],
            )
            .with_columns(
                pl.lit(asset).alias("asset"),
                pl.lit(contract).alias("contract"),
                pl.col("timestamp").dt.date().alias("calendar_date"),
            )
        )
        frames.append(frame)

    selected_keys = selection.select(
        "asset",
        "date",
        "contract",
        "price_adjustment",
        "is_roll_day",
    )
    return (
        pl.concat(frames, how="vertical_relaxed")
        .join(date_map, on="calendar_date", how="inner")
        .join(selected_keys, on=["asset", "date", "contract"], how="inner")
        .sort("timestamp")
        .with_columns(
            (pl.col("open") + pl.col("price_adjustment")).alias("open_adjusted"),
            (pl.col("high") + pl.col("price_adjustment")).alias("high_adjusted"),
            (pl.col("low") + pl.col("price_adjustment")).alias("low_adjusted"),
            (pl.col("close") + pl.col("price_adjustment")).alias("close_adjusted"),
        )
    )


def resample_15_minutes(minutes: pl.DataFrame) -> pl.DataFrame:
    """Агрегирует только фактические сделки; пустые интервалы не создаются."""
    return (
        minutes.with_columns(
            pl.col("timestamp").dt.truncate("15m").alias("timestamp_15m")
        )
        .sort("timestamp")
        .group_by(
            ["asset", "contract", "date", "timestamp_15m"],
            maintain_order=True,
        )
        .agg(
            pl.col("open").first().alias("open_raw"),
            pl.col("high").max().alias("high_raw"),
            pl.col("low").min().alias("low_raw"),
            pl.col("close").last().alias("close_raw"),
            pl.col("open_adjusted").first().alias("open"),
            pl.col("high_adjusted").max().alias("high"),
            pl.col("low_adjusted").min().alias("low"),
            pl.col("close_adjusted").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("value").sum().alias("value"),
            pl.len().alias("source_minutes"),
            pl.col("price_adjustment").first().alias("price_adjustment"),
            pl.col("is_roll_day").first().alias("is_roll_day"),
        )
        .rename({"timestamp_15m": "timestamp"})
        .sort("asset", "timestamp")
        .with_columns(
            (
                pl.col("is_roll_day")
                & (
                    pl.col("contract")
                    != pl.col("contract").shift(1).over("asset")
                ).fill_null(False)
            ).alias("is_first_bar_after_roll")
        )
        .select(
            "asset",
            "timestamp",
            "date",
            "contract",
            "open",
            "high",
            "low",
            "close",
            "open_raw",
            "high_raw",
            "low_raw",
            "close_raw",
            "volume",
            "value",
            "source_minutes",
            "price_adjustment",
            "is_roll_day",
            "is_first_bar_after_roll",
        )
    )


def validate_result(asset: str, bars: pl.DataFrame, selection: pl.DataFrame) -> None:
    if bars.is_empty():
        raise ValueError(f"{asset}: итоговый ряд пуст")

    duplicate_bars = bars.select(
        pl.struct("asset", "timestamp").is_duplicated().sum()
    ).item()
    if duplicate_bars:
        raise ValueError(f"{asset}: найдено {duplicate_bars} дублей 15-минутных баров")

    invalid_ohlc = bars.select(
        (
            (pl.col("high") < pl.max_horizontal("open", "close", "low"))
            | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
        ).sum()
    ).item()
    if invalid_ohlc:
        raise ValueError(f"{asset}: найдено {invalid_ohlc} некорректных OHLC")

    maturities = [
        contract_maturity(contract)
        for contract in selection.get_column("contract").to_list()
    ]
    if any(right < left for left, right in zip(maturities, maturities[1:])):
        raise ValueError(f"{asset}: обнаружен возврат к более ранней экспирации")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_bars: list[pl.DataFrame] = []
    all_selections: list[pl.DataFrame] = []
    all_rolls: list[pl.DataFrame] = []

    for asset in args.assets:
        print(f"{asset}: чтение контрактов")
        daily, paths, date_map = load_daily_market(
            asset,
            RAW_DIR / ASSET_DIRECTORIES[asset],
        )
        selection, rolls = build_selection(asset, daily, args.from_date)
        minutes = load_selected_minutes(asset, paths, selection, date_map)
        bars = resample_15_minutes(minutes)
        validate_result(asset, bars, selection)

        bars.write_parquet(
            output_dir / f"{asset.lower()}_15m.parquet",
            compression="zstd",
        )
        all_bars.append(bars)
        all_selections.append(selection)
        if not rolls.is_empty():
            all_rolls.append(rolls)

        print(
            f"{asset}: {bars.height:,} баров, "
            f"{selection.height:,} дней, {rolls.height} переходов"
        )

    combined_bars = pl.concat(all_bars, how="vertical_relaxed").sort(
        "timestamp", "asset"
    )
    combined_selection = pl.concat(all_selections, how="vertical_relaxed").sort(
        "date", "asset"
    )
    combined_rolls = (
        pl.concat(all_rolls, how="vertical_relaxed").sort("roll_date", "asset")
        if all_rolls
        else pl.DataFrame()
    )

    combined_bars.write_parquet(
        output_dir / "continuous_futures_15m.parquet",
        compression="zstd",
    )
    combined_selection.write_parquet(
        output_dir / "daily_contract_selection.parquet",
        compression="zstd",
    )
    combined_rolls.write_csv(output_dir / "roll_schedule.csv")

    print(f"\nРезультаты сохранены в {output_dir}")
    print(f"Всего 15-минутных баров: {combined_bars.height:,}")
    print(f"Всего переходов: {combined_rolls.height}")


if __name__ == "__main__":
    main()
