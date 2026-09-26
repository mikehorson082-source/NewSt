"""Строит непрерывные 15-минутные ряды фьючерсов (общий слой данных).

Правило роллирования:
1. Решение на календарный день (MSK) принимается по завершенным дням с данными
   строго до этого дня; сам день решения не используется.
2. Сравниваются текущий контракт и ближайший следующий контракт.
3. Переход выполняется, когда средний дневной объем следующего контракта за
   окно последних завершенных дней с данными (по умолчанию 3) стал больше
   среднего объема текущего. Усреднение убирает ложные однодневные пересечения,
   особенно в коротких сессиях выходного дня.
4. Если сигнал по объему не возник, переход с начала LSTTRADE (запас 0 дней).
   Старый контракт в последний день обращения не выбирается. Это ограничение
   дневного исследовательского ряда, а не модель исполнения переноса позиции.
5. Поправка берется по общей минуте вчерашних торгов. Возврат запрещен.
6. NAV (ETF Trick) — доходность самофинансируемой позиции: внутри контракта
   close/prev_close, на первом баре нового контракта close/open, на переходе
   вычитается roll_cost_bps. NAV служит для признаков и таргетов, реальные
   цены — для исполнения. Разрыв базиса в NAV не попадает.

date — календарная дата MSK, НЕ биржевая дата торговой сессии.
open/high/low/close — реальные цены выбранного контракта; *_adjusted —
аддитивный ряд для изменений в пунктах. log_return считается по реальным ценам
и отсутствует на первом баре и на переходах. Это не доходность капитала.
Последний день raw по умолчанию исключается как потенциально незавершенный.
Старт рынка может быть позже --from-date: структурный разрыв истории
(continuous_from в реестре) нельзя перешагнуть склейкой, ряд начинается
после него.

Скрипт только читает data/raw и data/reference/futures_contracts.parquet.
Сетевых запросов нет. Результаты сохраняются в data/continuous_15m.
Список рынков берётся из реестра src/core/instruments.py. По умолчанию
(all) склеиваются все готовые рынки; --assets plan — четыре рынка плана
(MXI/CNYRUB/BRENT/GOLD), их ярлыки — псевдонимы имён реестра.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from core.instruments import ASSETS, Cadence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "futures"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "continuous_15m"
DEFAULT_FROM_DATE = date(2022, 5, 1)

PLAN_ASSETS = ["MXI", "CNYRUB", "BRENT", "GOLD"]
DEFAULT_ASSETS = ["all"]
ASSET_ALIASES = {
    "MXI": "mix_mini",
    "CNYRUB": "cnyrub",
    "BRENT": "brent",
    "GOLD": "gold",
}


def select_requested_assets(assets: list[str]) -> tuple[list[str], bool]:
    """Разворачивает all/plan: возвращает (список рынков, признак all).

    По умолчанию запускаются все готовые рынки реестра (all). Значение plan
    оставляет только четыре рынка плана; явный список имён/ярлыков — как есть.
    """
    if len(assets) != len(set(assets)):
        raise ValueError("Повторные --assets запрещены")
    if ({"all", "plan"} & set(assets)) and len(assets) > 1:
        raise ValueError("all и plan указываются отдельно, без других рынков")
    if assets == ["all"]:
        registry = sorted(
            name for name, spec in ASSETS.items() if spec.cadence != Cadence.PERPETUAL
        )
        return registry, True
    if assets == ["plan"]:
        return list(PLAN_ASSETS), False
    return list(assets), False


def registry_name(asset: str) -> str:
    """Имя актива в реестре src/core/instruments.py по ярлыку или имени."""
    name = ASSET_ALIASES.get(asset, asset)
    if name not in ASSETS:
        available = ", ".join(available_assets())
        raise ValueError(f"Неизвестный актив {asset}. Доступные: {available}")
    if ASSETS[name].cadence == Cadence.PERPETUAL:
        raise ValueError(f"{asset}: вечный фьючерс не склеивается")
    return name


def effective_start_date(asset: str, from_date: date) -> date:
    """Старт ряда не раньше structural-границы реестра.

    Если у рынка была пауза торгов, контракты до и после нее не имеют общих
    минут, и склейка невозможна. continuous_from задает первый день, с
    которого ряд строится; --from-date может сдвинуть старт только позже.
    """
    continuous_from = ASSETS[asset].continuous_from
    if continuous_from is not None and continuous_from > from_date:
        return continuous_from
    return from_date


def available_assets() -> list[str]:
    """Ярлыки плана и все имена реестра, кроме вечных фьючерсов."""
    registry = [
        name for name, spec in ASSETS.items() if spec.cadence != Cadence.PERPETUAL
    ]
    return sorted(set(ASSET_ALIASES) | set(registry))


def select_ready_assets(
    candidates: dict[str, str],
    reference: set[str],
    raw_dir: Path = RAW_DIR,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Оставляет рынки с raw-данными и полным покрытием справочника."""
    ready: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []
    ref_upper = {t.upper() for t in reference}
    for label, name in candidates.items():
        directory = raw_dir / name
        files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
        if not files:
            skipped.append((label, "нет raw-файлов"))
            continue
        missing = [path.stem for path in files if path.stem.upper() not in ref_upper]
        if missing:
            skipped.append((label, f"нет в справочнике: {len(missing)}"))
            continue
        ready[label] = name
    return ready, skipped


def reference_tickers(path: Path) -> set[str]:
    """Тикеры контрактов из локального справочника (только чтение)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет справочника {path}. Сначала загрузите futures_contracts.parquet."
        )
    return set(pl.read_parquet(path, columns=["ticker"])["ticker"].to_list())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Склейка контрактов MXI, CNY/RUB, Brent и Gold в 15-минутные ряды."
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=DEFAULT_ASSETS,
        metavar="ASSET",
        help="По умолчанию all — все готовые рынки реестра. Явно: имена реестра "
        "(brent, rts, ...), ярлыки MXI/CNYRUB/BRENT/GOLD или plan — только "
        "четыре рынка плана.",
    )
    parser.add_argument(
        "--from-date",
        type=date.fromisoformat,
        default=DEFAULT_FROM_DATE,
        help="Начало ряда в формате YYYY-MM-DD. Для рынков со структурным "
        "разрывом истории реестр сдвигает старт позже.",
    )
    parser.add_argument(
        "--roll-days-before-last-trade",
        type=int,
        default=0,
        help="Запас календарных дней до LSTTRADE. По умолчанию 0: переход "
        "с начала последнего дня обращения, если объем не дал сигнал раньше.",
    )
    parser.add_argument(
        "--volume-window-days",
        type=int,
        default=3,
        help="Окно завершенных дней с данными для сравнения объемов (по умолчанию 3).",
    )
    parser.add_argument(
        "--roll-cost-bps",
        type=float,
        default=0.0,
        help="Издержка одного перехода в б.п. NAV (закрыть старый + открыть новый).",
    )
    parser.add_argument(
        "--completed-through",
        type=date.fromisoformat,
        help="Последняя подтвержденно полностью загруженная календарная дата MSK. "
        "Без параметра последний день каждого рынка исключается.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=REFERENCE_DIR / "futures_contracts.parquet",
        help="Локальный Parquet-справочник контрактов (только чтение).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог производных данных.",
    )
    return parser.parse_args()


def parse_contract_dates(contract: str, fields: dict[str, Any]) -> dict[str, date]:
    """LSTTRADE нужен для ролла; FRSTTRADE — необязательная проверка."""
    result: dict[str, date] = {}
    for key in ("LSTTRADE", "FRSTTRADE"):
        value = fields.get(key)
        if value in (None, ""):
            if key == "FRSTTRADE":
                continue
            raise ValueError(
                f"{contract}: в справочнике отсутствует обязательное поле LSTTRADE (last_trade_date). "
                "Дата последней свечи не заменяет дату окончания обращения. "
                f"Доступные поля: {', '.join(sorted(fields))}"
            )
        try:
            if isinstance(value, datetime):
                result[key] = value.date()
            elif isinstance(value, date):
                result[key] = value
            else:
                result[key] = date.fromisoformat(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{contract}: некорректная дата {key}={value!r}"
            ) from error
    if "FRSTTRADE" in result and result["FRSTTRADE"] >= result["LSTTRADE"]:
        raise ValueError(f"{contract}: некорректные сроки обращения: {result}")
    return result


def load_contract_metadata(
    contracts: list[str],
    path: Path,
) -> dict[str, dict[str, date]]:
    """Читает только указанный справочник, без сети и скрытых запасных источников.

    Исторические изменения расписания биржей требуют архивных снимков справочника.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет справочника {path}. Сначала загрузите futures_contracts.parquet."
        )
    reference = pl.read_parquet(path)
    missing_columns = {"ticker", "last_trade_date"} - set(reference.columns)
    if missing_columns:
        raise ValueError(
            f"{path}: отсутствуют столбцы {', '.join(sorted(missing_columns))}"
        )
    requested = set(contracts)
    duplicates = reference.group_by("ticker").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(
            f"{path}: неоднозначные записи контрактов {', '.join(sorted(duplicates['ticker'].to_list()))}"
        )
    ref_by_upper: dict[str, dict[str, Any]] = {
        str(row["ticker"]).upper(): row for row in reference.iter_rows(named=True)
    }
    missing = [c for c in sorted(requested) if c.upper() not in ref_by_upper]
    if missing:
        raise ValueError(
            f"{path}: нет контрактов {', '.join(sorted(missing))}. Дополните локальный справочник."
        )
    result: dict[str, dict[str, date]] = {}
    for contract in sorted(requested):
        row = ref_by_upper[contract.upper()]
        result[contract] = parse_contract_dates(
            contract,
            {
                "LSTTRADE": row["last_trade_date"],
                "FRSTTRADE": row.get("first_trade_date"),
            },
        )
    return result


def aggregate_daily(path: Path, asset: str) -> pl.DataFrame:
    """Дневная ликвидность и граничные цены одного контракта."""
    frame = pl.read_parquet(path, columns=["timestamp", "open", "close", "volume"])
    if frame.is_empty():
        raise ValueError(f"{path}: пустой исходный файл")
    if frame["timestamp"].null_count() or frame["timestamp"].n_unique() != frame.height:
        raise ValueError(f"{path}: пустые или повторные timestamp в raw")
    invalid = frame.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_null()
                | ~pl.col(column).is_finite()
                | (pl.col(column) <= 0)
                for column in ("open", "close", "volume")
            ]
        )
    )
    if invalid.height:
        raise ValueError(f"{path}: некорректные данные для решения о переходе")
    return (
        frame.sort("timestamp")
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
    """Явная календарная шкала: выходные не переносятся на будущую дату."""
    date_map = (
        daily.select("calendar_date")
        .unique()
        .with_columns(pl.col("calendar_date").alias("date"))
    )
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


def choose_initial_contract(
    rows: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, date]],
) -> str:
    return max(
        rows,
        key=lambda contract: (
            float(rows[contract]["daily_volume"] or 0),
            -metadata[contract]["LSTTRADE"].toordinal(),
        ),
    )


def nearest_later_contract(
    current: str,
    known_yesterday: set[str],
    metadata: dict[str, dict[str, date]],
    eligible_after: date,
) -> str | None:
    current_maturity = metadata[current]["LSTTRADE"]
    candidates = [
        contract
        for contract in known_yesterday
        if metadata[contract]["LSTTRADE"] > max(current_maturity, eligible_after)
    ]
    return (
        min(candidates, key=lambda c: (metadata[c]["LSTTRADE"], c))
        if candidates
        else None
    )


def synchronized_closes(
    old_path: Path,
    new_path: Path,
    decision_date: date,
) -> tuple[float, float, datetime, float]:
    """Последняя общая минута дня и ее возраст относительно конца данных.

    Редкие сделки не делают синхронную пару несинхронной. Возраст сохраняется
    для контроля качества, но сам по себе не запрещает построение ряда.
    Поправка описывает спред в adjustment_timestamp, а не спред закрытия.
    """
    frames = []
    for path in (old_path, new_path):
        frame = (
            pl.read_parquet(path, columns=["timestamp", "close"])
            .filter(pl.col("timestamp").dt.date() == decision_date)
            .sort("timestamp")
        )
        if frame.is_empty() or frame["timestamp"].n_unique() != frame.height:
            raise ValueError(f"{path.stem}: нет однозначных цен за {decision_date}")
        frames.append(frame)
    common = frames[0].join(frames[1], on="timestamp", suffix="_new").sort("timestamp")
    if common.is_empty():
        raise ValueError(
            f"Нет общей минуты {old_path.stem}/{new_path.stem}: {decision_date}"
        )
    row = common.row(-1, named=True)
    latest = max(frame["timestamp"].max() for frame in frames)
    age_minutes = (latest - row["timestamp"]).total_seconds() / 60
    return float(row["close"]), float(row["close_new"]), row["timestamp"], age_minutes


def build_selection(
    asset: str,
    daily: pl.DataFrame,
    from_date: date,
    metadata: dict[str, dict[str, date]],
    paths: dict[str, Path],
    roll_days: int = 0,
    volume_window_days: int = 3,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Строит выбранный контракт на каждый день и журнал переходов."""
    if roll_days < 0:
        raise ValueError("Запас до LSTTRADE не может быть отрицательным")
    if volume_window_days < 1:
        raise ValueError("Окно усреднения объема должно быть положительным")
    all_rows = rows_by_date(daily)
    ordered_dates = sorted(all_rows)
    date_positions = {day: index for index, day in enumerate(ordered_dates)}

    def window_average_volume(contract: str, days: list[date]) -> float:
        if not days:
            return 0.0
        total = sum(
            float(all_rows[day].get(contract, {}).get("daily_volume") or 0)
            for day in days
        )
        return total / len(days)

    dates = sorted(day for day in all_rows if day >= from_date)
    if not dates:
        raise ValueError(f"{asset}: нет данных начиная с {from_date}")

    prior_dates = sorted(day for day in all_rows if day < dates[0])
    if prior_dates:
        initial_decision = prior_dates[-1]
        series_dates = dates
    else:
        if len(dates) < 2:
            raise ValueError(
                f"{asset}: слишком короткая история для инициализации с {dates[0]}"
            )
        initial_decision = dates[0]
        series_dates = dates[1:]

    initial_rows = {
        contract: row
        for contract, row in all_rows[initial_decision].items()
        if metadata[contract]["LSTTRADE"] > series_dates[0] + timedelta(days=roll_days)
        and row["daily_volume"] > 0
    }
    if not initial_rows:
        raise ValueError(f"{asset}: нет подходящего начального контракта")
    current = choose_initial_contract(initial_rows, metadata)
    if current not in all_rows[series_dates[0]]:
        raise ValueError(
            f"{asset}: выбранный заранее {current} отсутствует {series_dates[0]}"
        )
    current_adjustment = 0.0
    selection: list[dict[str, Any]] = []
    rolls: list[dict[str, Any]] = []

    selection.append(
        {
            "asset": asset,
            "date": series_dates[0],
            "contract": current,
            "selection_reason": "initial_previous_volume",
            "decision_date": initial_decision,
            "daily_volume": all_rows[series_dates[0]][current]["daily_volume"],
            "price_adjustment": current_adjustment,
            "is_roll_day": False,
        }
    )

    for index in range(1, len(series_dates)):
        today = series_dates[index]
        yesterday = series_dates[index - 1]
        today_rows = all_rows[today]
        yesterday_rows = all_rows[yesterday]
        if yesterday >= metadata[current]["LSTTRADE"]:
            raise ValueError(
                f"{asset} {today}: текущий {current} уже достиг LSTTRADE "
                f"в день решения {yesterday}; проверьте справочник и данные"
            )
        reason = "keep_current"
        next_contract = nearest_later_contract(
            current,
            set(yesterday_rows),
            metadata,
            today + timedelta(days=roll_days),
        )

        roll_to: str | None = None
        deadline_reached = metadata[current]["LSTTRADE"] <= today + timedelta(
            days=roll_days
        )
        window_dates = ordered_dates[
            max(0, date_positions[today] - volume_window_days) : date_positions[today]
        ]
        current_average: float | None = None
        next_average: float | None = None
        if next_contract is not None:
            current_average = window_average_volume(current, window_dates)
            next_average = window_average_volume(next_contract, window_dates)
            if next_average > current_average:
                roll_to = next_contract
                reason = "next_avg_volume_exceeded_current"

        if roll_to is None and deadline_reached:
            if next_contract is None:
                raise ValueError(
                    f"{asset} {today}: срок перехода, но нет следующего контракта"
                )
            roll_to = next_contract
            reason = "last_trade_deadline"

        if roll_to is not None:
            old_row = yesterday_rows.get(current)
            new_row = yesterday_rows.get(roll_to)
            if old_row is None or new_row is None:
                raise ValueError(
                    f"{asset} {today}: для ценовой поправки нет закрытия "
                    f"{current} или {roll_to} за {yesterday}"
                )

            old_close, new_close, adjustment_timestamp, adjustment_age_minutes = (
                synchronized_closes(
                    paths[current],
                    paths[roll_to],
                    yesterday,
                )
            )
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
                    "old_avg_volume": (
                        current_average if current_average is not None else 0.0
                    ),
                    "new_avg_volume": (
                        next_average if next_average is not None else 0.0
                    ),
                    "volume_window_days": volume_window_days,
                    "old_close": old_close,
                    "new_close": new_close,
                    "gap_pct": (new_close / old_close - 1.0) * 100.0,
                    "adjustment_timestamp": adjustment_timestamp,
                    "adjustment_age_minutes": adjustment_age_minutes,
                    "adjustment_is_stale": adjustment_age_minutes > 15,
                    "adjustment_delta": adjustment_delta,
                    "cumulative_adjustment": current_adjustment,
                    "reason": reason,
                    "deadline_reached": deadline_reached,
                    "roll_days_before_last_trade": roll_days,
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
        frame = pl.read_parquet(
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
        ).with_columns(
            pl.lit(asset).alias("asset"),
            pl.lit(contract).alias("contract"),
            pl.col("timestamp").dt.date().alias("calendar_date"),
        )
        contract_dates = selection.filter(pl.col("contract") == contract)[
            "date"
        ].to_list()
        frame = frame.filter(pl.col("calendar_date").is_in(contract_dates))
        price_columns = ["open", "high", "low", "close"]
        if (
            frame["timestamp"].null_count()
            or frame["timestamp"].n_unique() != frame.height
        ):
            raise ValueError(f"{contract}: пустые или повторные timestamp в raw")
        bad = frame.filter(
            pl.any_horizontal(
                [
                    pl.col(column).is_null()
                    | ~pl.col(column).is_finite()
                    | (pl.col(column) <= 0)
                    for column in price_columns
                ]
            )
            | pl.col("volume").is_null()
            | (pl.col("volume") <= 0)
            | (pl.col("high") < pl.max_horizontal("open", "close", "low"))
            | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
        )
        if not bad.is_empty():
            raise ValueError(f"{contract}: некорректные цены/объем в raw: {bad.height}")
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
            pl.col("open_adjusted").first().alias("open_adjusted"),
            pl.col("high_adjusted").max().alias("high_adjusted"),
            pl.col("low_adjusted").min().alias("low_adjusted"),
            pl.col("close_adjusted").last().alias("close_adjusted"),
            pl.col("volume").sum().alias("volume"),
            pl.col("value").sum().alias("value"),
            pl.len().alias("source_minutes"),
            pl.col("timestamp").min().alias("first_source_timestamp"),
            pl.col("timestamp").max().alias("last_source_timestamp"),
            pl.col("price_adjustment").first().alias("price_adjustment"),
            pl.col("is_roll_day").first().alias("is_roll_day"),
        )
        .rename({"timestamp_15m": "timestamp"})
        .sort("asset", "timestamp")
        .with_columns(
            (
                pl.col("is_roll_day")
                & (
                    pl.col("contract") != pl.col("contract").shift(1).over("asset")
                ).fill_null(False)
            ).alias("is_first_bar_after_roll"),
            *[
                pl.col(f"{column}_raw").alias(column)
                for column in ("open", "high", "low", "close")
            ],
            # Отсутствие сделок не обязательно означает неполную загрузку.
            # Признак описывает только плотность минутных свечей.
            (pl.col("source_minutes") == 15).alias("has_all_15_minutes"),
            (pl.col("timestamp") - pl.col("timestamp").shift(1).over("asset"))
            .dt.total_minutes()
            .alias("elapsed_minutes"),
        )
        .with_columns(
            pl.when(pl.col("contract") == pl.col("contract").shift(1).over("asset"))
            .then(
                (pl.col("close_raw") / pl.col("close_raw").shift(1).over("asset")).log()
            )
            .otherwise(None)
            .alias("log_return"),
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
            "open_adjusted",
            "high_adjusted",
            "low_adjusted",
            "close_adjusted",
            "volume",
            "value",
            "source_minutes",
            "first_source_timestamp",
            "last_source_timestamp",
            "has_all_15_minutes",
            "elapsed_minutes",
            "log_return",
            "price_adjustment",
            "is_roll_day",
            "is_first_bar_after_roll",
        )
    )


def add_etf_nav(bars: pl.DataFrame, roll_cost_bps: float = 0.0) -> pl.DataFrame:
    """NAV по ETF Trick: carry и разрыв ролла не искажают доходность.

    Применяется к ряду одного рынка. Внутри контракта множитель
    close/prev_close. На первом баре нового контракта — close/open того же
    бара, поэтому разрыв между контрактами в NAV не попадает. На каждом
    переходе дополнительно вычитается roll_cost_bps. Первый бар ряда задает
    NAV = 1.
    """
    if not 0.0 <= roll_cost_bps < 10_000.0:
        raise ValueError("Издержка перехода должна быть в диапазоне [0, 10000) б.п.")
    ordered = bars.sort("timestamp").with_row_index("_row", offset=1)
    delta = (
        pl.when(pl.col("_row") == 1)
        .then(0.0)
        .when(pl.col("is_first_bar_after_roll"))
        .then(pl.col("close") / pl.col("open") - 1.0)
        .otherwise(pl.col("close") / pl.col("close").shift(1) - 1.0)
    )
    cost_factor = (
        pl.when(pl.col("is_first_bar_after_roll"))
        .then(1.0 - roll_cost_bps / 10_000.0)
        .otherwise(1.0)
    )
    return ordered.with_columns(
        (cost_factor * (1.0 + delta)).cum_prod().alias("nav")
    ).drop("_row")


def validate_result(
    asset: str,
    bars: pl.DataFrame,
    selection: pl.DataFrame,
    metadata: dict[str, dict[str, date]],
) -> None:
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
        metadata[contract]["LSTTRADE"]
        for contract in selection.get_column("contract").to_list()
    ]
    if any(right < left for left, right in pairwise(maturities)):
        raise ValueError(f"{asset}: обнаружен возврат к более ранней экспирации")
    if any(
        expiry <= day
        for expiry, day in zip(maturities, selection.get_column("date").to_list())
    ):
        raise ValueError(f"{asset}: контракт выбран в день LSTTRADE или позже")
    if selection.filter(pl.col("decision_date") >= pl.col("date")).height:
        raise ValueError(f"{asset}: решение использует незавершенный день")
    if selection.join(bars.select("date").unique(), on="date", how="anti").height:
        raise ValueError(f"{asset}: выбранный контракт не дал ни одного бара")
    if bars.filter(
        pl.col("is_first_bar_after_roll") & pl.col("log_return").is_not_null()
    ).height:
        raise ValueError(f"{asset}: доходность пересекает смену контракта")

    invalid_nav = bars.select(
        (
            pl.col("nav").is_null() | ~pl.col("nav").is_finite() | (pl.col("nav") <= 0)
        ).sum()
    ).item()
    if invalid_nav:
        raise ValueError(f"{asset}: найдено {invalid_nav} некорректных значений NAV")
    if bars["nav"][0] != 1.0:
        raise ValueError(f"{asset}: NAV должен начинаться с 1.0")


def require_derived_path(path: Path) -> Path:
    """Записи разрешены только внутри data, но не в raw и не в reference."""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT / "data"):
        raise ValueError(f"Путь записи должен находиться внутри data: {resolved}")
    if resolved.is_relative_to(RAW_DIR.parent) or resolved.is_relative_to(
        REFERENCE_DIR
    ):
        raise ValueError(f"raw и reference доступны только для чтения: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    if args.roll_days_before_last_trade < 0:
        raise ValueError("Запас до LSTTRADE не может быть отрицательным")
    if args.volume_window_days < 1:
        raise ValueError("Окно усреднения объема должно быть положительным")
    if not 0.0 <= args.roll_cost_bps < 10_000.0:
        raise ValueError("Издержка перехода должна быть в диапазоне [0, 10000) б.п.")
    if len(args.assets) != len(set(args.assets)):
        raise ValueError("Повторные --assets запрещены")
    output_dir = require_derived_path(args.output_dir)
    metadata_path = args.metadata_file.resolve()
    today_msk = datetime.now(ZoneInfo("Europe/Moscow")).date()
    if args.completed_through and args.completed_through >= today_msk:
        raise ValueError("--completed-through должен быть раньше текущего дня MSK")

    requested, process_all = select_requested_assets(args.assets)
    candidates = {asset: registry_name(asset) for asset in requested}
    if len(set(candidates.values())) != len(candidates):
        raise ValueError("В --assets повторяются рынки")
    if process_all:
        candidates, skipped = select_ready_assets(
            candidates, reference_tickers(metadata_path)
        )
        for label, reason in skipped:
            print(f"{label}: пропущен ({reason})")
    if not candidates:
        raise ValueError("Нет рынков, готовых к склейке")

    contracts = [
        path.stem
        for name in candidates.values()
        for path in (RAW_DIR / name).glob("*.parquet")
    ]
    metadata = load_contract_metadata(contracts, metadata_path)
    print(f"Локальный справочник: {metadata_path}, контрактов: {len(metadata)}")
    print(f"Рынки: {', '.join(candidates)}")

    all_bars: list[pl.DataFrame] = []
    all_selections: list[pl.DataFrame] = []
    all_rolls: list[pl.DataFrame] = []
    processed: list[str] = []

    failures: list[tuple[str, str]] = []
    for asset, name in candidates.items():
        print(f"{asset}: чтение контрактов")
        try:
            daily, paths, date_map = load_daily_market(
                asset,
                RAW_DIR / name,
            )
            last_raw_date = daily["date"].max()
            completed_through = args.completed_through or min(
                last_raw_date - timedelta(days=1),
                today_msk - timedelta(days=1),
            )
            if completed_through > last_raw_date:
                raise ValueError(
                    f"{asset}: данные заканчиваются раньше --completed-through"
                )
            daily = daily.filter(pl.col("date") <= completed_through)
            start_date = effective_start_date(name, args.from_date)
            if start_date != args.from_date:
                print(
                    f"{asset}: старт сдвинут с {args.from_date} до {start_date} "
                    "(continuous_from реестра)"
                )
            print(
                f"{asset}: календарные дни MSK, данные по {completed_through}, "
                f"старт {start_date}"
            )
            selection, rolls = build_selection(
                asset,
                daily,
                start_date,
                metadata,
                paths,
                args.roll_days_before_last_trade,
                args.volume_window_days,
            )
            minutes = load_selected_minutes(asset, paths, selection, date_map)
            bars = resample_15_minutes(minutes)
            bars = add_etf_nav(bars, args.roll_cost_bps)
            validate_result(asset, bars, selection, metadata)
        except (ValueError, FileNotFoundError) as error:
            if not process_all:
                raise
            failures.append((asset, str(error)))
            print(f"{asset}: ошибка, пропущен ({error})")
            continue

        all_bars.append(bars)
        all_selections.append(selection)
        processed.append(asset)
        if not rolls.is_empty():
            all_rolls.append(rolls)
            volume_rolls = rolls.filter(
                pl.col("reason") == "next_avg_volume_exceeded_current"
            ).height
            print(
                f"{asset}: переходов по объему {volume_rolls}, по дедлайну {rolls.height - volume_rolls}"
            )
            gap_abs = rolls["gap_pct"].abs()
            print(
                f"{asset}: разрыв на переходах ср. {gap_abs.mean():.2f}%, "
                f"макс {gap_abs.max():.2f}% — в NAV не попадает"
            )
            stale = rolls.filter(pl.col("adjustment_is_stale"))
            if stale.height:
                print(
                    f"{asset}: {stale.height} поправок по общей минуте старше 15 минут "
                    f"относительно конца данных дня; максимум "
                    f"{stale['adjustment_age_minutes'].max():.0f} мин. "
                    "Подробности в roll_schedule.csv."
                )

        print(
            f"{asset}: {bars.height:,} баров, "
            f"{selection.height:,} дней, {rolls.height} переходов, "
            f"NAV {bars['nav'][-1]:.4f}"
        )

    if not all_bars:
        raise ValueError("Ни один рынок не удалось обработать")
    if failures:
        print(
            f"\nПропущено рынков: {len(failures)} — "
            + "; ".join(f"{asset}: {reason}" for asset, reason in failures)
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
        else pl.DataFrame(
            schema={
                "asset": pl.String,
                "roll_date": pl.Date,
                "decision_date": pl.Date,
                "from_contract": pl.String,
                "to_contract": pl.String,
                "old_volume": pl.Int64,
                "new_volume": pl.Int64,
                "old_avg_volume": pl.Float64,
                "new_avg_volume": pl.Float64,
                "volume_window_days": pl.Int64,
                "old_close": pl.Float64,
                "new_close": pl.Float64,
                "gap_pct": pl.Float64,
                "adjustment_timestamp": pl.Datetime,
                "adjustment_age_minutes": pl.Float64,
                "adjustment_is_stale": pl.Boolean,
                "adjustment_delta": pl.Float64,
                "cumulative_adjustment": pl.Float64,
                "reason": pl.String,
                "deadline_reached": pl.Boolean,
                "roll_days_before_last_trade": pl.Int64,
            }
        )
    )

    # До этого места все рынки должны пройти проверки. Запись отдельных рынков
    # внутри цикла оставляла смешанные результаты при ошибке следующего рынка.
    output_dir.mkdir(parents=True, exist_ok=True)
    for asset, bars in zip(processed, all_bars):
        bars.write_parquet(
            output_dir / f"{asset.lower()}_15m.parquet", compression="zstd"
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
