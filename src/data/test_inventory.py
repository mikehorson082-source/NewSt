"""Проверки инвентаря данных. Запуск: uv run pytest src/data/test_inventory.py

Данные — маленькие parquet-файлы во временной папке с той же раскладкой, что data/.
"""

# Даты в данных — naive MSK, как в data/raw.
# ruff: noqa: DTZ001

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from data.inventory import InventoryReport, build_inventory
from data.inventory.catalog import REBUILD_HINT, futoi_market
from data.inventory.scan import (
    expected_last_date,
    freshness,
    plural,
    trading_days_behind,
)

# Пн 14.09.2026: ожидаемая дата — пт 11.09.2026.
TODAY = date(2026, 9, 14)
EXPECTED = date(2026, 9, 11)


def write(path: Path, frame: pl.DataFrame, statistics: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, statistics=statistics)


def minutes(path: Path, first: datetime, last: datetime) -> None:
    """Минутки контракта: первая и последняя минута."""
    write(path, pl.DataFrame({"timestamp": [first, last], "close": [1.0, 2.0]}))


def days(start: date, end: date, skip: tuple[date, ...] = ()) -> list[date]:
    """Будние дни от start до end без skip."""
    out, day = [], start
    while day <= end:
        if day.weekday() < 5 and day not in skip:
            out.append(day)
        day += timedelta(days=1)
    return out


def row(report: InventoryReport, row_id: str):
    return next(item for item in report.datasets if item.id == row_id)


def instrument(report: InventoryReport, instrument_id: str):
    return next(item for section in report.sections for item in section.instruments if item.id == instrument_id)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Раскладка data/: справочники, минутки двух рынков, дневки, склейка, FUTOI, Algopack."""
    root = tmp_path / "data"
    ref = root / "reference"
    write(
        ref / "margin_rates.parquet",
        pl.DataFrame(
            {
                "date": [date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 11)],
                "asset_code": ["BR", "BR", "GOLD"],
                "mr1": [0.18, 0.18, 0.07],
            }
        ),
    )
    write(ref / "rates_moex_clearing.parquet", pl.DataFrame({"date": [date(2026, 9, 11)], "rate": [80.0]}))
    write(ref / "custom_table.parquet", pl.DataFrame({"x": [1, 2, 3]}))

    # brent свежий: последний бар в субботней сессии 12.09; gold застрял на 01.09.
    minutes(root / "raw/futures/brent/BRV6.parquet", datetime(2026, 8, 3, 10), datetime(2026, 9, 1, 18))
    minutes(root / "raw/futures/brent/BRX6.parquet", datetime(2026, 9, 1, 10), datetime(2026, 9, 12, 15))
    minutes(root / "raw/futures/gold/GDZ6.parquet", datetime(2026, 8, 3, 10), datetime(2026, 9, 1, 18))

    # Торговый календарь — даты дневных свечей. 07.09 — «праздник»: дневной свечи нет.
    holiday = date(2026, 9, 7)
    trade_days = days(date(2026, 9, 1), date(2026, 9, 11), skip=(holiday,))
    write(root / "raw/futures_daily/brent/BRX6.parquet", pl.DataFrame({"TRADEDATE": trade_days}))

    # Склейка brent: есть бар в праздник (торговалась часть рынков) и нет бара 09.09 — пропуск.
    bars = [d for d in trade_days if d != date(2026, 9, 9)] + [holiday]
    write(root / "continuous_15m/brent_15m.parquet", pl.DataFrame({"date": sorted(bars)}))
    roll = root / "continuous_15m/roll_schedule.csv"
    roll.write_text("asset,roll_date,adjustment_is_stale\nbrent,2026-09-01,false\nbrent,2026-09-08,true\n")

    write(root / "raw/futoi/5min/BR.parquet", pl.DataFrame({"tradedate": [date(2026, 9, 11)]}))
    write(root / "raw/futoi/daily/BR.parquet", pl.DataFrame({"tradedate": [date(2026, 9, 4)]}))
    return root


@pytest.fixture
def stocks_root(data_root: Path) -> Path:
    """Плюс акции: Algopack по годам, дивиденды и листинг с названиями."""
    tradestats = data_root / "raw/algopack/eq/tradestats"
    write(
        tradestats / "tradestats_2022.parquet",
        pl.DataFrame({"tradedate": [date(2022, 1, 3), date(2022, 12, 30), date(2022, 6, 1)], "secid": ["SBER", "SBER", "GAZP"]}),
    )
    write(tradestats / "tradestats_2023.parquet", pl.DataFrame({"tradedate": [date(2023, 3, 1)], "secid": ["SBER"]}))
    write(
        data_root / "reference/corp_dividends.parquet",
        pl.DataFrame({"ticker": ["SBER", "MTSS"], "declared_date": [date(2025, 7, 1), date(2025, 6, 1)]}),
    )
    write(
        data_root / "reference/corp_listing.parquet",
        pl.DataFrame({"ticker": ["SBER", "GAZP"], "shortname": ["Сбербанк", "Газпром"]}),
    )
    # Фьючерс на Сбербанк: его карточка покажет дивиденды и Algopack базовой акции.
    minutes(data_root / "raw/futures/sber/SRZ6.parquet", datetime(2026, 9, 1, 10), datetime(2026, 9, 11, 18))
    return data_root


# ---------------------------------------------------------------- calendar
def test_expected_date_is_last_completed_trading_day():
    assert expected_last_date(date(2026, 9, 14)) == date(2026, 9, 11)  # пн → пт
    assert expected_last_date(date(2026, 9, 13)) == date(2026, 9, 11)  # вс → пт
    assert expected_last_date(date(2026, 9, 9)) == date(2026, 9, 8)  # ср → вт


def test_working_saturday_counts_as_trading_day():
    saturday = frozenset({date(2026, 9, 12)})
    assert expected_last_date(date(2026, 9, 14), saturday) == date(2026, 9, 12)
    assert trading_days_behind(date(2026, 9, 11), date(2026, 9, 14), saturday) == 2


def test_trading_days_behind_skips_weekends():
    assert trading_days_behind(date(2026, 9, 11), date(2026, 9, 14)) == 1  # пт → пн: только пн
    assert trading_days_behind(date(2026, 9, 10), date(2026, 9, 14)) == 2  # чт → пн: пт и пн
    assert trading_days_behind(date(2026, 9, 14), date(2026, 9, 14)) == 0
    assert trading_days_behind(date(2026, 9, 15), date(2026, 9, 14)) == 0  # данные свежее ожидаемой даты


@pytest.mark.parametrize(
    ("last", "allowed", "expected"),
    [
        (date(2026, 9, 11), 0, ("ok", 0)),
        (date(2026, 9, 10), 0, ("lagging", 1)),
        (date(2026, 9, 10), 1, ("ok", 1)),
        (date(2026, 9, 4), 0, ("lagging", 5)),
        (date(2026, 9, 3), 0, ("stale", 6)),
    ],
)
def test_freshness_thresholds(last, allowed, expected):
    assert freshness(last, EXPECTED, allowed) == expected


@pytest.mark.parametrize(
    ("n", "word"),
    [(1, "день"), (2, "дня"), (5, "дней"), (11, "дней"), (14, "дней"), (21, "день"), (22, "дня"), (112, "дней")],
)
def test_plural(n, word):
    assert plural(n, ("день", "дня", "дней")) == word


# ---------------------------------------------------------------- datasets
def test_contract_dir_sums_files_and_uses_parquet_statistics(data_root):
    brent = row(build_inventory(data_root, today=TODAY), "raw/futures/brent")
    assert (brent.title, brent.kind) == ("Минутные свечи", "futures_minutes")
    assert brent.rows == 4
    assert (brent.first_date, brent.last_date, brent.status) == (date(2026, 8, 3), date(2026, 9, 12), "ok")
    assert [(fact.label, fact.value) for fact in brent.facts] == [("Контрактов", 2)]


def test_stale_market(data_root):
    gold = row(build_inventory(data_root, today=TODAY), "raw/futures/gold")
    assert (gold.last_date, gold.lag_days, gold.status) == (date(2026, 9, 1), 8, "stale")


def test_reference_files_whole(data_root):
    report = build_inventory(data_root, today=TODAY)
    margins = row(report, "reference/margin_rates.parquet")
    assert margins.status == "ok"
    assert [(fact.label, fact.value) for fact in margins.facts] == [("Рынков", 2)]
    assert row(report, "reference/indices_daily.parquet").note == "нет файла"
    custom = row(report, "reference/custom_table.parquet")
    assert (custom.status, custom.rows, custom.description) == ("unchecked", 3, "Нет в каталоге инвентаря")


def test_unreadable_file_is_error_but_others_still_counted(data_root):
    (data_root / "raw/futures/brent/BROKEN.parquet").write_bytes(b"not a parquet")
    brent = row(build_inventory(data_root, today=TODAY), "raw/futures/brent")
    assert brent.status == "error"
    assert "BROKEN.parquet" in (brent.note or "")
    assert (brent.files, brent.rows) == (3, 4)  # два читаемых контракта посчитаны


def test_dates_without_parquet_statistics(data_root):
    path = data_root / "raw/futures/silver/SVZ6.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"timestamp": [datetime(2026, 9, 2, 10), datetime(2026, 9, 11, 18)]}).write_parquet(
        path, statistics=False
    )
    silver = row(build_inventory(data_root, today=TODAY), "raw/futures/silver")
    assert (silver.first_date, silver.last_date, silver.status) == (date(2026, 9, 2), date(2026, 9, 11), "ok")


def test_continuous_gaps_follow_daily_calendar(data_root):
    brent = row(build_inventory(data_root, today=TODAY), "continuous_15m/brent_15m.parquet")
    facts = {fact.label: fact.value for fact in brent.facts}
    # 09.09 — пропуск; праздник 07.09 в календаре дневок отсутствует и пропуском не считается.
    assert facts["Пропусков"] == 1
    assert "2026-09-09" in (brent.note or "")
    assert "2026-09-07" not in (brent.note or "")
    assert (facts["Роллов"], facts["Поправок по устаревшей минуте"]) == (2, 1)


def test_continuous_behind_raw_minutes_asks_to_rebuild(data_root):
    # Минутки brent дошли до вт 15.09, а склейка — только до 11.09.
    minutes(data_root / "raw/futures/brent/BRZ6.parquet", datetime(2026, 9, 14, 10), datetime(2026, 9, 15, 12))
    brent = row(build_inventory(data_root, today=date(2026, 9, 15)), "continuous_15m/brent_15m.parquet")
    assert brent.status == "lagging"
    assert "отстаёт от минуток на 2 торговых дня" in (brent.note or "")
    assert REBUILD_HINT in (brent.note or "")


def test_working_saturday_shifts_expected_date(data_root):
    (data_root / "reference/ru_working_saturdays.json").write_text(json.dumps(["2026-09-12"]))
    report = build_inventory(data_root, today=TODAY)
    assert report.expected_date == date(2026, 9, 12)
    assert row(report, "raw/futoi/5min/BR.parquet").lag_days == 1  # данные по 11.09, не хватает субботы 12.09
    saturdays = row(report, "reference/ru_working_saturdays.json")
    assert (saturdays.rows, saturdays.status) == (1, "unchecked")


@pytest.mark.parametrize(
    ("last", "lag", "status"),
    [
        (date(2026, 8, 28), 10, "ok"),  # две недели: так отдаёт бесплатный API
        (date(2026, 8, 27), 11, "lagging"),  # FUTOI давно не обновляли
        (date(2026, 8, 20), 16, "stale"),
    ],
)
def test_futoi_two_week_delay_is_normal(data_root, last, lag, status):
    write(data_root / "raw/futoi/daily/BR.parquet", pl.DataFrame({"tradedate": [last]}))
    daily = row(build_inventory(data_root, today=TODAY), "raw/futoi/daily/BR.parquet")
    assert (daily.lag_days, daily.status) == (lag, status)


# ------------------------------------------------------------- instruments
def test_futures_instrument_collects_everything_about_the_market(data_root):
    report = build_inventory(data_root, today=TODAY)
    brent = instrument(report, "futures/brent")
    assert (brent.title, brent.subtitle) == ("Нефть Brent", "brent · BR · сырьё")
    assert brent.datasets == (
        "raw/futures/brent",
        "raw/futures_daily/brent",
        "continuous_15m/brent_15m.parquet",
        "raw/futoi/5min/BR.parquet",
        "raw/futoi/daily/BR.parquet",
        "reference/margin_rates.parquet#BR",
        "reference/rates_moex_clearing.parquet",  # нефть котируется в долларах
    )
    margins = row(report, "reference/margin_rates.parquet#BR")
    assert (margins.rows, margins.first_date, margins.size_bytes) == (2, date(2026, 9, 10), 0)
    assert margins.description == "Минимальные ставки ГО mr1–mr3 · BR"


def test_shared_dataset_is_stored_once(data_root):
    report = build_inventory(data_root, today=TODAY)
    clearing = "reference/rates_moex_clearing.parquet"
    assert clearing in instrument(report, "futures/brent").datasets
    assert clearing in instrument(report, "futures/gold").datasets
    assert clearing in instrument(report, "common/reference").datasets
    assert [item.id for item in report.datasets].count(clearing) == 1
    # Каждая ссылка инструмента ведёт на набор отчёта, и каждый набор у кого-то есть.
    referenced = {d for section in report.sections for item in section.instruments for d in item.datasets}
    assert referenced == {item.id for item in report.datasets}
    assert sum(report.summary.values()) == len(report.datasets)


def test_futoi_ticker_matches_market_case_insensitively(data_root):
    assert (futoi_market("Si"), futoi_market("USDRUBF"), futoi_market("XX")) == ("si", "perpetual", None)
    write(data_root / "raw/futoi/daily/Si.parquet", pl.DataFrame({"tradedate": [date(2026, 9, 1)]}))
    write(data_root / "raw/futoi/daily/XX.parquet", pl.DataFrame({"tradedate": [date(2026, 9, 1)]}))
    report = build_inventory(data_root, today=TODAY)
    assert instrument(report, "futures/si").datasets == ("raw/futoi/daily/Si.parquet",)
    assert instrument(report, "common/futoi").datasets == ("raw/futoi/daily/XX.parquet",)


def test_stocks_from_algopack_and_corporate_events(stocks_root):
    report = build_inventory(stocks_root, today=TODAY)
    sber = instrument(report, "stocks/SBER")
    assert (sber.title, sber.subtitle) == ("Сбербанк", "SBER")
    assert sber.datasets == ("raw/algopack/eq/tradestats#SBER", "reference/corp_dividends.parquet#SBER")
    algopack = row(report, "raw/algopack/eq/tradestats#SBER")
    assert (algopack.files, algopack.rows) == (2, 3)  # два файла-года, три строки
    assert (algopack.first_date, algopack.last_date, algopack.status) == (date(2022, 1, 3), date(2023, 3, 1), "unchecked")
    # MTSS есть только в дивидендах, названия в листинге нет — заголовок по тикеру.
    assert instrument(report, "stocks/MTSS").title == "MTSS"
    # Фьючерс на Сбербанк показывает и данные базовой акции.
    assert "reference/corp_dividends.parquet#SBER" in instrument(report, "futures/sber").datasets


def test_unknown_folders_and_missing_folders_go_to_common(data_root):
    (data_root / "experiments").mkdir()
    (data_root / "experiments/runs.csv").write_text("a\n1\n")
    (data_root / "reports").mkdir()  # пустая — не показывается
    report = build_inventory(data_root, today=TODAY)
    other = instrument(report, "common/other").datasets
    assert "experiments" in other
    assert "reports" not in other
    assert "raw/algopack" in other  # папки нет — строка «нет папки»
    assert row(report, "raw/algopack").note == "нет папки"


def test_empty_data_root(tmp_path):
    report = build_inventory(tmp_path, today=TODAY)
    assert [section.id for section in report.sections] == ["common"]
    assert row(report, "raw/futures").status == "missing"


def test_inventory_only_reads(stocks_root):
    """raw никогда не переписывается: после инвентаря ни один файл не изменился и не появился."""

    def snapshot() -> dict[str, int]:
        return {str(p): p.stat().st_mtime_ns for p in stocks_root.rglob("*")}

    before = snapshot()
    build_inventory(stocks_root, today=TODAY)
    assert snapshot() == before
