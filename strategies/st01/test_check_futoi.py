"""Проверки признаков FUTOI. Запуск: uv run pytest strategies/st01"""

# FUTOI и свечи в тестах — naive timestamp MSK, как в данных.
# ruff: noqa: DTZ001

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from check_futoi import retail_share, share_at, with_retail, year_rank

from backtest.position import LONG, SHORT

START = datetime(2023, 1, 1, 12, 0)


def futoi_rows(
    time: datetime,
    fiz_pos: int,
    fiz_long: int,
    yur_long: int,
    people: tuple[int, int] = (1, 1),
) -> list[dict]:
    """Две строки FUTOI на момент: физлица и зеркальные юрлица.

    `people` — сколько физлиц в лонге и в шорте.
    """
    long_num, short_num = people
    return [
        {
            "timestamp": time,
            "clgroup": "FIZ",
            "pos": fiz_pos,
            "pos_long": fiz_long,
            "pos_long_num": long_num,
            "pos_short_num": short_num,
        },
        {
            "timestamp": time,
            "clgroup": "YUR",
            "pos": -fiz_pos,
            "pos_long": yur_long,
            "pos_long_num": 1,
            "pos_short_num": 1,
        },
    ]


def daily_shares(values: list[float], people: float = 50.0) -> pl.DataFrame:
    """Доли, известные раз в день с `START`; доля людей в лонге — `people`."""
    return pl.DataFrame(
        {
            "known_at": [START + timedelta(days=i) for i in range(len(values))],
            "share": [float(v) for v in values],
            "people": people,
        }
    )


def trades_at(signal_time: datetime) -> pl.DataFrame:
    """Лонг и шорт на одной свече."""
    return pl.DataFrame({"signal_time": [signal_time] * 2, "side": [LONG, SHORT]})


def test_retail_share_example_from_docstring():
    # Физлица 5.31 млн лонгов, нетто 4.47 млн; юрлица 5.77 млн лонгов: всего 11.08 млн.
    # В лонге 15 798 физлиц, в шорте 6 851: доля людей в лонге 70%.
    futoi = pl.DataFrame(
        futoi_rows(START, 4_470_996, 5_309_936, 5_769_074, people=(15_798, 6_851))
    )
    shares = retail_share(futoi)
    assert shares["known_at"].to_list() == [START + timedelta(minutes=1)]
    assert shares["share"][0] == pytest.approx(40.36, abs=0.01)
    assert shares["people"][0] == pytest.approx(69.75, abs=0.01)


def test_share_at_takes_last_known_and_drops_stale():
    known = np.array(["2024-01-01T10:00", "2024-01-02T10:00"], dtype="datetime64[us]")
    times = np.array(
        [
            "2024-01-01T09:00",
            "2024-01-01T12:00",
            "2024-01-02T10:00",
            "2024-01-08T10:00",
        ],
        dtype="datetime64[us]",
    )
    result = share_at(known, np.array([1.0, 2.0]), times)
    # До первой записи — пусто; запись, известная ровно в момент, — уже видна;
    # через 6 дней после последней записи — устарела.
    assert np.isnan(result[0])
    assert result[1:3] == pytest.approx([1.0, 2.0])
    assert np.isnan(result[3])


def test_year_rank_needs_a_year_and_ignores_future():
    shares = daily_shares(list(range(400)))
    known = shares["known_at"].to_numpy()
    values = shares["share"].to_numpy()
    times = known[[100, 399]]
    # Через 100 дней года истории нет. На день 399 окно — дни 34–399 (366 записей),
    # ниже 399 — 365 из них. Будущие записи не мешают: передаём значение 200
    # на день 399 — ниже него 166 записей (дни 34–199).
    ranks = year_rank(known, values, times, np.array([50.0, 399.0]))
    assert np.isnan(ranks[0])
    assert ranks[1] == pytest.approx(365 / 366)
    mid = year_rank(known, values, times[1:], np.array([200.0]))
    assert mid[0] == pytest.approx(166 / 366)


def test_short_gets_opposite_sign():
    # Доля растёт на 1 п.п. в день: за неделю +7. Для лонга — с нами, для шорта — против.
    shares = daily_shares(list(range(400)))
    trades = with_retail(trades_at(START + timedelta(days=399, hours=1)), shares)
    assert trades["fiz_week"].to_list() == pytest.approx([7.0, -7.0])
    long_level, short_level = trades["fiz_level"].to_list()
    assert long_level > 0.99
    assert short_level == pytest.approx(1 - long_level)


def test_lag_uses_data_two_weeks_earlier():
    # Доля 0 по день 380, с дня 381 — 10. Без сдвига (день 399) неделя назад — уже 10,
    # изменение 0; со сдвигом (день 385) — неделя назад было 0, изменение +10.
    shares = daily_shares([0.0] * 381 + [10.0] * 19)
    trades = with_retail(trades_at(START + timedelta(days=399, hours=1)), shares)
    assert trades["fiz_week"][0] == pytest.approx(0.0)
    assert trades["fiz_week_lag"][0] == pytest.approx(10.0)


def test_signal_does_not_see_record_at_bar_close():
    # Свеча 10:00 закрывается в 10:15. Запись FUTOI 10:15 выложена в 10:15:09 —
    # сигнал её не видит, берёт запись 10:10.
    day = datetime(2024, 3, 12)
    rows = [
        *futoi_rows(day - timedelta(days=7), 0, 50, 50),
        *futoi_rows(day.replace(hour=10, minute=10), 10, 50, 50),
        *futoi_rows(day.replace(hour=10, minute=15), 90, 95, 5),
    ]
    shares = retail_share(pl.DataFrame(rows))
    trades = with_retail(trades_at(day.replace(hour=10)), shares)
    assert trades["fiz_week"][0] == pytest.approx(10.0)


def test_no_futoi_after_its_end():
    # FUTOI кончается раньше свечей: сделки через 5 дней после конца — без признака.
    shares = daily_shares(list(range(400)))
    trades = with_retail(trades_at(START + timedelta(days=404)), shares)
    assert trades["fiz_week"].null_count() == 2
    assert trades["fiz_level"].null_count() == 2


def test_day_flow_and_people_follow_trade_side():
    # Доля растёт на 1 п.п. в день: за сутки +1. Людей в лонге 70%:
    # для лонга на его стороне 70%, для шорта — 30%.
    shares = daily_shares(list(range(400)), people=70.0)
    trades = with_retail(trades_at(START + timedelta(days=399, hours=1)), shares)
    assert trades["fiz_day"].to_list() == pytest.approx([1.0, -1.0])
    assert trades["fiz_people"].to_list() == pytest.approx([70.0, 30.0])


def test_time_column_can_be_named_otherwise():
    shares = daily_shares(list(range(400)))
    signals = trades_at(START + timedelta(days=399, hours=1)).rename(
        {"signal_time": "timestamp"}
    )
    trades = with_retail(signals, shares, time_column="timestamp")
    assert trades["fiz_week"].to_list() == pytest.approx([7.0, -7.0])
