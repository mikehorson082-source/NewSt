"""Проверки паспортов рынков. Запуск: uv run pytest src/backtest/test_markets.py"""

# бары и курсы — naive timestamp MSK, как в data/continuous_15m и data/reference.
# ruff: noqa: DTZ001

import math
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from backtest.markets import (
    MARKETS,
    Market,
    MarketParams,
    add_bar_sigma,
    add_margin_rate,
    add_point_value,
    add_session_marks,
    add_usual_volume,
    check_tick,
    settled_bars,
    snapshot_params,
)

BRENT = MARKETS["brent"]
CNY = MARKETS["cnyrub"]


def clearing_rates(rows: list[tuple[date, str, str, float]]) -> pl.DataFrame:
    """Курсы клиринга в формате rates_moex_clearing: (дата, tc/mc, время фиксинга, курс)."""
    return pl.DataFrame(
        [
            {
                "date": day,
                "clearing": clearing,
                "fixing_time": datetime.fromisoformat(f"{day}T{fixing}"),
                "pair": "USD/RUB",
                "rate": rate,
            }
            for day, clearing, fixing, rate in rows
        ]
    )


# Чт 10.09, Пт 11.09 и Пн 15.09.2025; у каждого курса своё значение,
# чтобы по результату было видно, какой клиринг выбран.
RATES = clearing_rates(
    [
        (date(2025, 9, 11), "tc", "13:45", 78.0),
        (date(2025, 9, 11), "mc", "18:49", 79.0),
        (date(2025, 9, 12), "tc", "13:45", 80.0),
        (date(2025, 9, 12), "mc", "18:49", 81.0),
        (date(2025, 9, 15), "tc", "13:45", 82.0),
        (date(2025, 9, 15), "mc", "18:49", 83.0),
    ]
)


def bars_at(*timestamps: str) -> pl.DataFrame:
    return pl.DataFrame({"timestamp": [datetime.fromisoformat(t) for t in timestamps]})


def test_rub_market_point_value_is_constant():
    bars = add_point_value(bars_at("2025-09-12T10:00", "2025-09-12T19:00"), CNY)
    assert bars["point_value"].to_list() == [1000.0, 1000.0]
    assert bars["point_value_known"].to_list() == [1000.0, 1000.0]


@pytest.mark.parametrize(
    ("timestamp", "settle", "known"),
    [
        # Свеча 13:15–13:30: рассчитывает дневной клиринг пятницы (80),
        # а на её закрытии известен только вечерний курс четверга (79).
        ("2025-09-12T13:15", 80.0, 79.0),
        # Свеча 13:45–14:00 кончается ровно на дневном клиринге.
        ("2025-09-12T13:45", 80.0, 80.0),
        # Свеча «14:00» — минутки 14:05–14:14, после дневного клиринга.
        ("2025-09-12T14:00", 81.0, 80.0),
        # Свеча 18:45–19:00: вечерний курс зафиксирован в 18:49 и уже известен.
        ("2025-09-12T18:45", 81.0, 81.0),
        # Вечерняя сессия пятницы и субботняя сессия — дневной клиринг понедельника.
        ("2025-09-12T19:00", 82.0, 81.0),
        ("2025-09-13T10:00", 82.0, 81.0),
        ("2025-09-15T10:00", 82.0, 81.0),
        ("2025-09-15T14:00", 83.0, 82.0),
    ],
)
def test_usd_bar_is_settled_by_its_clearing(timestamp, settle, known):
    bars = add_point_value(bars_at(timestamp), BRENT, RATES)
    assert bars["point_value"][0] == pytest.approx(settle * 10)
    assert bars["point_value_known"][0] == pytest.approx(known * 10)


def test_usd_bars_after_last_clearing_rejected():
    # Вечерняя сессия понедельника: следующего клиринга в данных ещё нет.
    with pytest.raises(ValueError, match="clearing_fx"):
        add_point_value(bars_at("2025-09-15T19:00"), BRENT, RATES)


def test_unsettled_tail_is_dropped_with_warning():
    # Последний клиринг в данных — понедельник 15.09, 19:00. Свеча 18:45
    # кончается в 19:00 и рассчитана, вечерняя сессия — ещё нет.
    bars = bars_at("2025-09-15T18:30", "2025-09-15T18:45", "2025-09-15T19:00")
    with pytest.warns(UserWarning, match="brent: у 1 свечей с 15.09.2025 19:00"):
        settled = settled_bars(bars, RATES, "brent")
    assert settled["timestamp"].to_list() == bars["timestamp"].to_list()[:2]
    # Оставшиеся свечи получают рубли за пункт без ошибки.
    assert add_point_value(settled, BRENT, RATES)["point_value"].null_count() == 0


def test_settled_bars_keep_everything_when_all_settled(recwarn):
    bars = bars_at("2025-09-15T10:00", "2025-09-15T18:45")
    assert settled_bars(bars, RATES, "brent").equals(bars)
    assert not recwarn.list


def test_unsettled_tail_longer_than_a_week_means_stale_rates():
    bars = bars_at("2025-09-15T18:45", "2025-09-23T10:00")
    with pytest.raises(ValueError, match="clearing_fx"):
        settled_bars(bars, RATES, "brent")


# Те же дни недели после 23.03.2026: единая торговая сессия, дневного
# клиринга нет. Биржа публикует tc — это вчерашний вечерний курс.
SINGLE_SESSION_RATES = clearing_rates(
    [
        (date(2026, 9, 10), "tc", "14:00", 78.0),
        (date(2026, 9, 10), "mc", "19:00", 79.0),
        (date(2026, 9, 11), "tc", "14:00", 79.0),
        (date(2026, 9, 11), "mc", "19:00", 81.0),
        (date(2026, 9, 14), "tc", "14:00", 81.0),
        (date(2026, 9, 14), "mc", "19:00", 83.0),
    ]
)


@pytest.mark.parametrize(
    ("timestamp", "settle", "known"),
    [
        # Утро пятницы: клиринг пятницы считает по курсу 19:00 пятницы (81),
        # а не по дневному (79); на закрытии свечи известен вчерашний (79).
        ("2026-09-11T10:00", 81.0, 79.0),
        ("2026-09-11T14:00", 81.0, 79.0),
        # Свеча 18:45–19:00 кончается на расчётной цене дня.
        ("2026-09-11T18:45", 81.0, 81.0),
        # После 19:00 цена уходит в расчёт следующего торгового дня:
        # вечер пятницы и суббота — курс понедельника (83).
        ("2026-09-11T19:00", 83.0, 81.0),
        ("2026-09-12T10:00", 83.0, 81.0),
        ("2026-09-14T10:00", 83.0, 81.0),
    ],
)
def test_single_session_has_one_clearing_rate_per_day(timestamp, settle, known):
    bars = add_point_value(bars_at(timestamp), BRENT, SINGLE_SESSION_RATES)
    assert bars["point_value"][0] == pytest.approx(settle * 10)
    assert bars["point_value_known"][0] == pytest.approx(known * 10)


def test_day_before_single_session_keeps_two_clearings():
    rates = clearing_rates(
        [
            (date(2026, 3, 19), "mc", "18:49", 84.0),
            (date(2026, 3, 20), "tc", "13:45", 84.0),
            (date(2026, 3, 20), "mc", "18:49", 83.0),
            (date(2026, 3, 23), "tc", "14:00", 83.0),
            (date(2026, 3, 23), "mc", "19:00", 81.0),
        ]
    )
    bars = add_point_value(
        bars_at("2026-03-20T10:00", "2026-03-23T10:00"), BRENT, rates
    )
    # Пятница 20.03 — ещё дневной клиринг (84), понедельник 23.03 — уже нет (81).
    assert bars["point_value"].to_list() == [840.0, 810.0]


def test_usd_market_needs_clearing_rates():
    with pytest.raises(ValueError, match="курс клиринга"):
        add_point_value(bars_at("2026-09-11T10:00"), BRENT)


MARGIN_RATES = pl.DataFrame(
    {
        "date": [date(2026, 9, 10), date(2026, 9, 14), date(2026, 9, 10)],
        "asset_code": ["BR", "BR", "GOLD"],
        "mr1": [0.10, 0.20, 0.50],
    }
)


def test_margin_rate_follows_mr1_by_date():
    bars = add_margin_rate(
        bars_at("2026-09-11T10:00", "2026-09-12T10:00", "2026-09-14T10:00"),
        BRENT,
        MARGIN_RATES,
        margin_factor=1.2,
    )
    assert bars["margin_rate"].to_list() == pytest.approx([0.12, 0.12, 0.24])


def test_margin_rate_missing_rejected():
    with pytest.raises(ValueError, match="--only margins"):
        add_margin_rate(bars_at("2026-09-09T10:00"), BRENT, MARGIN_RATES, 1.2)


def snapshot_row(
    snapshot_day: date,
    secid: str,
    open_interest: float,
    margin_factor: float,
    step_price: float = 0.83,
    asset_code: str = "BR",
) -> dict:
    return {
        "snapshot_date": snapshot_day,
        "im_time": datetime.combine(snapshot_day, datetime.min.time()).replace(hour=10),
        "secid": secid,
        "asset_code": asset_code,
        "open_interest": open_interest,
        "min_step": 0.01,
        "step_price": step_price,
        "margin_factor": margin_factor,
    }


def test_snapshot_params_take_latest_most_liquid_contract():
    # Курс к утру 15.09 — вечерний понедельника, 83: $10 за пункт = 830 ₽,
    # шаг 0.01 стоит 8.30 ₽. Старый снимок и неликвидный контракт не берутся.
    snapshot = pl.DataFrame(
        [
            snapshot_row(date(2026, 9, 12), "BRX6", 5000, 1.10, step_price=8.10),
            snapshot_row(date(2026, 9, 15), "BRX6", 1000, 1.21, step_price=8.30),
            snapshot_row(date(2026, 9, 15), "BRZ6", 10, 1.38, step_price=8.30),
            snapshot_row(date(2026, 9, 15), "GDZ6", 9000, 1.31, asset_code="GOLD"),
        ]
    )
    params = snapshot_params(BRENT, snapshot, RATES)
    assert params.snapshot_contract == "BRX6"
    assert params.snapshot_date == date(2026, 9, 15)
    assert params.margin_factor == pytest.approx(1.21)
    assert params.tick_size == pytest.approx(0.01)


def test_contract_spec_keeps_only_constant_rules():
    # Рубли за пункт и ГО меняются по датам — они в барах, а не в спецификации.
    params = MarketParams(BRENT, 0.01, 1.21, date(2026, 9, 15), "BRX6")
    spec = params.contract_spec()
    assert spec.asset == "brent"
    assert spec.tick_size == pytest.approx(0.01)
    assert spec.slippage == pytest.approx(0.01)
    assert spec.exchange_fee_rate == pytest.approx(BRENT.exchange_fee_rate)
    assert spec.broker_fee_rate == 0.0


def test_snapshot_params_reject_wrong_point_value():
    # Если бы в паспорте Brent было $100 за пункт, снимок биржи (830 ₽) не сошёлся бы.
    wrong = Market("brent", "BR", "USD", 100.0, 0.0)
    snapshot = pl.DataFrame([snapshot_row(date(2026, 9, 15), "BRX6", 1000, 1.21, 8.30)])
    with pytest.raises(ValueError, match="не сходится"):
        snapshot_params(wrong, snapshot, RATES)


def test_snapshot_without_market_rejected():
    snapshot = pl.DataFrame(
        [snapshot_row(date(2026, 9, 15), "GDZ6", 9000, 1.31, asset_code="GOLD")]
    )
    with pytest.raises(ValueError, match="--only margins"):
        snapshot_params(BRENT, snapshot, RATES)


def test_check_tick_accepts_grid_and_rejects_off_grid_price():
    prices = {"open": [80.07], "high": [80.10], "low": [79.99], "close": [80.0]}
    bars = pl.DataFrame({"timestamp": [datetime(2026, 9, 11, 10)], **prices})
    check_tick(bars, 0.01, "brent")
    with pytest.raises(ValueError, match="вне шага"):
        check_tick(bars.with_columns(pl.lit(80.005).alias("close")), 0.01, "brent")


def test_market_rejects_unknown_currency():
    with pytest.raises(ValueError, match="RUB или USD"):
        Market("test", "TST", "EUR", 1.0, 0.0)


def volume_bars(rows: list[tuple[datetime, int]]) -> pl.DataFrame:
    return pl.DataFrame([{"timestamp": ts, "volume": volume} for ts, volume in rows])


def usual(bars: pl.DataFrame, ts: datetime) -> float | None:
    return bars.filter(pl.col("timestamp") == ts)["usual_volume"][0]


def test_usual_volume_uses_only_past_days_at_same_time():
    # Пн 07.09.2026 — пт 11.09: в 08:45 торгуют 1, 2, 3, 4, 5, 6 контрактов
    # (6-й день — пн 14.09), днём в 12:00 — по 1 000.
    days = [date(2026, 9, d) for d in (7, 8, 9, 10, 11, 14)]
    rows = []
    for i, day in enumerate(days):
        rows.append(
            (
                datetime(day.year, day.month, day.day, 8, 45),
                i + 1,
            )
        )
        rows.append((datetime(day.year, day.month, day.day, 12), 1_000))
    bars = add_usual_volume(volume_bars(rows), days=20, min_days=5)

    # Шестой день: медиана пяти прошлых дней 1..5 = 3, свой объём 6 не входит.
    assert usual(bars, datetime(2026, 9, 14, 8, 45)) == 3.0
    assert usual(bars, datetime(2026, 9, 14, 12)) == 1_000.0
    # Пятый день: прошлых дней 4 — меньше min_days, значение неизвестно.
    assert usual(bars, datetime(2026, 9, 11, 8, 45)) is None


def test_usual_volume_window_and_weekends():
    # Будни в 12:00 по 1 000, выходные в 12:00 по 10. Окно 2 дня.
    start = datetime(2026, 9, 7, 12)  # понедельник
    rows = []
    for i in range(14):
        ts = start + timedelta(days=i)
        rows.append((ts, 10 if ts.weekday() >= 5 else 1_000 + i))
    bars = add_usual_volume(volume_bars(rows), days=2, min_days=1)

    # Сб 19.09: прошлые выходные — сб 12.09 и вс 13.09, будни не мешают.
    assert usual(bars, datetime(2026, 9, 19, 12)) == 10.0
    # Пт 18.09: медиана двух прошлых будних дней — чт 17.09 (1 010) и ср 16.09 (1 009).
    assert usual(bars, datetime(2026, 9, 18, 12)) == 1_009.5


def test_bar_sigma_uses_only_past_bars():
    start = datetime(2026, 9, 10, 10)
    navs = [1.0, 1.01, 0.99, 1.02, 1.0]
    bars = pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=15 * k) for k in range(5)],
            "nav": navs,
        }
    )
    sigma = add_bar_sigma(bars, window="20d", min_bars=2)["bar_sigma"].to_list()
    r1 = math.log(1.01)
    r2 = math.log(0.99 / 1.01)
    # Свеча 3: доходности свечей 1 и 2; своя доходность (1.02 / 0.99) не входит.
    mean = (r1 + r2) / 2
    expected = math.sqrt(((r1 - mean) ** 2 + (r2 - mean) ** 2) / 1)
    assert sigma[3] == pytest.approx(expected)
    # Свеча 2: в прошлом одна доходность — меньше min_bars, σ неизвестна.
    assert sigma[2] is None


def test_bar_sigma_window_drops_old_bars():
    days = [0, 1, 2, 30, 31, 32]
    bars = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(days=d) for d in days],
            "nav": [1.0, 1.5, 0.5, 1.0, 1.01, 1.0],
        }
    )
    sigma = add_bar_sigma(bars, window="20d", min_bars=2)["bar_sigma"].to_list()
    # День 32: окно 12–32 — только доходности дней 30 и 31, огромные
    # движения дней 1 и 2 в него не попадают.
    r30 = math.log(1.0 / 0.5)
    r31 = math.log(1.01)
    mean = (r30 + r31) / 2
    assert sigma[5] == pytest.approx(
        math.sqrt(((r30 - mean) ** 2 + (r31 - mean) ** 2) / 1)
    )


def marks(bars: pl.DataFrame) -> dict[str, tuple[bool, bool]]:
    """Свеча → (последняя свеча дня, последняя перед перерывом)."""
    marked = add_session_marks(bars)
    return {
        timestamp.strftime("%a %H:%M"): (day_end, before_break)
        for timestamp, day_end, before_break in marked.select(
            "timestamp", "is_day_end", "is_before_break"
        ).iter_rows()
    }


def test_session_marks_ordinary_night_and_weekend_sessions():
    # Чт 10.09 — Вт 15.09.2026, в субботу и воскресенье сессии выходных.
    bars = bars_at(
        "2026-09-10T23:30",
        "2026-09-10T23:45",
        "2026-09-11T08:45",
        "2026-09-11T23:45",
        "2026-09-12T09:45",
        "2026-09-12T18:45",
        "2026-09-13T18:45",
        "2026-09-14T23:45",
        "2026-09-15T08:45",
    )
    assert marks(bars) == {
        "Thu 23:30": (False, False),
        "Thu 23:45": (True, False),  # обычная ночь чт→пт
        "Fri 08:45": (False, False),
        "Fri 23:45": (True, True),  # впереди суббота
        "Sat 09:45": (False, False),
        "Sat 18:45": (True, True),  # позиция выходных закрывается в тот же день
        "Sun 18:45": (True, True),
        "Mon 23:45": (True, False),  # обычная ночь пн→вт
        "Tue 08:45": (False, False),  # последняя свеча ряда: дальше неизвестно
    }


def test_session_marks_weekend_without_sessions_and_holiday():
    bars = bars_at(
        "2024-04-26T23:45",  # пятница, выходных сессий ещё нет
        "2024-04-29T23:45",  # понедельник
        "2024-04-30T18:45",  # вторник, укороченный день перед 1 мая
        "2024-05-02T23:45",  # четверг
        "2024-05-03T08:45",
    )
    marked = add_session_marks(bars)
    assert marked["is_day_end"].to_list() == [True, True, True, True, False]
    # Пт→пн и вт→чт через праздник — перерывы; пн→вт и чт→пт — обычные ночи.
    assert marked["is_before_break"].to_list() == [True, False, True, False, False]


def test_session_marks_sort_bars_by_time():
    bars = bars_at("2026-09-11T08:45", "2026-09-10T23:45")
    marked = add_session_marks(bars)
    assert marked["timestamp"].is_sorted()
    assert marked["is_day_end"].to_list() == [True, False]
