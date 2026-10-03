"""Проверки паспортов рынков. Запуск: uv run pytest src/backtest/test_markets.py"""

# бары и курсы — naive timestamp MSK, как в data/continuous_15m и data/reference.
# ruff: noqa: DTZ001

from datetime import date, datetime

import polars as pl
import pytest

from backtest.markets import (
    MARKETS,
    Market,
    add_margin_rate,
    add_point_value,
    check_tick,
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


# Чт 10.09, Пт 11.09 и Пн 14.09.2026; у каждого курса своё значение,
# чтобы по результату было видно, какой клиринг выбран.
RATES = clearing_rates(
    [
        (date(2026, 9, 10), "tc", "13:45", 78.0),
        (date(2026, 9, 10), "mc", "18:49", 79.0),
        (date(2026, 9, 11), "tc", "13:45", 80.0),
        (date(2026, 9, 11), "mc", "18:49", 81.0),
        (date(2026, 9, 14), "tc", "13:45", 82.0),
        (date(2026, 9, 14), "mc", "18:49", 83.0),
    ]
)


def bars_at(*timestamps: str) -> pl.DataFrame:
    return pl.DataFrame({"timestamp": [datetime.fromisoformat(t) for t in timestamps]})


def test_rub_market_point_value_is_constant():
    bars = add_point_value(bars_at("2026-09-11T10:00", "2026-09-11T19:00"), CNY)
    assert bars["point_value"].to_list() == [1000.0, 1000.0]
    assert bars["point_value_known"].to_list() == [1000.0, 1000.0]


@pytest.mark.parametrize(
    ("timestamp", "settle", "known"),
    [
        # Свеча 13:15–13:30: рассчитывает дневной клиринг пятницы (80),
        # а на её закрытии известен только вечерний курс четверга (79).
        ("2026-09-11T13:15", 80.0, 79.0),
        # Свеча 13:45–14:00 кончается ровно на дневном клиринге.
        ("2026-09-11T13:45", 80.0, 80.0),
        # Свеча «14:00» — минутки 14:05–14:14, после дневного клиринга.
        ("2026-09-11T14:00", 81.0, 80.0),
        # Свеча 18:45–19:00: вечерний курс зафиксирован в 18:49 и уже известен.
        ("2026-09-11T18:45", 81.0, 81.0),
        # Вечерняя сессия пятницы и субботняя сессия — дневной клиринг понедельника.
        ("2026-09-11T19:00", 82.0, 81.0),
        ("2026-09-12T10:00", 82.0, 81.0),
        ("2026-09-14T10:00", 82.0, 81.0),
        ("2026-09-14T14:00", 83.0, 82.0),
    ],
)
def test_usd_bar_is_settled_by_its_clearing(timestamp, settle, known):
    bars = add_point_value(bars_at(timestamp), BRENT, RATES)
    assert bars["point_value"][0] == pytest.approx(settle * 10)
    assert bars["point_value_known"][0] == pytest.approx(known * 10)


def test_usd_bars_after_last_clearing_rejected():
    # Вечерняя сессия понедельника: следующего клиринга в данных ещё нет.
    with pytest.raises(ValueError, match="clearing_fx"):
        add_point_value(bars_at("2026-09-14T19:00"), BRENT, RATES)


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
