"""Проверки симулятора одной позиции. Запуск: uv run pytest src/backtest/test_position.py"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest.position import LONG, SHORT, Signal, simulate_position
from backtest.specs import ContractSpec

SPEC = ContractSpec(
    asset="TEST",
    point_value=1.0,
    tick_size=0.5,
    margin_rate=0.12,
    exchange_fee_rate=0.0001,
)


def make_bars(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def bar(
    minute: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    adjustment: float = 0.0,
    roll: bool = False,
) -> dict:
    return {
        "timestamp": datetime(2026, 9, 10, 10) + timedelta(minutes=minute),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "price_adjustment": adjustment,
        "is_first_bar_after_roll": roll,
    }


def long_signal(**kwargs) -> Signal:
    params = {"stop": 95.0, "target": 110.0, **kwargs}
    return Signal(side=LONG, **params)


def test_long_target_fills_by_next_open_with_costs():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 102.0, 106.0, 101.0, 105.0),
            bar(45, 108.0, 111.0, 107.0, 110.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal())
    assert trade.entry_price == pytest.approx(101.5)
    assert trade.exit_price == pytest.approx(109.5)
    assert trade.exit_reason == "target"
    assert trade.bars_held == 3
    assert trade.gross_pnl == pytest.approx(8.0)
    assert trade.commissions == pytest.approx(0.0211)
    assert trade.roll_costs == 0.0
    assert trade.net_pnl == pytest.approx(8.0 - 0.0211)


def test_stop_has_priority_when_stop_and_target_hit_in_one_bar():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 102.0, 106.0, 101.0, 105.0),
            bar(45, 108.0, 111.0, 94.0, 110.5),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal())
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(94.5)
    assert trade.gross_pnl == pytest.approx(-7.0)


def test_gap_through_stop_fills_at_open():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 102.0, 106.0, 101.0, 105.0),
            bar(45, 90.0, 92.0, 88.0, 89.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal())
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(89.5)
    assert trade.gross_pnl == pytest.approx(-12.0)


def test_stop_on_entry_bar():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 102.0, 94.0, 95.0),
            bar(30, 96.0, 97.0, 95.5, 96.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal())
    assert trade.exit_reason == "stop"
    assert trade.bars_held == 1
    assert trade.exit_price == pytest.approx(94.5)
    assert trade.gross_pnl == pytest.approx(-7.0)


def test_time_exit_at_close():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 102.0, 106.0, 101.0, 105.0),
            bar(45, 106.0, 109.0, 105.0, 108.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(max_bars=2))
    assert trade.exit_reason == "time"
    assert trade.bars_held == 2
    assert trade.exit_price == pytest.approx(104.5)
    assert trade.gross_pnl == pytest.approx(3.0)


def test_short_target_mirrors_long():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 99.0, 101.0, 98.0, 98.0),
            bar(30, 97.0, 99.0, 95.0, 96.0),
            bar(45, 93.0, 94.0, 88.0, 89.0),
        ]
    )
    trade = simulate_position(
        SPEC, bars, 0, Signal(side=SHORT, stop=105.0, target=90.0)
    )
    assert trade.side == SHORT
    assert trade.entry_price == pytest.approx(98.5)
    assert trade.exit_price == pytest.approx(90.5)
    assert trade.exit_reason == "target"
    assert trade.gross_pnl == pytest.approx(8.0)
    assert trade.commissions == pytest.approx(0.0189)


def test_roll_transfer_hides_gap_and_charges_cost():
    bars = make_bars(
        [
            bar(0, 99.5, 100.5, 99.0, 100.0),
            bar(15, 100.0, 100.5, 99.5, 100.0),
            bar(30, 10.0, 10.6, 9.9, 10.5, adjustment=90.0, roll=True),
            bar(45, 10.5, 11.0, 10.4, 11.0, adjustment=90.0),
            bar(60, 11.0, 11.2, 10.8, 11.0, adjustment=90.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(target=200.0, max_bars=4))
    assert trade.entry_price == pytest.approx(100.5)
    assert trade.exit_price == pytest.approx(10.5)
    assert trade.exit_reason == "time"
    assert trade.bars_held == 4
    assert trade.gross_pnl == pytest.approx(0.0)
    assert trade.roll_costs == pytest.approx(1.011)
    assert trade.commissions == pytest.approx(0.0111)
    assert trade.net_pnl == pytest.approx(-1.0221)


def test_roll_keeps_overnight_gap_through_stop():
    # Базис 90 убран поправкой, но за ночь рынок упал на 5 пунктов
    # (100 -> 95 в adjusted) и выбил стоп 97: убыток гэпа остаётся в P&L.
    bars = make_bars(
        [
            bar(0, 99.5, 100.5, 99.0, 100.0),
            bar(15, 100.0, 100.5, 99.5, 100.0),
            bar(30, 5.0, 5.5, 4.0, 5.0, adjustment=90.0, roll=True),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(stop=97.0))
    assert trade.exit_reason == "stop"
    assert trade.bars_held == 2
    assert trade.exit_price == pytest.approx(4.5)
    assert trade.gross_pnl == pytest.approx(-6.0)
    assert trade.roll_costs == pytest.approx(1.0105)


def test_roll_keeps_overnight_gain():
    bars = make_bars(
        [
            bar(0, 99.5, 100.5, 99.0, 100.0),
            bar(15, 100.0, 100.5, 99.5, 100.0),
            bar(30, 13.0, 14.5, 12.5, 14.0, adjustment=90.0, roll=True),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(max_bars=2))
    assert trade.exit_reason == "time"
    assert trade.exit_price == pytest.approx(13.5)
    assert trade.gross_pnl == pytest.approx(3.0)
    assert trade.roll_costs == pytest.approx(1.0113)


def test_negative_adjusted_prices_do_not_change_result():
    # Аддитивная поправка может увести adjusted-цены ниже нуля (как у NG).
    # Результат сделки от уровня поправки не зависит.
    rows = [
        (0, 99.0, 101.0, 98.0, 100.0),
        (15, 101.0, 103.0, 100.0, 102.0),
        (30, 102.0, 106.0, 101.0, 105.0),
        (45, 108.0, 111.0, 107.0, 110.0),
    ]
    plain = simulate_position(
        SPEC, make_bars([bar(*row) for row in rows]), 0, long_signal()
    )
    shifted = simulate_position(
        SPEC,
        make_bars([bar(*row, adjustment=-150.0) for row in rows]),
        0,
        long_signal(),
    )
    assert shifted.entry_price == pytest.approx(plain.entry_price)
    assert shifted.exit_price == pytest.approx(plain.exit_price)
    assert shifted.exit_reason == plain.exit_reason
    assert shifted.gross_pnl == pytest.approx(plain.gross_pnl)
    assert shifted.net_pnl == pytest.approx(plain.net_pnl)


def test_future_perturbation_does_not_change_result():
    rows = [
        bar(0, 99.0, 101.0, 98.0, 100.0),
        bar(15, 101.0, 103.0, 100.0, 102.0),
        bar(30, 102.0, 111.0, 101.0, 108.0),
        bar(45, 108.0, 109.0, 107.0, 108.0),
        bar(60, 108.0, 110.0, 107.0, 109.0),
        bar(75, 109.0, 112.0, 108.0, 111.0),
    ]
    trade = simulate_position(SPEC, make_bars(rows), 0, long_signal())
    perturbed = [
        {
            **row,
            "open": row["open"] * 2,
            "high": row["high"] * 2,
            "low": row["low"] * 2,
            "close": row["close"] * 2,
        }
        if index > 2
        else row
        for index, row in enumerate(rows)
    ]
    trade_perturbed = simulate_position(SPEC, make_bars(perturbed), 0, long_signal())
    assert trade == trade_perturbed


def test_end_of_data_closes_position():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 103.0, 104.0, 102.0, 103.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(target=200.0))
    assert trade.exit_reason == "end"
    assert trade.exit_price == pytest.approx(102.5)


def test_open_position_without_end_exit_is_rejected():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 103.0, 104.0, 102.0, 103.0),
        ]
    )
    with pytest.raises(ValueError, match="осталась открытой"):
        simulate_position(SPEC, bars, 0, long_signal(target=200.0), exit_at_end=False)


@pytest.mark.parametrize("signal_bar", [-1, 2])
def test_signal_needs_a_next_bar(signal_bar):
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 103.0, 104.0, 102.0, 103.0),
        ]
    )
    with pytest.raises(ValueError, match="следующий бар"):
        simulate_position(SPEC, bars, signal_bar, long_signal())


@pytest.mark.parametrize(
    "signal",
    [
        Signal(side=LONG, stop=100.0, target=110.0),
        Signal(side=LONG, stop=95.0, target=100.0),
        Signal(side=SHORT, stop=100.0, target=90.0),
        Signal(side=SHORT, stop=105.0, target=100.0),
    ],
)
def test_levels_must_bracket_signal_close(signal):
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
        ]
    )
    with pytest.raises(ValueError, match="требуется"):
        simulate_position(SPEC, bars, 0, signal)


def test_missing_columns_rejected():
    bars = pl.DataFrame({"timestamp": [datetime(2026, 9, 10)], "open": [100.0]})
    with pytest.raises(ValueError, match="нет столбцов"):
        simulate_position(SPEC, bars, 0, long_signal())


def test_long_gap_up_past_target_fills_target_at_open_not_stop():
    bars = make_bars(
        [
            bar(0, 100.0, 101.0, 99.0, 100.0),
            bar(15, 115.0, 120.0, 85.0, 90.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(stop=90.0, target=110.0))
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(114.5)
    assert trade.gross_pnl == pytest.approx(-1.0)
    assert trade.net_pnl == pytest.approx(-1.0 - 0.023)


def test_short_gap_down_past_target_fills_target_at_open_not_stop():
    bars = make_bars(
        [
            bar(0, 100.0, 101.0, 99.0, 100.0),
            bar(15, 85.0, 115.0, 80.0, 110.0),
        ]
    )
    trade = simulate_position(
        SPEC, bars, 0, Signal(side=SHORT, stop=110.0, target=90.0)
    )
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(85.5)
    assert trade.gross_pnl == pytest.approx(-1.0)
    assert trade.net_pnl == pytest.approx(-1.0 - 0.017)


def test_entry_open_gap_through_stop_exits_on_entry_bar():
    bars = make_bars(
        [
            bar(0, 100.0, 101.0, 99.0, 100.0),
            bar(15, 90.0, 91.0, 89.0, 90.5),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(stop=95.0, target=110.0))
    assert trade.exit_reason == "stop"
    assert trade.bars_held == 1
    assert trade.exit_price == pytest.approx(89.5)
    assert trade.gross_pnl == pytest.approx(-1.0)
