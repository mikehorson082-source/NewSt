"""Проверки симулятора одной позиции. Запуск: uv run pytest src/backtest/test_position.py"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest.markets import MARKETS, add_point_value
from backtest.minutes import RawMinutes
from backtest.position import LONG, SHORT, Signal, simulate_position
from backtest.specs import ContractSpec

# Рубли за пункт приходят из баров; в большинстве тестов 1 ₽ за пункт,
# поэтому результат в рублях равен результату в пунктах.
SPEC = ContractSpec(asset="TEST", tick_size=0.5, exchange_fee_rate=0.0001)
START = datetime(2026, 9, 10, 10)


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
    contract: str = "TSZ6",
    point_value: float | None = 1.0,
) -> dict:
    return {
        "timestamp": START + timedelta(minutes=minute),
        "contract": contract,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "price_adjustment": adjustment,
        "is_first_bar_after_roll": roll,
        "point_value": point_value,
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
    assert trade.exit_resolution == "bar"
    assert trade.bars_held == 3
    assert trade.gross_pnl == pytest.approx(8.0)
    assert trade.commissions == pytest.approx(0.0211)
    assert trade.roll_costs == 0.0
    assert trade.net_pnl == pytest.approx(8.0 - 0.0211)


def test_stop_has_priority_in_bar_with_both_levels_without_minutes():
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
    assert trade.exit_resolution == "no_minutes"
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


# Рубли за пункт по барам: Brent — $10 за пункт × курс клиринга этого бара.
# Курс 80 — 800 ₽ за пункт, курс 90 — 900 ₽. Бар 1440 — следующий день.

BRENT_NO_COSTS = ContractSpec(
    asset="BRENT", tick_size=0.01, exchange_fee_rate=0.0, slippage_ticks=0.0
)


def test_profit_uses_point_value_of_each_bar():
    # +$1 в первый день при курсе 80 и ещё +$1 во второй при курсе 90.
    bars = make_bars(
        [
            bar(0, 70.0, 70.0, 70.0, 70.0, point_value=800.0),
            bar(15, 70.0, 71.0, 70.0, 71.0, point_value=800.0),
            bar(1440, 71.0, 72.0, 71.0, 72.0, point_value=900.0),
        ]
    )
    signal = long_signal(stop=60.0, target=80.0, max_bars=2)
    trade = simulate_position(BRENT_NO_COSTS, bars, 0, signal)
    assert trade.gross_pnl == pytest.approx(1 * 800 + 1 * 900)
    assert trade.net_pnl == pytest.approx(1_700.0)

    # «Одно число» по последнему курсу переоценило бы и первый день.
    single = bars.with_columns(pl.lit(900.0).alias("point_value"))
    trade_single = simulate_position(BRENT_NO_COSTS, single, 0, signal)
    assert trade_single.gross_pnl == pytest.approx(2 * 900)


def test_flat_price_with_rising_rate_gives_zero_profit():
    # Рублёвая стоимость контракта выросла с 56 000 до 63 000 ₽, но биржа
    # платит только за изменение цены: вариационная маржа равна нулю.
    bars = make_bars(
        [
            bar(0, 70.0, 70.0, 70.0, 70.0, point_value=800.0),
            bar(15, 70.0, 70.0, 70.0, 70.0, point_value=800.0),
            bar(1440, 70.0, 70.0, 70.0, 70.0, point_value=850.0),
            bar(2880, 70.0, 70.0, 70.0, 70.0, point_value=900.0),
        ]
    )
    signal = long_signal(stop=60.0, target=80.0, max_bars=3)
    trade = simulate_position(BRENT_NO_COSTS, bars, 0, signal)
    assert trade.bars_held == 3
    assert trade.gross_pnl == pytest.approx(0.0)
    assert trade.net_pnl == pytest.approx(0.0)


def test_costs_use_point_value_of_their_bar():
    # Вход при 800 ₽ за пункт, перенос при 850, выход при 900.
    # Новый контракт дешевле старого на $1 (поправка 1.0).
    fee = 0.000132
    spec = ContractSpec(asset="BRENT", tick_size=0.01, exchange_fee_rate=fee)
    bars = make_bars(
        [
            bar(0, 70.00, 70.10, 69.90, 70.00, point_value=800.0),
            bar(15, 70.00, 70.60, 69.95, 70.50, point_value=800.0),
            bar(
                1440,
                69.00,
                69.60,
                68.95,
                69.50,
                adjustment=1.0,
                roll=True,
                contract="TSH7",
                point_value=850.0,
            ),
            bar(
                2880,
                69.50,
                70.10,
                69.45,
                70.00,
                adjustment=1.0,
                contract="TSH7",
                point_value=900.0,
            ),
        ]
    )
    signal = long_signal(stop=65.0, target=75.0, max_bars=3)
    trade = simulate_position(spec, bars, 0, signal)
    assert trade.entry_price == pytest.approx(70.01)
    assert trade.exit_price == pytest.approx(69.99)
    # Вход 70.01 → закрытие 70.50 при 800; перенос без движения; 70.50 → 70.99
    # (выход 69.99 + поправка 1.0) при 900.
    assert trade.gross_pnl == pytest.approx(0.49 * 800 + 0.49 * 900)
    assert trade.commissions == pytest.approx((70.01 * 800 + 69.99 * 900) * fee)
    # Две ноги проскальзывания по $0.01 и две комиссии — всё при 850.
    expected_roll = 2 * 0.01 * 850 + (70.50 + 69.00) * 850 * fee
    assert trade.roll_costs == pytest.approx(expected_roll)


@pytest.mark.parametrize("name", ["cnyrub", "mix"])
def test_rub_markets_keep_single_point_value_result(name):
    # У рублёвых рынков рубли за пункт постоянны (CNY 1000 ₽, MIX 1 ₽):
    # результат по барам совпадает с прежней формулой «одно число k».
    # Шаг цены условный.
    market = MARKETS[name]
    k = market.quote_point_value
    fee = market.exchange_fee_rate
    bars = add_point_value(
        make_bars(
            [
                bar(0, 99.5, 100.5, 99.0, 100.0),
                bar(15, 100.0, 100.5, 99.5, 100.0),
                bar(30, 13.0, 14.5, 12.5, 14.0, adjustment=90.0, roll=True),
            ]
        ),
        market,
    )
    spec = ContractSpec(asset=name, tick_size=0.5, exchange_fee_rate=fee)
    trade = simulate_position(spec, bars, 0, long_signal(contracts=3, max_bars=2))
    assert trade.entry_price == pytest.approx(100.5)
    assert trade.exit_price == pytest.approx(13.5)
    assert trade.gross_pnl == pytest.approx((13.5 + 90.0 - 100.5) * k * 3)
    assert trade.commissions == pytest.approx((100.5 + 13.5) * k * 3 * fee)
    assert trade.roll_costs == pytest.approx(
        2 * 0.5 * k * 3 + (100.0 + 13.0) * k * 3 * fee
    )


@pytest.mark.parametrize("point_value", [None, 0.0])
def test_bar_without_point_value_is_rejected(point_value):
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0, point_value=point_value),
        ]
    )
    with pytest.raises(ValueError, match="point_value"):
        simulate_position(SPEC, bars, 0, long_signal())


# Бар задел и стоп, и цель: порядок касаний берётся из минуток того же контракта.


def write_minutes(
    directory: Path,
    contract: str,
    bar_minute: int,
    prices: list[tuple[float, float, float, float]],
) -> RawMinutes:
    """Минутки одного 15-минутного бара в формате raw: (open, high, low, close)."""
    start = START + timedelta(minutes=bar_minute)
    pl.DataFrame(
        [
            {
                "timestamp": start + timedelta(minutes=i),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
            for i, (open_, high, low, close) in enumerate(prices)
        ]
    ).write_parquet(directory / f"{contract}.parquet")
    return RawMinutes(directory)


def helicopter_bars() -> pl.DataFrame:
    """Сигнал на 100; бар входа задел и стоп 95, и цель 110."""
    return make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 111.0, 94.0, 100.0),
        ]
    )


def test_minutes_show_target_was_first(tmp_path):
    minutes = write_minutes(
        tmp_path,
        "TSZ6",
        15,
        [
            (101.0, 105.0, 100.0, 104.0),
            (104.0, 111.0, 103.0, 108.0),  # 10:16 — цель 110
            (108.0, 108.0, 94.0, 96.0),  # 10:17 — стоп 95, уже поздно
            (96.0, 101.0, 95.5, 100.0),
        ],
    )
    trade = simulate_position(
        SPEC, helicopter_bars(), 0, long_signal(), minutes=minutes
    )
    assert trade.exit_reason == "target"
    assert trade.exit_resolution == "minutes"
    assert trade.exit_price == pytest.approx(109.5)
    assert trade.gross_pnl == pytest.approx(8.0)


def test_minutes_show_stop_was_first(tmp_path):
    minutes = write_minutes(
        tmp_path,
        "TSZ6",
        15,
        [
            (101.0, 102.0, 100.0, 100.5),
            (100.5, 100.5, 94.0, 95.5),  # 10:16 — стоп 95
            (95.5, 111.0, 95.5, 108.0),  # 10:17 — цель 110, уже поздно
            (108.0, 108.0, 99.0, 100.0),
        ],
    )
    trade = simulate_position(
        SPEC, helicopter_bars(), 0, long_signal(), minutes=minutes
    )
    assert trade.exit_reason == "stop"
    assert trade.exit_resolution == "minutes"
    assert trade.exit_price == pytest.approx(94.5)
    assert trade.gross_pnl == pytest.approx(-7.0)


def test_both_levels_in_one_minute_fall_back_to_stop(tmp_path):
    minutes = write_minutes(
        tmp_path,
        "TSZ6",
        15,
        [
            (101.0, 102.0, 100.0, 101.0),
            (101.0, 111.0, 94.0, 100.0),  # 10:16 — и стоп, и цель
        ],
    )
    trade = simulate_position(
        SPEC, helicopter_bars(), 0, long_signal(), minutes=minutes
    )
    assert trade.exit_reason == "stop"
    assert trade.exit_resolution == "same_minute"
    assert trade.exit_price == pytest.approx(94.5)


def test_minutes_not_matching_bar_fall_back_to_stop(tmp_path):
    # Максимум минуток 109, а у бара 111: минутки описывают не этот бар.
    minutes = write_minutes(tmp_path, "TSZ6", 15, [(101.0, 109.0, 94.0, 100.0)])
    trade = simulate_position(
        SPEC, helicopter_bars(), 0, long_signal(), minutes=minutes
    )
    assert trade.exit_reason == "stop"
    assert trade.exit_resolution == "minutes_mismatch"
    assert trade.exit_price == pytest.approx(94.5)


def test_minute_opening_beyond_stop_exits_at_its_open(tmp_path):
    # 10:16 открылась на 93, сразу за стопом 95: как и у 15-минутного бара,
    # выход по цене гэпа, хотя позже в той же минутке цена дошла до цели.
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 111.0, 93.0, 108.0),
        ]
    )
    minutes = write_minutes(
        tmp_path,
        "TSZ6",
        15,
        [(101.0, 102.0, 100.0, 100.0), (93.0, 111.0, 93.0, 108.0)],
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(), minutes=minutes)
    assert trade.exit_reason == "stop"
    assert trade.exit_resolution == "minutes"
    assert trade.exit_price == pytest.approx(92.5)
    assert trade.gross_pnl == pytest.approx(-9.0)


def test_short_minutes_show_target_was_first(tmp_path):
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 99.0, 106.0, 89.0, 100.0),
        ]
    )
    minutes = write_minutes(
        tmp_path,
        "TSZ6",
        15,
        [
            (99.0, 100.0, 95.0, 96.0),
            (96.0, 96.0, 89.0, 91.0),  # 10:16 — цель 90
            (91.0, 106.0, 91.0, 104.0),  # 10:17 — стоп 105, уже поздно
            (104.0, 104.0, 99.0, 100.0),
        ],
    )
    trade = simulate_position(
        SPEC, bars, 0, Signal(side=SHORT, stop=105.0, target=90.0), minutes=minutes
    )
    assert trade.exit_reason == "target"
    assert trade.exit_resolution == "minutes"
    assert trade.exit_price == pytest.approx(90.5)
    assert trade.gross_pnl == pytest.approx(8.0)


def test_minutes_after_roll_use_new_contract_and_adjustment(tmp_path):
    # Новый контракт дешевле на 90: стоп 95 и цель 110 в его ценах — 5 и 20.
    # Файл минуток есть только у нового контракта TSH7.
    bars = make_bars(
        [
            bar(0, 99.5, 100.5, 99.0, 100.0),
            bar(15, 100.0, 100.5, 99.5, 100.0),
            bar(30, 10.0, 21.0, 4.0, 10.0, adjustment=90.0, roll=True, contract="TSH7"),
        ]
    )
    minutes = write_minutes(
        tmp_path,
        "TSH7",
        30,
        [
            (10.0, 15.0, 9.0, 14.0),
            (14.0, 21.0, 13.0, 18.0),  # 10:31 — цель 20
            (18.0, 18.0, 4.0, 10.0),  # 10:32 — стоп 5, уже поздно
        ],
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(), minutes=minutes)
    assert trade.exit_reason == "target"
    assert trade.exit_resolution == "minutes"
    assert trade.exit_price == pytest.approx(19.5)
    assert trade.gross_pnl == pytest.approx(9.0)
    assert trade.roll_costs == pytest.approx(1.011)


def test_minutes_are_read_only_for_bars_with_both_levels(tmp_path):
    # Каталог минуток пуст: бары, задевшие один уровень, минутки не запрашивают.
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 108.0, 111.0, 107.0, 110.0),
        ]
    )
    trade = simulate_position(
        SPEC, bars, 0, long_signal(), minutes=RawMinutes(tmp_path)
    )
    assert trade.exit_reason == "target"
    assert trade.exit_resolution == "bar"


def test_missing_minutes_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="TSZ6"):
        simulate_position(
            SPEC, helicopter_bars(), 0, long_signal(), minutes=RawMinutes(tmp_path)
        )


def test_minutes_need_contract_column(tmp_path):
    bars = helicopter_bars().drop("contract")
    with pytest.raises(ValueError, match="contract"):
        simulate_position(SPEC, bars, 0, long_signal(), minutes=RawMinutes(tmp_path))


def random_minute_path(rng: random.Random) -> list[tuple[float, float, float, float]]:
    """15 минуток от цены 100; изредка гэп между минутками."""
    path = []
    close = 100.0
    for minute in range(15):
        open_ = close
        if minute > 0 and rng.random() < 0.2:
            open_ += rng.choice([-2.0, 2.0])
        close = open_ + rng.choice([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        high = max(open_, close) + rng.choice([0.0, 0.5, 1.0, 2.0])
        low = min(open_, close) - rng.choice([0.0, 0.5, 1.0, 2.0])
        path.append((open_, high, low, close))
    return path


def test_minute_resolution_matches_full_minute_simulation(tmp_path):
    # Разбор бара по минуткам должен дать тот же выход, что и прогон
    # симулятора прямо по минутным свечам. Случайные пути, в которых
    # 15-минутный бар задел и стоп, и цель (98 и 102 при входе на 100).
    rng = random.Random(7)
    resolutions = set()
    checked = 0
    while checked < 200:
        path = random_minute_path(rng)
        high = max(minute[1] for minute in path)
        low = min(minute[2] for minute in path)
        if high < 102.0 or low > 98.0:
            continue
        if rng.random() < 0.5:
            signal = Signal(side=LONG, stop=98.0, target=102.0)
        else:
            signal = Signal(side=SHORT, stop=102.0, target=98.0)

        contract = f"TS{checked}"
        signal_bar = bar(0, 100.0, 100.0, 100.0, 100.0, contract=contract)
        bar_15m = bar(15, path[0][0], high, low, path[-1][3], contract=contract)
        minutes = write_minutes(tmp_path, contract, 15, path)
        by_minutes = simulate_position(
            SPEC, make_bars([signal_bar, bar_15m]), 0, signal, minutes=minutes
        )
        bars_1m = make_bars(
            [signal_bar] + [bar(15 + i, *prices) for i, prices in enumerate(path)]
        )
        reference = simulate_position(SPEC, bars_1m, 0, signal)

        assert by_minutes.exit_reason == reference.exit_reason
        assert by_minutes.exit_price == pytest.approx(reference.exit_price)
        assert by_minutes.gross_pnl == pytest.approx(reference.gross_pnl)
        resolutions.add(by_minutes.exit_resolution)
        checked += 1
    assert resolutions == {"minutes", "same_minute"}


def test_path_mae_mfe_and_slippage_of_target_trade():
    # Вход 101 + 0.5 проскальзывания = 101.5, выход по цели 110 − 0.5 = 109.5.
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 101.0, 103.0, 100.0, 102.0),
            bar(30, 102.0, 106.0, 101.0, 105.0),
            bar(45, 108.0, 111.0, 107.0, 110.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal())
    # Проскальзывание отдельной строкой: по 0.5 на вход и выход.
    assert trade.slippage == pytest.approx(1.0)
    # По ценам рынка 101 → 110: 9 пунктов; по ценам исполнения 8.
    assert trade.pnl_before_costs == pytest.approx(9.0)
    assert trade.costs == pytest.approx(1.0 + 0.0211)
    assert trade.pnl_before_costs - trade.costs == pytest.approx(trade.net_pnl)
    # Путь: результат после издержек на закрытии каждого бара, на выходе — итог.
    assert [ts for ts, _ in trade.path] == [
        START + timedelta(minutes=m) for m in (15, 30, 45)
    ]
    entry_fee = 101.5 * 0.0001
    assert [value for _, value in trade.path] == pytest.approx(
        [0.5 - entry_fee, 3.5 - entry_fee, trade.net_pnl]
    )
    # Худшее — low 100 бара входа: 100 − 101.5. Лучшее — цель 110 в баре
    # выхода: high 111 мог случиться уже после выхода и не считается.
    assert trade.mae == pytest.approx(-1.5)
    assert trade.mfe == pytest.approx(8.5)


def test_mfe_ignores_high_of_stop_bar():
    # Бар выхода: high 104, потом стоп 95. Был ли high до стопа — неизвестно.
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 100.0, 101.0, 99.0, 100.0),
            bar(30, 100.0, 104.0, 94.0, 96.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal())
    assert trade.exit_reason == "stop"
    # Вход 100.5: лучшее — high 101 бара входа, худшее — стоп 95.
    assert trade.mfe == pytest.approx(0.5)
    assert trade.mae == pytest.approx(-5.5)


def test_time_exit_counts_whole_last_bar():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 100.0, 101.0, 99.0, 100.0),
            bar(30, 100.0, 104.0, 96.0, 100.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(max_bars=2))
    assert trade.exit_reason == "time"
    # Выход по закрытию: весь бар прошёл при открытой позиции.
    assert trade.mfe == pytest.approx(3.5)
    assert trade.mae == pytest.approx(-4.5)


def test_short_excursions_mirror_long():
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0),
            bar(15, 100.0, 103.0, 97.0, 100.0),
            bar(30, 100.0, 101.0, 99.0, 100.0),
        ]
    )
    signal = Signal(side=SHORT, stop=105.0, target=90.0, max_bars=2)
    trade = simulate_position(SPEC, bars, 0, signal)
    # Шорт по 99.5: рост до 103 — против нас, падение до 97 — в нашу пользу.
    assert trade.mae == pytest.approx(-3.5)
    assert trade.mfe == pytest.approx(2.5)


def test_mae_uses_point_value_of_each_bar():
    # 2 ₽ за пункт в баре входа, 3 ₽ — в следующем: просадка до 97 во втором
    # баре = (100 − 100.5) × 2 + (97 − 100) × 3 = −10 ₽.
    bars = make_bars(
        [
            bar(0, 99.0, 101.0, 98.0, 100.0, point_value=2.0),
            bar(15, 100.0, 100.0, 100.0, 100.0, point_value=2.0),
            bar(30, 100.0, 100.0, 97.0, 100.0, point_value=3.0),
        ]
    )
    trade = simulate_position(SPEC, bars, 0, long_signal(max_bars=2))
    assert trade.mae == pytest.approx(-10.0)


def random_trade_bars(rng: random.Random) -> pl.DataFrame:
    """Случайный ряд с переносами и рублями за пункт, меняющимися по барам."""
    rows = [bar(0, 99.5, 100.5, 99.0, 100.0)]
    adjusted = 100.0
    adjustment = 0.0
    for k in range(1, 40):
        roll = rng.random() < 0.1
        if roll:
            adjustment = rng.choice([-30.0, 0.0, 30.0])
        open_ = adjusted + rng.uniform(-1.5, 1.5)
        close = open_ + rng.uniform(-2.0, 2.0)
        high = max(open_, close) + rng.uniform(0.0, 1.5)
        low = min(open_, close) - rng.uniform(0.0, 1.5)
        rows.append(
            bar(
                15 * k,
                open_ - adjustment,
                high - adjustment,
                low - adjustment,
                close - adjustment,
                adjustment=adjustment,
                roll=roll,
                point_value=rng.choice([1.0, 2.0, 3.0]),
            )
        )
        adjusted = close
    return make_bars(rows)


def test_costs_are_charged_exactly_once():
    # Каждая издержка учтена ровно один раз: результат «до издержек» равен
    # результату той же сделки на рынке без издержек (выходы от издержек не
    # зависят). Путь кончается на net_pnl, MAE <= 0 <= MFE.
    free = ContractSpec(
        asset="TEST", tick_size=0.5, exchange_fee_rate=0.0, slippage_ticks=0.0
    )
    rng = random.Random(11)
    reasons = set()
    for _ in range(300):
        bars = random_trade_bars(rng)
        side = rng.choice((LONG, SHORT))
        distance = rng.uniform(1.0, 6.0)
        signal = Signal(
            side=side,
            stop=100.0 - side * distance,
            target=100.0 + side * rng.uniform(1.0, 8.0),
            contracts=rng.randint(1, 5),
            max_bars=rng.choice([None, 3, 10]),
        )
        trade = simulate_position(SPEC, bars, 0, signal)
        no_costs = simulate_position(free, bars, 0, signal)

        assert no_costs.costs == 0.0
        assert trade.exit_time == no_costs.exit_time
        assert trade.pnl_before_costs == pytest.approx(no_costs.net_pnl)
        assert trade.pnl_before_costs - trade.costs == pytest.approx(trade.net_pnl)
        assert len(trade.path) == trade.bars_held
        assert trade.path[0][0] == trade.entry_time
        assert trade.path[-1][0] == trade.exit_time
        assert trade.path[-1][1] == pytest.approx(trade.net_pnl)
        assert trade.mae <= 0.0 <= trade.mfe
        # После издержек результат ни в одной точке не лучше MFE.
        assert max(value for _, value in trade.path) <= trade.mfe + 1e-9
        reasons.add(trade.exit_reason)
    assert reasons == {"stop", "target", "time", "end"}
