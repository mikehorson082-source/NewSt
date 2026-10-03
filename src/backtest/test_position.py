"""Проверки симулятора одной позиции. Запуск: uv run pytest src/backtest/test_position.py"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest.minutes import RawMinutes
from backtest.position import LONG, SHORT, Signal, simulate_position
from backtest.specs import ContractSpec

SPEC = ContractSpec(
    asset="TEST",
    point_value=1.0,
    tick_size=0.5,
    margin_rate=0.12,
    exchange_fee_rate=0.0001,
)
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
