"""Проверки сетки издержек. Запуск: uv run pytest src/backtest/test_sensitivity.py"""

# время в тестах — naive MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

from datetime import datetime

import polars as pl
import pytest

from backtest.portfolio import MarketData
from backtest.report import market_timeline
from backtest.sensitivity import (
    break_even,
    by_session,
    cost_grid,
    session_of,
    with_costs,
)
from backtest.specs import ContractSpec
from backtest.test_report import DAY1, DAY2, DAY3, at, signals, three_days

# Шаг 0.1, сбор 0.1% — заметный, чтобы множитель сбора что-то менял.
SPEC = ContractSpec(asset="a", tick_size=0.1, exchange_fee_rate=0.001)


def grid() -> pl.DataFrame:
    bars = three_days()
    return cost_grid(
        signals(at(DAY1, 10), at(DAY2, 10), at(DAY3, 10)),
        {"a": MarketData(spec=SPEC, bars=bars)},
        market_timeline([bars]),
        slippage_ticks=(0.0, 1.0, 5.0, 20.0),
        fee_multipliers=(0.0, 1.0),
    )


def test_zero_costs_row_has_no_costs():
    free = grid().filter(
        (pl.col("slippage_ticks") == 0) & (pl.col("fee_multiplier") == 0)
    )
    row = free.row(0, named=True)
    assert row["costs"] == 0.0
    assert row["net_pnl"] == pytest.approx(row["pnl_before_costs"])
    # Сделки трёх дней из test_report: +20 000, +20 400, −10 400 ₽.
    assert row["net_pnl"] == pytest.approx(30_000.0)


def test_result_falls_as_costs_grow():
    table = grid()
    for fee in (0.0, 1.0):
        line = table.filter(pl.col("fee_multiplier") == fee).sort("slippage_ticks")
        assert line["net_pnl"].is_sorted(descending=True)
    # При том же проскальзывании сбор только ухудшает результат.
    by_fee = table.filter(pl.col("slippage_ticks") == 1.0).sort("fee_multiplier")
    assert by_fee["net_pnl"][0] > by_fee["net_pnl"][1]


def test_grid_reruns_sizing():
    # Риск контракта включает проскальзывание: при 20 шагах (2 пункта на
    # сторону при стопе в 5) контрактов меньше, издержки не просто вычтены.
    table = grid()
    cheap = table.filter(
        (pl.col("slippage_ticks") == 0) & (pl.col("fee_multiplier") == 0)
    )
    dear = table.filter(
        (pl.col("slippage_ticks") == 20) & (pl.col("fee_multiplier") == 0)
    )
    # До издержек сделки те же, но контрактов меньше — результат меньше.
    assert dear["pnl_before_costs"][0] < cheap["pnl_before_costs"][0]


def test_break_even_interpolates():
    table = pl.DataFrame(
        {
            "slippage_ticks": [0.0, 1.0, 2.0],
            "fee_multiplier": [1.0, 1.0, 1.0],
            "net_pnl": [100.0, 50.0, -50.0],
        }
    )
    # Между 1 (+50) и 2 (−50) шагами ноль ровно посередине.
    assert break_even(table, value="net_pnl") == pytest.approx(1.5)


@pytest.mark.parametrize("values", [[-1.0, -2.0, -3.0], [3.0, 2.0, 1.0]])
def test_break_even_none_without_sign_change(values):
    table = pl.DataFrame(
        {
            "slippage_ticks": [0.0, 1.0, 2.0],
            "fee_multiplier": [1.0, 1.0, 1.0],
            "net_pnl": values,
        }
    )
    assert break_even(table, value="net_pnl") is None


def test_with_costs_keeps_original_markets():
    markets = {"a": MarketData(spec=SPEC, bars=three_days())}
    changed = with_costs(markets, slippage_ticks=3.0, fee_multiplier=2.0)
    assert changed["a"].spec.slippage_ticks == 3.0
    assert changed["a"].spec.exchange_fee_rate == pytest.approx(0.002)
    assert markets["a"].spec.slippage_ticks == 1.0
    assert markets["a"].spec.exchange_fee_rate == 0.001


def test_break_even_uses_expectancy_by_default():
    # Рубли упёрлись в «дно» (−1 000 и −1 000), R продолжает падать.
    table = pl.DataFrame(
        {
            "slippage_ticks": [0.0, 1.0, 2.0],
            "fee_multiplier": [1.0, 1.0, 1.0],
            "net_pnl": [10.0, -1_000.0, -1_000.0],
            "expectancy_r": [0.04, 0.01, -0.02],
        }
    )
    assert break_even(table) == pytest.approx(1 + 0.01 / 0.03)


def test_impact_rows_charge_impact():
    bars = three_days().with_columns(pl.lit(0.002).alias("bar_sigma"))
    table = cost_grid(
        signals(at(DAY1, 10), at(DAY2, 10), at(DAY3, 10)),
        {"a": MarketData(spec=SPEC, bars=bars)},
        market_timeline([bars]),
        slippage_ticks=(1.0,),
        fee_multipliers=(1.0,),
        impacts=(0.0, 1.0),
    )
    off, on = table.sort("impact").iter_rows(named=True)
    assert off["impact_costs"] == 0.0
    assert on["impact_costs"] > 0.0
    assert on["net_pnl"] < off["net_pnl"]


def test_session_boundaries():
    times = [
        datetime(2026, 9, 10, 9, 45),  # чт, утро
        datetime(2026, 9, 10, 10, 0),  # день
        datetime(2026, 9, 10, 18, 45),  # день
        datetime(2026, 9, 10, 19, 5),  # вечер
        datetime(2026, 9, 12, 12, 0),  # суббота
    ]
    frame = pl.DataFrame({"t": times}).select(session_of(pl.col("t")))
    assert frame.to_series().to_list() == ["утро", "день", "день", "вечер", "выходные"]


def test_by_session_groups_trades_and_skips_end():
    trades = pl.DataFrame(
        {
            "entry_time": [
                datetime(2026, 9, 10, 11),
                datetime(2026, 9, 10, 12),
                datetime(2026, 9, 10, 21),
                datetime(2026, 9, 10, 22),
            ],
            "exit_reason": ["target", "stop", "stop", "end"],
            "net_pnl": [200.0, -100.0, -350.0, 50.0],
            "r_multiple": [2.0, -1.0, -3.5, 0.5],
            "costs": [7.0, 7.0, 7.0, 7.0],
            "risk": [100.0, 100.0, 100.0, 100.0],
            "volume_share": [0.01, 0.02, 0.2, 0.3],
            "mae_r": [-0.2, -1.0, -3.4, -0.1],
        }
    )
    table = by_session(trades)
    assert table["session"].to_list() == ["день", "вечер"]
    day, evening = table.iter_rows(named=True)
    assert day["trades"] == 2
    assert day["expectancy_r"] == pytest.approx(0.5)
    # Сделка вечера, закрытая концом данных, не считается.
    assert evening["trades"] == 1
    assert evening["over_10pct"] == 1
    assert evening["gaps_over_1_5r"] == 1
