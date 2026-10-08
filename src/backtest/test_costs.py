"""Проверки спецификаций и издержек. Запуск: uv run pytest src/backtest/test_costs.py"""

import pytest

from backtest.costs import BUY, SELL, commission, fill_price, impact_points, roll_cost
from backtest.specs import ContractSpec

SPEC = ContractSpec(
    asset="TEST",
    tick_size=0.5,
    exchange_fee_rate=0.001,
    broker_fee_rate=0.0005,
    slippage_ticks=1.0,
)


def test_rates():
    assert SPEC.fee_rate == pytest.approx(0.0015)
    assert SPEC.slippage == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"tick_size": -1.0}, "tick_size"),
        ({"exchange_fee_rate": -0.001}, "комиссий"),
        ({"slippage_ticks": -1.0}, "slippage_ticks"),
    ],
)
def test_spec_validation_rejects_bad_values(kwargs, message):
    base = {"asset": "TEST", "tick_size": 0.5, "exchange_fee_rate": 0.0001}
    with pytest.raises(ValueError, match=message):
        ContractSpec(**{**base, **kwargs})


def test_fill_price_worsens_each_side():
    assert fill_price(SPEC, 100.0, BUY) == pytest.approx(100.5)
    assert fill_price(SPEC, 100.0, SELL) == pytest.approx(99.5)


def test_commission_is_share_of_notional():
    assert commission(SPEC, 100.0, 2.0, contracts=2) == pytest.approx(400.0 * 0.0015)


def test_roll_cost_is_two_legs_slippage_plus_two_commissions():
    cost = roll_cost(SPEC, old_price=100.0, new_price=10.0, point_value=2.0)
    expected = 2 * 0.5 * 2.0 + 100.0 * 2.0 * 0.0015 + 10.0 * 2.0 * 0.0015
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize(
    "function,args",
    [
        (fill_price, (SPEC, 0.0, BUY)),
        (fill_price, (SPEC, 100.0, 0)),
        (commission, (SPEC, -1.0, 2.0)),
        (commission, (SPEC, 100.0, 0.0)),
        (commission, (SPEC, 100.0, float("nan"))),
        (commission, (SPEC, 100.0, 2.0, 0)),
        (roll_cost, (SPEC, 0.0, 10.0, 2.0)),
        (roll_cost, (SPEC, 100.0, 10.0, -2.0)),
        (roll_cost, (SPEC, 100.0, 10.0, 2.0, 0)),
    ],
)
def test_invalid_cost_inputs_rejected(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_impact_points_square_root_of_share():
    spec = ContractSpec(asset="TEST", tick_size=0.01, exchange_fee_rate=0.0, impact=1.0)
    # Пример из описания: Brent $70, σ 0.22%, 25 контрактов в свечу на 50.
    assert impact_points(spec, 70.0, 25, 50, 0.0022) == pytest.approx(
        0.0022 * 70.0 * 0.5**0.5
    )
    # Вчетверо больше доля — вдвое больше влияние.
    assert impact_points(spec, 70.0, 100, 50, 0.0022) == pytest.approx(
        2 * impact_points(spec, 70.0, 25, 50, 0.0022)
    )


def test_impact_off_or_unknown_sigma_is_zero():
    assert impact_points(SPEC, 70.0, 25, 50, 0.0022) == 0.0
    on = ContractSpec(asset="TEST", tick_size=0.01, exchange_fee_rate=0.0, impact=1.0)
    assert impact_points(on, 70.0, 25, 50, None) == 0.0
    with pytest.raises(ValueError, match="объём"):
        impact_points(on, 70.0, 25, 0, 0.0022)


def test_fill_price_adds_impact_against_order():
    assert fill_price(SPEC, 100.0, BUY, impact=0.3) == pytest.approx(100.8)
    assert fill_price(SPEC, 100.0, SELL, impact=0.3) == pytest.approx(99.2)


def test_negative_impact_rejected():
    with pytest.raises(ValueError, match="impact"):
        ContractSpec(asset="TEST", tick_size=0.5, exchange_fee_rate=0.0, impact=-1.0)
