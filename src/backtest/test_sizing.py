"""Проверки размера позиции. Запуск: uv run pytest src/backtest/test_sizing.py"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest.position import LONG, SHORT, Signal
from backtest.sizing import MARGIN, RISK, VOLUME, SizingRules, size_position
from backtest.specs import ContractSpec

# Brent: шаг $0.01, проскальзывание 1 шаг, сбор 0.0132%.
FEE = 0.000132
BRENT = ContractSpec(asset="brent", tick_size=0.01, exchange_fee_rate=FEE)
CAPITAL = 1_000_000.0


def signal_bar(
    close: float = 70.0,
    point_value_known: float | None = 850.0,
    margin_rate: float | None = 0.25,
    point_value: float = 860.0,
    usual_volume: float | None = 10_000.0,
) -> pl.DataFrame:
    """Свеча сигнала и свеча входа после неё.

    Курс 85 известен, свечу рассчитает курс 86. Обычный объём свечи входа
    10 000 контрактов: лимит ликвидности 1 000, ни на что не влияет.
    """
    start = datetime(2026, 9, 10, 10)
    rows = [
        {
            "timestamp": start,
            "close": close,
            "point_value": point_value,
            "point_value_known": point_value_known,
            "margin_rate": margin_rate,
            "usual_volume": 10_000.0,
        },
        {
            "timestamp": start + timedelta(minutes=15),
            "close": close,
            "point_value": point_value,
            "point_value_known": point_value,
            "margin_rate": margin_rate,
            "usual_volume": usual_volume,
        },
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def long_signal(stop: float, target: float = 80.0) -> Signal:
    return Signal(side=LONG, stop=stop, target=target)


def risk_per_contract(close: float, stop: float, point_value: float = 850.0) -> float:
    """Расстояние до стопа и проскальзывание двух сторон плюс два сбора."""
    distance = abs(close - stop) + 2 * 0.01
    return distance * point_value + (close + stop) * point_value * FEE


def test_plan_example_risk_limit():
    # Стоп на $1 ниже, 850 ₽ за пункт: риск контракта ≈ 883 ₽ → 11 контрактов.
    size = size_position(BRENT, signal_bar(), 0, long_signal(69.0), CAPITAL)
    assert size.risk_per_contract == pytest.approx(risk_per_contract(70.0, 69.0))
    assert size.risk_per_contract == pytest.approx(882.6, abs=0.1)
    assert size.contracts == 11
    assert size.limit == RISK
    assert not size.skipped
    assert size.risk <= CAPITAL * 0.01
    # ГО контракта 70 × 850 × 25% = 14 875 ₽, позиции 163 625 ₽ — 16.4% капитала.
    assert size.margin_per_contract == pytest.approx(14_875.0)
    assert size.margin == pytest.approx(163_625.0)
    assert size.contracts_by_margin == 33


def test_cost_share_of_risk():
    # Издержки контракта: 2 шага по 850 ₽ = 17 ₽ плюс сборы ≈ 15.6 ₽ — 3.7% риска.
    size = size_position(BRENT, signal_bar(), 0, long_signal(69.0), CAPITAL)
    costs = 2 * 0.01 * 850.0 + (70.0 + 69.0) * 850.0 * FEE
    assert size.costs_per_contract == pytest.approx(costs)
    assert size.risk_per_contract == pytest.approx(1.0 * 850.0 + costs)
    assert size.cost_share == pytest.approx(costs / size.risk_per_contract)
    assert size.cost_share == pytest.approx(0.0369, abs=1e-4)
    # Дальний стоп — та же цена издержек, но меньшая доля риска.
    wide = size_position(BRENT, signal_bar(), 0, long_signal(65.0), CAPITAL)
    assert wide.cost_share < size.cost_share


def test_short_mirrors_long():
    signal = Signal(side=SHORT, stop=71.0, target=60.0)
    size = size_position(BRENT, signal_bar(), 0, signal, CAPITAL)
    assert size.risk_per_contract == pytest.approx(risk_per_contract(70.0, 71.0))
    assert size.contracts == 11


def test_uses_rate_known_at_signal_not_settlement_rate():
    # Курс, по которому биржа рассчитает свечу, в момент сигнала не известен.
    base = size_position(BRENT, signal_bar(), 0, long_signal(69.0), CAPITAL)
    other = size_position(
        BRENT, signal_bar(point_value=2_000.0), 0, long_signal(69.0), CAPITAL
    )
    assert other == base
    # А известный курс меняет размер: 1 000 ₽ за пункт → риск ≈ 1 038 ₽ → 9.
    higher = size_position(
        BRENT, signal_bar(point_value_known=1_000.0), 0, long_signal(69.0), CAPITAL
    )
    assert higher.contracts == 9


def test_risk_budget_follows_capital():
    size = size_position(BRENT, signal_bar(), 0, long_signal(69.0), 2 * CAPITAL)
    assert size.contracts == 22


def test_margin_limit_with_tight_stop():
    # Стоп в 5 центах: по риску можно 132 контракта, но их ГО 1.96 млн ₽.
    # Предел ГО 500 000 ₽ ÷ 14 875 ₽ = 33 контракта.
    size = size_position(BRENT, signal_bar(), 0, long_signal(69.95), CAPITAL)
    assert size.contracts_by_risk == 132
    assert size.contracts_by_margin == 33
    assert size.contracts == 33
    assert size.limit == MARGIN
    assert size.margin <= CAPITAL * 0.5


def test_margin_share_is_configurable():
    rules = SizingRules(max_margin_fraction=0.25)
    size = size_position(BRENT, signal_bar(), 0, long_signal(69.95), CAPITAL, rules)
    assert size.contracts == 16
    assert size.limit == MARGIN


def test_skip_when_one_contract_risks_more_than_budget():
    # Стоп в $12: один контракт рискует ≈ 10 230 ₽ при бюджете 10 000 ₽.
    size = size_position(BRENT, signal_bar(), 0, long_signal(58.0), CAPITAL)
    assert size.risk_per_contract > CAPITAL * 0.01
    assert size.contracts == 0
    assert size.skipped
    assert size.limit == RISK


def test_skip_when_one_contract_needs_more_margin_than_allowed():
    # Капитал 20 000 ₽: по риску 1 контракт (≈ 118 ₽ при бюджете 200 ₽),
    # но его ГО 14 875 ₽ больше половины капитала.
    size = size_position(BRENT, signal_bar(), 0, long_signal(69.9), 20_000.0)
    assert size.contracts_by_risk == 1
    assert size.contracts_by_margin == 0
    assert size.skipped
    assert size.limit == MARGIN


def test_exact_budget_is_not_lost_to_float_rounding():
    # 10.3 − 10.0 в float = 0.3000000000000007: риск контракта 300.0000000000007 ₽.
    # Бюджет 3 000 ₽ — ровно 10 контрактов, а не 9.
    spec = ContractSpec(
        asset="cnyrub", tick_size=0.001, exchange_fee_rate=0.0, slippage_ticks=0.0
    )
    bars = signal_bar(close=10.3, point_value_known=1_000.0, margin_rate=0.1)
    size = size_position(spec, bars, 0, long_signal(10.0, 11.0), 300_000.0)
    assert size.contracts == 10
    assert size.limit == RISK


def test_sized_signal_goes_to_simulator():
    size = size_position(BRENT, signal_bar(), 0, long_signal(69.0), CAPITAL)
    signal = replace(long_signal(69.0), contracts=size.contracts)
    assert signal.contracts == 11


@pytest.mark.parametrize(
    "signal",
    [
        Signal(side=LONG, stop=70.0, target=80.0),
        Signal(side=LONG, stop=71.0, target=80.0),
        Signal(side=SHORT, stop=69.0, target=60.0),
    ],
)
def test_stop_on_wrong_side_rejected(signal):
    with pytest.raises(ValueError, match="Стоп"):
        size_position(BRENT, signal_bar(), 0, signal, CAPITAL)


@pytest.mark.parametrize("column", ["point_value_known", "margin_rate"])
def test_missing_known_values_rejected(column):
    bars = signal_bar(**{column: None})
    with pytest.raises(ValueError, match=column):
        size_position(BRENT, bars, 0, long_signal(69.0), CAPITAL)


def test_missing_columns_rejected():
    bars = signal_bar().drop("margin_rate")
    with pytest.raises(ValueError, match="нет столбцов"):
        size_position(BRENT, bars, 0, long_signal(69.0), CAPITAL)


@pytest.mark.parametrize(
    ("signal_index", "capital", "message"),
    [
        (2, CAPITAL, "свечи сигнала"),
        (-1, CAPITAL, "свечи сигнала"),
        (0, 0.0, "Капитал"),
    ],
)
def test_bad_inputs_rejected(signal_index, capital, message):
    with pytest.raises(ValueError, match=message):
        size_position(BRENT, signal_bar(), signal_index, long_signal(69.0), capital)


def test_volume_limit_on_thin_entry_bar():
    # Обычно в это время свечи входа торгуют 50 контрактов (у свечи сигнала —
    # 10 000, но решает вход): 10% — 5, а по риску можно 11.
    size = size_position(
        BRENT, signal_bar(usual_volume=50.0), 0, long_signal(69.0), CAPITAL
    )
    assert size.contracts_by_risk == 11
    assert size.usual_volume == 50.0
    assert size.contracts_by_volume == 5
    assert size.contracts == 5
    assert size.limit == VOLUME
    # Риск плановый — по 5 контрактам, а не по 11.
    assert size.risk == pytest.approx(5 * size.risk_per_contract)


def test_skip_when_entry_bar_is_usually_empty():
    # MIX в 08:45: обычно 2 контракта, 10% — это 0.
    size = size_position(
        BRENT, signal_bar(usual_volume=2.0), 0, long_signal(69.0), CAPITAL
    )
    assert size.skipped
    assert size.limit == VOLUME


def test_unknown_usual_volume_skips_signal():
    # Меньше 5 дней истории — обычный объём неизвестен, ликвидность не доказана.
    size = size_position(
        BRENT, signal_bar(usual_volume=None), 0, long_signal(69.0), CAPITAL
    )
    assert size.usual_volume is None
    assert size.contracts_by_volume == 0
    assert size.skipped
    assert size.limit == VOLUME


def test_equal_limits_report_risk_first():
    # 110 контрактов обычного объёма → 11 по ликвидности, столько же по риску.
    size = size_position(
        BRENT, signal_bar(usual_volume=110.0), 0, long_signal(69.0), CAPITAL
    )
    assert size.contracts_by_risk == size.contracts_by_volume == 11
    assert size.limit == RISK


def test_volume_limit_can_be_switched_off():
    bars = signal_bar(usual_volume=2.0).drop("usual_volume").head(1)
    rules = SizingRules(max_volume_share=None)
    size = size_position(BRENT, bars, 0, long_signal(69.0), CAPITAL, rules)
    assert size.contracts == 11
    assert size.contracts_by_volume is None
    assert size.usual_volume is None


def test_volume_limit_needs_entry_bar_and_column():
    with pytest.raises(ValueError, match="свечи входа"):
        size_position(BRENT, signal_bar(), 1, long_signal(69.0), CAPITAL)
    with pytest.raises(ValueError, match="usual_volume"):
        size_position(
            BRENT, signal_bar().drop("usual_volume"), 0, long_signal(69.0), CAPITAL
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_volume_share": 0.0},
        {"max_volume_share": 1.5},
        {"risk_fraction": 0.0},
        {"risk_fraction": 1.0},
        {"max_margin_fraction": 0.0},
        {"max_margin_fraction": 1.5},
    ],
)
def test_rules_validation(kwargs):
    with pytest.raises(ValueError, match="Доля"):
        SizingRules(**kwargs)
