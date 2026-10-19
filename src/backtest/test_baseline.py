"""Проверки «купил и держи». Запуск: uv run pytest src/backtest/test_baseline.py"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest.baseline import (
    buy_and_hold,
    buy_and_hold_result,
    free_spec,
    nav_comparison,
    nav_summary,
    roll_costs_bps,
)
from backtest.report import build_report, market_timeline
from backtest.specs import ContractSpec
from data.build_continuous_futures import add_etf_nav

SPEC = ContractSpec(asset="TEST", tick_size=0.5, exchange_fee_rate=0.0001)
START = datetime(2026, 9, 10, 10)


def series_with_rolls(seed: int = 5) -> pl.DataFrame:
    """40 свечей, переносы на 10-й и 25-й, рубли за пункт меняются по свечам.

    Скорректированная цена — случайное блуждание; поправка меняется на
    переносе, реальная цена = скорректированная − поправка. NAV считает
    функция склейки `add_etf_nav`, а не тест.
    """
    rng = random.Random(seed)
    adjusted = 100.0
    rows = []
    for k in range(40):
        adjustment = 0.0 if k < 10 else (-12.0 if k < 25 else 7.0)
        open_ = adjusted + rng.uniform(-1.0, 1.0)
        close = open_ + rng.uniform(-1.5, 1.5)
        high = max(open_, close) + rng.uniform(0.0, 0.5)
        low = min(open_, close) - rng.uniform(0.0, 0.5)
        rows.append(
            {
                "timestamp": START + timedelta(minutes=15 * k),
                "contract": "A" if k < 10 else ("B" if k < 25 else "C"),
                "open": open_ - adjustment,
                "high": high - adjustment,
                "low": low - adjustment,
                "close": close - adjustment,
                "close_adjusted": close,
                "price_adjustment": adjustment,
                "is_first_bar_after_roll": k in (10, 25),
                "point_value": rng.choice([1.0, 2.0, 3.0]),
            }
        )
        adjusted = close
    return add_etf_nav(pl.DataFrame(rows))


@pytest.mark.parametrize("contracts", [1, 3])
def test_simulator_matches_nav_on_every_bar(contracts):
    bars = series_with_rolls()
    trade = buy_and_hold(free_spec(SPEC), bars, contracts=contracts)
    assert trade.exit_reason == "end"
    comparison = nav_comparison(trade, bars)
    # Все свечи после входа, включая обе свечи переноса.
    assert comparison.height == bars.height - 2
    assert comparison["is_first_bar_after_roll"].sum() == 2
    assert comparison["difference"].max() < 1e-12


def test_roll_costs_in_bps_match_simulator():
    bars = series_with_rolls()
    rolls = roll_costs_bps(SPEC, bars)
    assert rolls["contract"].to_list() == ["B", "C"]
    # Перенос: 2 шага по 0.5 × рубли за пункт + сборы закрытия и открытия.
    first = bars.row(10, named=True)
    old_close = bars["close"][9]
    pv = first["point_value"]
    cost = 2 * 0.5 * pv + (old_close + first["open"]) * pv * 0.0001
    assert rolls["cost_rub"][0] == pytest.approx(cost)
    assert rolls["cost_bps"][0] == pytest.approx(cost / (first["open"] * pv) * 10_000)
    # Симулятор с издержками платит за переносы ровно столько же.
    trade = buy_and_hold(SPEC, bars)
    assert trade.roll_costs == pytest.approx(rolls["cost_rub"].sum())


def test_nav_summary():
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1),
                datetime(2025, 6, 1),
                datetime(2026, 1, 1),
            ],
            "nav": [1.0, 1.2, 1.1],
        }
    )
    summary = nav_summary(bars)
    assert summary["total_return"] == pytest.approx(0.1)
    # Пик 1.2 → 1.1: −8.3%.
    assert summary["max_drawdown"] == pytest.approx(1.1 / 1.2 - 1)
    assert summary["cagr"] == pytest.approx(1.1 ** (365.25 / 366) - 1)


def test_buy_and_hold_result_fills_capital_and_feeds_report():
    bars = series_with_rolls()
    capital = 10_000.0
    result = buy_and_hold_result(SPEC, bars, capital)
    item = result.trades[0]
    notional = bars["close"][0] * bars["point_value"][0]
    # Плечо 1×: контрактов столько, сколько помещается в капитал.
    assert item.size.contracts == int(capital // notional)
    assert item.trade.contracts == item.size.contracts
    assert item.trade.exit_reason == "end"
    assert result.final_capital == pytest.approx(capital + item.trade.net_pnl)
    # 1R — весь номинал: результат в R — доля номинала.
    assert item.r_multiple == pytest.approx(
        item.trade.net_pnl / (item.size.contracts * notional)
    )

    report = build_report(result, market_timeline([bars]))
    s = report.summary
    assert s["end_capital"] == pytest.approx(result.final_capital)
    # Сделку закрыл конец данных: в деньгах есть, в проценте выигрышных — нет.
    assert s["trades"] == 0
    assert s["net_pnl"] == pytest.approx(item.trade.net_pnl)
    # В позиции все свечи, кроме сигнальной.
    assert s["exposure"] == pytest.approx((bars.height - 1) / bars.height)


def test_buy_and_hold_survives_adjusted_prices_below_zero():
    # Как у газа: цена падает, новый контракт дороже старого. Поправка уводит
    # adjusted-цены ниже нуля, реальные остаются выше. Стоп «у нуля» сработал
    # бы здесь, стопа нет — выход концом данных.
    rows = []
    for k in range(20):
        real = 10.0 - 0.2 * k if k < 10 else 17.2 - 1.5 * (k - 10)
        adjustment = 0.0 if k < 10 else -9.0
        rows.append(
            {
                "timestamp": START + timedelta(minutes=15 * k),
                "contract": "A" if k < 10 else "B",
                "open": real,
                "high": real + 0.1,
                "low": real - 0.1,
                "close": real,
                "close_adjusted": real + adjustment,
                "price_adjustment": adjustment,
                "is_first_bar_after_roll": k == 10,
                "point_value": 1.0,
            }
        )
    bars = pl.DataFrame(rows)
    assert (bars["close"] + bars["price_adjustment"]).min() < 0 < bars["low"].min()
    trade = buy_and_hold(free_spec(SPEC), bars, contracts=2)
    assert trade.exit_reason == "end"
    # Вход по открытию второй свечи (9.8), выход по последнему adjusted
    # закрытию (3.7 − 9 = −5.3): (−5.3 − 9.8) × 2 контракта.
    assert trade.net_pnl == pytest.approx((-5.3 - 9.8) * 2)
