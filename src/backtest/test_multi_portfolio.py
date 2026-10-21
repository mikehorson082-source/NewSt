"""Проверки портфеля с позицией на каждом рынке (plan.md, 5.5).

Запуск: uv run pytest src/backtest/test_multi_portfolio.py
"""

from datetime import timedelta

import polars as pl
import pytest

from backtest.portfolio import (
    BUSY,
    NO_CAPITAL,
    PORTFOLIO_MARGIN,
    TAKEN,
    ZERO_SIZE,
    MultiRules,
    run_multi_portfolio,
    run_portfolio,
)
from backtest.report import build_report, market_timeline
from backtest.test_portfolio import (
    CAPITAL,
    FREE,
    START,
    bar,
    flat_bars,
    make_bars,
    make_signals,
    market,
    signal,
    status_of,
    trend_bars,
)


def rules(**fractions: float) -> MultiRules:
    return MultiRules(risk_fractions=fractions)


def test_positions_on_different_markets_overlap():
    # Оба рынка доходят до цели 110 на свече 45. Риск a — 1%, b — 0.5%:
    # стоп в 5 пунктах → 2 000 и 1 000 контрактов.
    markets = {"a": market(FREE, trend_bars()), "b": market(FREE, trend_bars())}
    signals = make_signals([signal("a", 0, 95.0, 110.0), signal("b", 0, 95.0, 110.0)])
    result = run_multi_portfolio(signals, markets, CAPITAL, rules(a=0.01, b=0.005))

    assert status_of(result, "a", 0)["status"] == TAKEN
    assert status_of(result, "b", 0)["status"] == TAKEN
    a, b = result.trades
    assert (a.size.contracts, b.size.contracts) == (2_000, 1_000)
    assert a.trade.net_pnl == pytest.approx(20_000.0)
    assert b.trade.net_pnl == pytest.approx(10_000.0)
    assert result.final_capital == pytest.approx(1_030_000.0)

    report = build_report(result, market_timeline([trend_bars()]))
    curve = report.curve
    # На свече 30 обе сделки открыты: закрытие 102, вход 100.
    row = curve.filter(pl.col("timestamp") == START + timedelta(minutes=30)).row(
        0, named=True
    )
    assert row["positions"] == 2
    assert row["equity"] == pytest.approx(1_000_000.0 + 2_000 * 2 + 1_000 * 2)
    assert curve["positions"].max() == 2
    assert curve["equity"][-1] == pytest.approx(1_030_000.0)


def test_same_market_busy_until_its_exit_bar():
    signals = make_signals(
        [
            signal("a", 0, 95.0, 110.0),
            signal("a", 15, 95.0, 110.0),
            signal("a", 30, 95.0, 110.0),
            signal("a", 45, 105.0, 120.0),
        ]
    )
    result = run_multi_portfolio(
        signals, {"a": market(FREE, trend_bars())}, CAPITAL, rules(a=0.01)
    )
    for minute in (15, 30):
        row = status_of(result, "a", minute)
        assert row["status"] == BUSY
        assert row["trade_id"] == 0
    # Сделка закрылась на свече 45: сигнал по её закрытию исполняется.
    assert status_of(result, "a", 45)["status"] == TAKEN


def test_one_market_matches_single_position_portfolio():
    # С одним рынком оба портфеля обязаны дать одно и то же.
    signals = make_signals(
        [
            signal("a", 0, 95.0, 110.0),
            signal("a", 15, 95.0, 110.0),
            signal("a", 45, 105.0, 120.0),
        ]
    )
    markets = {"a": market(FREE, trend_bars())}
    single = run_portfolio(signals, markets, CAPITAL)
    multi = run_multi_portfolio(signals, markets, CAPITAL, rules(a=0.01))
    assert multi.trade_log().equals(single.trade_log())
    assert multi.signals["status"].to_list() == single.signals["status"].to_list()
    assert multi.final_capital == pytest.approx(single.final_capital)


def test_size_uses_only_trades_closed_by_the_signal():
    # a закрывается по цели на свече 45 (+20 000 ₽). Сигнал b на 15 — от
    # стартового капитала: a ещё открыта. Сигнал c на 45 — уже от 1 020 000.
    markets = {
        "a": market(FREE, trend_bars()),
        "b": market(FREE, flat_bars()),
        "c": market(FREE, flat_bars()),
    }
    signals = make_signals(
        [
            signal("a", 0, 95.0, 110.0),
            signal("b", 15, 45.0, 60.0),
            signal("c", 45, 45.0, 60.0),
        ]
    )
    result = run_multi_portfolio(
        signals, markets, CAPITAL, rules(a=0.01, b=0.01, c=0.01)
    )
    _, b, c = result.trades
    assert b.capital_before == pytest.approx(CAPITAL)
    assert b.size.contracts == 2_000
    assert c.capital_before == pytest.approx(1_020_000.0)
    assert c.size.contracts == 2_040


def test_new_position_shrinks_to_free_margin():
    # Цена 50, ГО 10% → 5 ₽ на контракт, стоп в 5 пунктах. a: риск 30% →
    # 60 000 контрактов, ГО 300 000 ₽ (30%). b хочет столько же, свободно 20%:
    # 200 000 ÷ 5 = 40 000 контрактов.
    markets = {"a": market(FREE, flat_bars()), "b": market(FREE, flat_bars())}
    signals = make_signals([signal("a", 0, 45.0, 60.0), signal("b", 15, 45.0, 60.0)])
    result = run_multi_portfolio(signals, markets, CAPITAL, rules(a=0.3, b=0.3))

    a, b = result.trades
    assert a.size.contracts == 60_000
    assert b.size.contracts == 40_000
    assert b.size.limit == PORTFOLIO_MARGIN
    assert status_of(result, "b", 15)["limit"] == PORTFOLIO_MARGIN
    assert a.size.margin + b.size.margin == pytest.approx(0.5 * CAPITAL)
    # 1R — риск того, что открыто: 40 000 × 5 ₽.
    assert b.size.risk == pytest.approx(200_000.0)


def test_simultaneous_signals_shrink_in_the_same_share():
    # Обе хотят по 300 000 ₽ ГО, свободно 500 000: каждая — 5/6 размера.
    markets = {"a": market(FREE, flat_bars()), "b": market(FREE, flat_bars())}
    signals = make_signals([signal("a", 0, 45.0, 60.0), signal("b", 0, 45.0, 60.0)])
    result = run_multi_portfolio(signals, markets, CAPITAL, rules(a=0.3, b=0.3))

    assert [item.size.contracts for item in result.trades] == [50_000, 50_000]
    assert {item.size.limit for item in result.trades} == {PORTFOLIO_MARGIN}


def test_no_free_margin_skips_signal():
    # a занимает все 50% ГО: у b 0 контрактов.
    markets = {"a": market(FREE, flat_bars()), "b": market(FREE, flat_bars())}
    signals = make_signals([signal("a", 0, 45.0, 60.0), signal("b", 15, 45.0, 60.0)])
    result = run_multi_portfolio(signals, markets, CAPITAL, rules(a=0.5, b=0.3))

    row = status_of(result, "b", 15)
    assert row["status"] == ZERO_SIZE
    assert row["limit"] == PORTFOLIO_MARGIN
    assert row["contracts"] == 0
    assert len(result.trades) == 1


def test_trading_stops_when_realized_capital_is_gone():
    # Гэп со 100 до 70 мимо стопа 99.9: 50 000 контрактов теряют 1.5 млн ₽.
    gap = make_bars(
        [
            bar(0, 100.0, 100.5, 99.5, 100.0),
            bar(15, 100.0, 100.5, 99.95, 100.0),
            *[bar(m, 70.0, 71.0, 69.0, 70.0) for m in range(30, 90, 15)],
        ]
    )
    markets = {"a": market(FREE, gap), "b": market(FREE, flat_bars())}
    signals = make_signals([signal("a", 0, 99.9, 110.0), signal("b", 45, 45.0, 60.0)])
    result = run_multi_portfolio(signals, markets, CAPITAL, rules(a=0.5, b=0.01))
    assert result.final_capital == pytest.approx(-500_000.0)
    assert status_of(result, "b", 45)["status"] == NO_CAPITAL


def test_risk_scale_multiplies_market_risk():
    # Доля a — 1%, стоп в 5 пунктах: 2 000 контрактов; с risk_scale 1.5 — 3 000,
    # с 0.5 — 1 000. Без столбца — множитель 1.
    for scale, contracts in ((1.5, 3_000), (0.5, 1_000)):
        signals = make_signals([signal("a", 0, 95.0, 110.0)]).with_columns(
            pl.lit(scale).alias("risk_scale")
        )
        result = run_multi_portfolio(
            signals, {"a": market(FREE, trend_bars())}, CAPITAL, rules(a=0.01)
        )
        assert result.trades[0].size.contracts == contracts


@pytest.mark.parametrize("scale", [0.0, -1.0, None])
def test_bad_risk_scale_rejected(scale):
    signals = make_signals([signal("a", 0, 95.0, 110.0)]).with_columns(
        pl.lit(scale, dtype=pl.Float64).alias("risk_scale")
    )
    with pytest.raises(ValueError, match="risk_scale"):
        run_multi_portfolio(
            signals, {"a": market(FREE, trend_bars())}, CAPITAL, rules(a=0.01)
        )


def test_every_market_needs_its_risk():
    signals = make_signals([signal("a", 0, 95.0, 110.0)])
    with pytest.raises(ValueError, match="доля риска"):
        run_multi_portfolio(
            signals, {"a": market(FREE, trend_bars())}, CAPITAL, rules(b=0.01)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_fractions": {"a": 0.0}},
        {"risk_fractions": {"a": 1.0}},
        {"risk_fractions": {"a": 0.01}, "max_margin_fraction": 0.0},
    ],
)
def test_bad_rules_rejected(kwargs):
    with pytest.raises(ValueError):
        MultiRules(**kwargs)
