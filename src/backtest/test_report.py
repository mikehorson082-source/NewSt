"""Проверки отчёта по прогону портфеля. Запуск: uv run pytest src/backtest/test_report.py"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

import math
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from backtest.portfolio import MarketData, run_portfolio
from backtest.position import LONG
from backtest.report import build_report, equity_curve, format_value, market_timeline
from backtest.specs import ContractSpec

CAPITAL = 1_000_000.0
FREE = ContractSpec(
    asset="free", tick_size=0.1, exchange_fee_rate=0.0, slippage_ticks=0.0
)
COSTLY = ContractSpec(asset="costly", tick_size=1.0, exchange_fee_rate=0.0)
DAY1 = date(2026, 9, 10)  # четверг
DAY2 = date(2026, 9, 11)  # пятница
DAY3 = date(2026, 9, 14)  # понедельник


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute)


def bar(ts: datetime, open_: float, high: float, low: float, close: float) -> dict:
    return {
        "timestamp": ts,
        "contract": "TSZ6",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "price_adjustment": 0.0,
        "is_first_bar_after_roll": False,
        "point_value": 1.0,
        "point_value_known": 1.0,
        "margin_rate": 0.1,
        "volume": 1_000_000,
        "usual_volume": 1_000_000.0,
    }


def three_days() -> pl.DataFrame:
    """Три сделки лонг, стоп 95, цель 110, сигнал по закрытию 10:00.

    День 1: цель — +10 пунктов × 2 000 = +20 000 ₽.
    День 2: просадка до 96 внутри сделки, потом цель: −8 160 ₽ на закрытии
    10:15, итог +10 × 2 040 = +20 400 ₽.
    День 3: стоп — −5 × 2 080 = −10 400 ₽.
    """
    rows = [
        bar(at(DAY1, 10), 100.0, 101.0, 99.0, 100.0),
        bar(at(DAY1, 10, 15), 100.0, 102.0, 99.0, 101.0),
        bar(at(DAY1, 10, 30), 101.0, 103.0, 100.0, 102.0),
        bar(at(DAY1, 10, 45), 102.0, 111.0, 101.0, 110.0),
        bar(at(DAY1, 11), 110.0, 111.0, 109.0, 110.0),
        bar(at(DAY2, 10), 100.0, 101.0, 99.0, 100.0),
        bar(at(DAY2, 10, 15), 100.0, 100.5, 95.5, 96.0),
        bar(at(DAY2, 10, 30), 96.0, 111.0, 96.0, 110.0),
        bar(at(DAY2, 10, 45), 110.0, 111.0, 109.0, 110.0),
        bar(at(DAY3, 10), 100.0, 101.0, 99.0, 100.0),
        bar(at(DAY3, 10, 15), 100.0, 100.5, 97.0, 98.0),
        bar(at(DAY3, 10, 30), 98.0, 99.0, 94.0, 95.0),
        bar(at(DAY3, 10, 45), 95.0, 96.0, 94.0, 95.0),
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def signals(*times: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "market": "a",
                "timestamp": ts,
                "side": LONG,
                "stop": 95.0,
                "target": 110.0,
            }
            for ts in times
        ]
    )


def three_day_report():
    bars = three_days()
    markets = {"a": MarketData(spec=FREE, bars=bars)}
    result = run_portfolio(
        signals(at(DAY1, 10), at(DAY2, 10), at(DAY3, 10)), markets, CAPITAL
    )
    return result, build_report(result, market_timeline([bars]))


def equity_at(report, ts: datetime) -> float:
    return report.curve.filter(pl.col("timestamp") == ts)["equity"][0]


def test_scenario_trades():
    result, _ = three_day_report()
    assert [t.size.contracts for t in result.trades] == [2_000, 2_040, 2_080]
    assert [t.trade.net_pnl for t in result.trades] == pytest.approx(
        [20_000.0, 20_400.0, -10_400.0]
    )
    assert result.final_capital == pytest.approx(1_030_000.0)


def test_curve_follows_trade_inside_and_stays_flat_between():
    _, report = three_day_report()
    # До первой сделки — стартовый капитал.
    assert equity_at(report, at(DAY1, 10)) == CAPITAL
    # Внутри сделки — по закрытиям свечей: 101 и 102 при входе по 100.
    assert equity_at(report, at(DAY1, 10, 15)) == pytest.approx(1_002_000.0)
    assert equity_at(report, at(DAY1, 10, 30)) == pytest.approx(1_004_000.0)
    # Между сделками капитал стоит.
    assert equity_at(report, at(DAY1, 11)) == pytest.approx(1_020_000.0)
    assert equity_at(report, at(DAY2, 10)) == pytest.approx(1_020_000.0)
    # Просадка внутри выигрышной сделки видна: (96 − 100) × 2 040.
    assert equity_at(report, at(DAY2, 10, 15)) == pytest.approx(1_011_840.0)
    assert report.curve["equity"][-1] == pytest.approx(1_030_000.0)
    in_position = report.curve.filter(pl.col("in_position"))["timestamp"].to_list()
    assert in_position == [
        at(DAY1, 10, 15),
        at(DAY1, 10, 30),
        at(DAY1, 10, 45),
        at(DAY2, 10, 15),
        at(DAY2, 10, 30),
        at(DAY3, 10, 15),
        at(DAY3, 10, 30),
    ]


def test_drawdown_counts_dip_inside_trade():
    _, report = three_day_report()
    s = report.summary
    # Худшая просадка — стоп дня 3 от пика 1 040 400.
    assert s["max_drawdown_rub"] == pytest.approx(-10_400.0)
    assert s["max_drawdown"] == pytest.approx(-10_400.0 / 1_040_400.0)
    # Просадка дня 2 есть только на кривой по свечам: по закрытиям сделок
    # капитал рос 1 020 000 → 1 040 400 без провала.
    day2 = report.curve.filter(pl.col("timestamp").dt.date() == DAY2)
    assert day2["equity"].min() == pytest.approx(1_011_840.0)
    # Самая долгая — время ниже прошлого максимума. Капитал стоял на пике
    # 1 040 400 с пт 10:30 до пн 10:00 (между сделками — не просадка), ниже —
    # с пн 10:15 до конца данных 10:45: 45 минут от последней точки на пике.
    assert s["longest_drawdown_days"] == pytest.approx(45 / 1440)


def test_daily_returns_and_ratios():
    _, report = three_day_report()
    assert report.daily["date"].to_list() == [DAY1, DAY2, DAY3]
    returns = [0.02, 1_040_400.0 / 1_020_000.0 - 1, 1_030_000.0 / 1_040_400.0 - 1]
    assert report.daily["return"].to_list() == pytest.approx(returns)

    s = report.summary
    # 3 торговых дня за 5 календарных: 3 ÷ (5 / 365.25) дней в году.
    periods = 3 / (5 / 365.25)
    r = np.array(returns)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(periods))
    downside = math.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert s["sortino"] == pytest.approx(r.mean() / downside * math.sqrt(periods))
    assert 0.0 < s["psr"] < 1.0
    assert s["sharpe_se"] > 0
    assert s["total_return"] == pytest.approx(0.03)
    assert s["cagr"] == pytest.approx(1.03 ** (365.25 / 5) - 1)
    assert s["calmar"] == pytest.approx(s["cagr"] / -s["max_drawdown"])


def test_trade_statistics():
    _, report = three_day_report()
    s = report.summary
    assert s["trades"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["profit_factor"] == pytest.approx(40_400.0 / 10_400.0)
    assert s["expectancy_rub"] == pytest.approx(10_000.0)
    # 2R, 2R и −1R.
    assert s["expectancy_r"] == pytest.approx(1.0)
    # 7 свечей в позиции из 13.
    assert s["exposure"] == pytest.approx(7 / 13)
    # MAE выигрышных: −0.2R (low 99) и −0.9R (low 95.5 × 2 040 ÷ 10 200).
    assert s["winners_mae_r"] == pytest.approx(-0.55)
    # MFE проигрышной: high 100.5 — +0.5 × 2 080 = 1 040 ₽ = 0.1R.
    assert s["losers_mfe_r"] == pytest.approx(0.1)
    # Без издержек: результат до и после совпадает, запаса не посчитать.
    assert s["costs"] == 0.0
    assert s["pnl_before_costs"] == pytest.approx(s["net_pnl"])
    assert s["cost_headroom"] is None


def test_single_year_matches_summary():
    _, report = three_day_report()
    assert report.by_year["year"].to_list() == [2026]
    row = report.by_year.row(0, named=True)
    for key, value in report.summary.items():
        assert row[key] == pytest.approx(value), key


def test_years_split_by_exit_and_carry_capital():
    # Сделка 31.12 → капитал второго года начинается с её результата.
    bars = pl.DataFrame(
        [
            bar(datetime(2025, 12, 31, 10), 100.0, 101.0, 99.0, 100.0),
            bar(datetime(2025, 12, 31, 10, 15), 100.0, 111.0, 99.0, 110.0),
            bar(datetime(2026, 1, 5, 10), 100.0, 101.0, 99.0, 100.0),
            bar(datetime(2026, 1, 5, 10, 15), 100.0, 100.0, 94.0, 95.0),
            bar(datetime(2026, 1, 5, 10, 30), 95.0, 96.0, 94.0, 95.0),
        ],
        infer_schema_length=None,
    )
    result = run_portfolio(
        signals(datetime(2025, 12, 31, 10), datetime(2026, 1, 5, 10)),
        {"a": MarketData(spec=FREE, bars=bars)},
        CAPITAL,
    )
    report = build_report(result, market_timeline([bars]))
    first, second = report.by_year.iter_rows(named=True)
    assert first["year"] == 2025
    assert first["end_capital"] == pytest.approx(1_020_000.0)
    assert first["net_pnl"] == pytest.approx(20_000.0)
    assert second["start_capital"] == pytest.approx(1_020_000.0)
    assert second["total_return"] == pytest.approx(-0.01)
    assert second["trades"] == 1


def test_cost_headroom():
    # Проскальзывание 1 пункт: риск контракта 5 + 2 = 7 ₽ → 1 428 контрактов.
    # По ценам рынка 100 → 110: +14 280 ₽, издержки 2 × 1 428 = 2 856 ₽.
    bars = three_days().head(5)
    result = run_portfolio(
        signals(at(DAY1, 10)), {"a": MarketData(spec=COSTLY, bars=bars)}, CAPITAL
    )
    s = build_report(result, market_timeline([bars])).summary
    assert s["pnl_before_costs"] == pytest.approx(14_280.0)
    assert s["costs"] == pytest.approx(2_856.0)
    assert s["net_pnl"] == pytest.approx(14_280.0 - 2_856.0)
    # Издержки могли бы вырасти в 5 раз, прежде чем результат обнулится.
    assert s["cost_headroom"] == pytest.approx(5.0)
    assert s["costs_per_trade_r"] == pytest.approx(2 / 7)


def test_trade_closed_by_end_of_data_counts_in_money_not_in_win_rate():
    # Цель не достигнута, данные кончились: сделку закрыл конец данных.
    bars = three_days().head(3)
    result = run_portfolio(
        signals(at(DAY1, 10)), {"a": MarketData(spec=FREE, bars=bars)}, CAPITAL
    )
    assert result.trades[0].trade.exit_reason == "end"
    s = build_report(result, market_timeline([bars])).summary
    assert s["end_capital"] == pytest.approx(1_004_000.0)
    assert s["net_pnl"] == pytest.approx(4_000.0)
    assert s["trades"] == 0
    assert s["win_rate"] is None
    assert s["expectancy_r"] is None


def test_no_trades_gives_flat_curve():
    bars = three_days()
    result = run_portfolio(
        signals(at(DAY3, 10, 45)), {"a": MarketData(spec=FREE, bars=bars)}, CAPITAL
    )
    report = build_report(result, market_timeline([bars]))
    assert report.curve["equity"].unique().to_list() == [CAPITAL]
    assert report.summary["sharpe"] is None
    assert report.summary["max_drawdown"] == 0.0
    assert report.summary["trades"] == 0


def test_timeline_must_contain_trade_bars():
    result, _ = three_day_report()
    short = market_timeline([three_days().head(5)])
    with pytest.raises(ValueError, match="нет в расписании"):
        equity_curve(result, short)


def test_timeline_merges_markets():
    a = three_days().head(2)
    b = three_days().slice(1, 3)
    assert market_timeline([a, b]).to_list() == [
        at(DAY1, 10),
        at(DAY1, 10, 15),
        at(DAY1, 10, 30),
        at(DAY1, 10, 45),
    ]


@pytest.mark.parametrize(
    ("key", "value", "text"),
    [
        ("total_return", 0.0312, "3.1%"),
        ("end_capital", 1_030_000.0, "1 030 000"),
        ("sharpe", 0.4567, "0.46"),
        ("trades", 12, "12"),
        ("longest_drawdown_days", 905.52, "906"),
        ("profit_factor", None, "—"),
    ],
)
def test_format_value(key, value, text):
    assert format_value(key, value) == text
