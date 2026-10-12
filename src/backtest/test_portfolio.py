"""Проверки портфеля с одной позицией. Запуск: uv run pytest src/backtest/test_portfolio.py"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

from datetime import datetime, timedelta

import polars as pl
import pytest

from backtest.minutes import RawMinutes
from backtest.portfolio import (
    BUSY,
    LOST_TIE,
    NO_CAPITAL,
    NO_NEXT_BAR,
    TAKEN,
    ZERO_SIZE,
    MarketData,
    run_portfolio,
)
from backtest.position import LONG
from backtest.specs import ContractSpec

START = datetime(2026, 9, 10, 10)
CAPITAL = 1_000_000.0

# Без издержек: риск контракта = расстояние до стопа × рубли за пункт.
FREE = ContractSpec(
    asset="free", tick_size=0.1, exchange_fee_rate=0.0, slippage_ticks=0.0
)
# Проскальзывание 1 пункт на сторону: издержки контракта 2 ₽.
COSTLY = ContractSpec(asset="costly", tick_size=1.0, exchange_fee_rate=0.0)


def bar(
    minute: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    point_value: float = 1.0,
    usual_volume: float = 1_000_000.0,
) -> dict:
    """Свеча; обычный объём по умолчанию огромный — лимит ликвидности не мешает."""
    return {
        "timestamp": START + timedelta(minutes=minute),
        "contract": "TSZ6",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "price_adjustment": 0.0,
        "is_first_bar_after_roll": False,
        "point_value": point_value,
        "point_value_known": point_value,
        "margin_rate": 0.1,
        "volume": 4000,
        "usual_volume": usual_volume,
    }


def make_bars(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def trend_bars() -> pl.DataFrame:
    """Сигнал на 100, вход по 100 на свече 15, цель 110 задета на свече 45."""
    return make_bars(
        [
            bar(0, 100.0, 101.0, 99.0, 100.0),
            bar(15, 100.0, 102.0, 99.0, 101.0),
            bar(30, 101.0, 103.0, 100.0, 102.0),
            bar(45, 102.0, 111.0, 101.0, 110.0),
            bar(60, 110.0, 111.0, 109.0, 110.0),
            bar(75, 110.0, 111.0, 109.0, 110.0),
        ]
    )


def flat_bars(point_value: float = 1.0) -> pl.DataFrame:
    """Цена стоит на 50: лонг со стопом 45 и целью 60 доживает до конца ряда."""
    return make_bars(
        [bar(m, 50.0, 51.0, 49.0, 50.0, point_value) for m in range(0, 90, 15)]
    )


def market(spec: ContractSpec, bars: pl.DataFrame, minutes=None) -> MarketData:
    return MarketData(spec=spec, bars=bars, minutes=minutes)


def signal(
    name: str, minute: int, stop: float, target: float, max_bars: int | None = None
) -> dict:
    return {
        "market": name,
        "timestamp": START + timedelta(minutes=minute),
        "side": LONG,
        "stop": stop,
        "target": target,
        "max_bars": max_bars,
    }


def make_signals(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def status_of(result, name: str, minute: int) -> dict:
    rows = result.signals.filter(
        (pl.col("market") == name)
        & (pl.col("timestamp") == START + timedelta(minutes=minute))
    )
    assert rows.height == 1
    return rows.row(0, named=True)


def two_markets() -> dict[str, MarketData]:
    return {"a": market(FREE, trend_bars()), "b": market(FREE, flat_bars())}


def test_busy_until_exit_bar_and_capital_carries_over():
    signals = make_signals(
        [
            signal("a", 0, 95.0, 110.0),
            signal("b", 15, 45.0, 60.0),
            signal("b", 30, 45.0, 60.0),
            signal("b", 45, 45.0, 60.0),
        ]
    )
    result = run_portfolio(signals, two_markets(), CAPITAL)

    assert status_of(result, "a", 0)["status"] == TAKEN
    # Сигналы во время сделки a — «занято», с номером мешавшей сделки.
    for minute in (15, 30):
        row = status_of(result, "b", minute)
        assert row["status"] == BUSY
        assert row["trade_id"] == 0
        assert row["contracts"] is None
    # Сделка a закрылась на свече 45: сигнал по её закрытию уже исполняется.
    assert status_of(result, "b", 45)["status"] == TAKEN
    assert status_of(result, "b", 45)["trade_id"] == 1

    first, second = result.trades
    # Риск 1% = 10 000 ₽, стоп в 5 пунктах по 1 ₽ → 2 000 контрактов.
    assert first.size.contracts == 2_000
    assert first.trade.exit_reason == "target"
    assert first.trade.net_pnl == pytest.approx(20_000.0)
    assert first.r_multiple == pytest.approx(2.0)
    # Вторая сделка — от капитала после первой: 10 200 ₽ ÷ 5 ₽ = 2 040.
    assert second.capital_before == pytest.approx(1_020_000.0)
    assert second.size.contracts == 2_040
    assert second.trade.entry_time == START + timedelta(minutes=60)
    assert result.final_capital == pytest.approx(1_020_000.0)


def test_cheaper_signal_wins_on_same_bar():
    markets = {"a": market(COSTLY, trend_bars()), "b": market(FREE, flat_bars())}
    signals = make_signals([signal("a", 0, 95.0, 110.0), signal("b", 0, 45.0, 60.0)])
    result = run_portfolio(signals, markets, CAPITAL)

    loser = status_of(result, "a", 0)
    assert loser["status"] == LOST_TIE
    assert loser["trade_id"] == 0
    # Издержки 2 ₽ при риске контракта 5 + 2 = 7 ₽.
    assert loser["cost_share"] == pytest.approx(2 / 7)
    winner = status_of(result, "b", 0)
    assert winner["status"] == TAKEN
    assert winner["cost_share"] == 0.0
    assert len(result.trades) == 1
    assert result.trades[0].trade.asset == "free"


@pytest.mark.parametrize(("order", "winner"), [("ab", "a"), ("ba", "b")])
def test_equal_costs_resolved_by_market_order(order, winner):
    data = two_markets()
    markets = {name: data[name] for name in order}
    signals = make_signals([signal("a", 0, 95.0, 110.0), signal("b", 0, 45.0, 60.0)])
    result = run_portfolio(signals, markets, CAPITAL)
    loser = "b" if winner == "a" else "a"
    assert status_of(result, winner, 0)["status"] == TAKEN
    assert status_of(result, loser, 0)["status"] == LOST_TIE


def test_zero_size_passes_turn_to_next_candidate():
    # b дешевле (без издержек), но стоп в 49 пунктах по 1 000 ₽ — 49 000 ₽
    # на контракт при бюджете 10 000 ₽: 0 контрактов, берётся a.
    markets = {
        "a": market(COSTLY, trend_bars()),
        "b": market(FREE, flat_bars(point_value=1_000.0)),
    }
    signals = make_signals([signal("a", 0, 95.0, 110.0), signal("b", 0, 1.0, 60.0)])
    result = run_portfolio(signals, markets, CAPITAL)

    skipped = status_of(result, "b", 0)
    assert skipped["status"] == ZERO_SIZE
    assert skipped["limit"] == "risk"
    assert skipped["contracts"] == 0
    assert skipped["trade_id"] is None
    assert status_of(result, "a", 0)["status"] == TAKEN
    assert result.trades[0].trade.asset == "costly"


def test_thin_entry_bar_passes_turn_to_next_candidate():
    # b дешевле, но свеча входа у b обычно почти пустая: 10% от 5 контрактов — 0.
    thin = make_bars(
        [
            bar(m, 50.0, 51.0, 49.0, 50.0, usual_volume=5.0 if m == 15 else 1e6)
            for m in range(0, 90, 15)
        ]
    )
    markets = {"a": market(COSTLY, trend_bars()), "b": market(FREE, thin)}
    signals = make_signals([signal("a", 0, 95.0, 110.0), signal("b", 0, 45.0, 60.0)])
    result = run_portfolio(signals, markets, CAPITAL)

    skipped = status_of(result, "b", 0)
    assert skipped["status"] == ZERO_SIZE
    assert skipped["limit"] == "volume"
    assert status_of(result, "a", 0)["status"] == TAKEN


def test_signal_on_last_bar_has_nothing_to_enter():
    signals = make_signals([signal("a", 75, 105.0, 120.0)])
    result = run_portfolio(signals, two_markets(), CAPITAL)
    assert status_of(result, "a", 75)["status"] == NO_NEXT_BAR
    assert result.trades == []
    assert result.final_capital == CAPITAL
    assert result.trade_log().is_empty()


def test_trading_stops_when_capital_is_gone():
    # Стоп в 0.1 пункта: по риску 100 000 контрактов, ГО ограничивает 50 000.
    # Гэп со 100 до 70 мимо стопа: убыток 1.5 млн ₽ при капитале 1 млн ₽.
    gap = make_bars(
        [
            bar(0, 100.0, 100.5, 99.5, 100.0),
            bar(15, 100.0, 100.5, 99.95, 100.0),
            *[bar(m, 70.0, 71.0, 69.0, 70.0) for m in range(30, 90, 15)],
        ]
    )
    markets = {"a": market(FREE, gap), "b": market(FREE, flat_bars())}
    signals = make_signals([signal("a", 0, 99.9, 110.0), signal("b", 45, 45.0, 60.0)])
    result = run_portfolio(signals, markets, CAPITAL)

    trade = result.trades[0]
    assert trade.size.limit == "margin"
    assert trade.size.contracts == 50_000
    assert trade.trade.exit_price == pytest.approx(70.0)
    assert result.final_capital == pytest.approx(-500_000.0)
    assert status_of(result, "b", 45)["status"] == NO_CAPITAL


def test_minutes_reach_the_simulator(tmp_path):
    # Свеча 45 задела и стоп 95, и цель 110; минутки: цель была раньше.
    bars = make_bars(
        [
            *trend_bars().head(3).iter_rows(named=True),
            bar(45, 102.0, 111.0, 94.0, 105.0),
            bar(60, 105.0, 106.0, 104.0, 105.0),
        ]
    )
    start = START + timedelta(minutes=45)
    pl.DataFrame(
        [
            {
                "timestamp": start,
                "open": 102.0,
                "high": 111.0,
                "low": 101.0,
                "close": 110.0,
            },
            {
                "timestamp": start + timedelta(minutes=1),
                "open": 110.0,
                "high": 110.0,
                "low": 94.0,
                "close": 105.0,
            },
        ]
    ).write_parquet(tmp_path / "TSZ6.parquet")
    signals = make_signals([signal("a", 0, 95.0, 110.0)])

    with_minutes = run_portfolio(
        signals, {"a": market(FREE, bars, RawMinutes(tmp_path))}, CAPITAL
    )
    without = run_portfolio(signals, {"a": market(FREE, bars)}, CAPITAL)
    assert with_minutes.trades[0].trade.exit_reason == "target"
    assert with_minutes.trades[0].trade.exit_resolution == "minutes"
    assert without.trades[0].trade.exit_reason == "stop"
    assert without.trades[0].trade.exit_resolution == "no_minutes"


def test_max_bars_reaches_the_simulator():
    signals = make_signals([signal("a", 0, 95.0, 110.0, max_bars=2)])
    result = run_portfolio(signals, two_markets(), CAPITAL)
    trade = result.trades[0].trade
    assert trade.exit_reason == "time"
    assert trade.exit_time == START + timedelta(minutes=30)


def test_max_bars_column_is_optional():
    signals = make_signals([signal("a", 0, 95.0, 110.0)]).drop("max_bars")
    result = run_portfolio(signals, two_markets(), CAPITAL)
    assert result.trades[0].trade.exit_reason == "target"


def test_hold_reaches_the_simulator():
    bars = (
        trend_bars()
        .with_row_index("_i")
        .with_columns(
            (pl.col("_i") == 2).alias("is_day_end"),
            pl.lit(False).alias("is_before_break"),
        )
    )
    signals = make_signals([signal("a", 0, 95.0, 110.0)]).with_columns(
        pl.lit("day").alias("hold")
    )
    result = run_portfolio(signals, {"a": market(FREE, bars.drop("_i"))}, CAPITAL)
    trade = result.trades[0].trade
    assert trade.exit_reason == "day_end"
    assert trade.exit_time == START + timedelta(minutes=30)
    # Следующий сигнал принимается с бара выхода, как после любой сделки.
    assert result.signals["hold"].to_list() == ["day"]


def test_hold_column_is_optional_and_empty_means_any():
    signals = make_signals([signal("a", 0, 95.0, 110.0)]).with_columns(
        pl.lit(None, dtype=pl.String).alias("hold")
    )
    result = run_portfolio(signals, two_markets(), CAPITAL)
    assert result.trades[0].trade.exit_reason == "target"


def test_signal_order_in_input_does_not_matter():
    rows = [
        signal("a", 0, 95.0, 110.0),
        signal("b", 15, 45.0, 60.0),
        signal("b", 45, 45.0, 60.0),
    ]
    forward = run_portfolio(make_signals(rows), two_markets(), CAPITAL)
    backward = run_portfolio(make_signals(rows[::-1]), two_markets(), CAPITAL)
    assert forward.signals.equals(backward.signals)
    assert forward.trade_log().equals(backward.trade_log())


def test_trade_log_and_volume_share():
    signals = make_signals([signal("a", 0, 95.0, 110.0)])
    result = run_portfolio(signals, two_markets(), CAPITAL)
    log = result.trade_log()
    assert log.height == 1
    row = log.row(0, named=True)
    assert row["market"] == "free"
    assert row["contracts"] == 2_000
    assert row["r_multiple"] == pytest.approx(2.0)
    # 2 000 контрактов при объёме свечи входа 4 000 — половина рынка.
    assert row["volume_share"] == pytest.approx(0.5)
    assert row["capital_after"] == pytest.approx(1_020_000.0)
    # Без издержек: low 99 бара входа — −1 пункт × 2 000 = −2 000 ₽ = −0.2R;
    # цель 110 — +10 × 2 000 = 20 000 ₽ = 2R.
    assert row["mae_r"] == pytest.approx(-0.2)
    assert row["mfe_r"] == pytest.approx(2.0)
    assert row["costs"] == 0.0
    assert row["pnl_before_costs"] == pytest.approx(row["net_pnl"])


def test_volume_share_without_volume_column():
    markets = {"a": market(FREE, trend_bars().drop("volume"))}
    result = run_portfolio(make_signals([signal("a", 0, 95.0, 110.0)]), markets)
    assert result.trades[0].entry_volume is None
    assert result.trades[0].volume_share is None


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([signal("x", 0, 95.0, 110.0)], "Нет данных рынков"),
        ([signal("a", 5, 95.0, 110.0)], "нет свечи сигнала"),
        ([signal("a", 0, 95.0, 110.0)] * 2, "Два сигнала"),
    ],
)
def test_bad_signals_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        run_portfolio(make_signals(rows), two_markets(), CAPITAL)


def test_missing_signal_columns_rejected():
    signals = make_signals([signal("a", 0, 95.0, 110.0)]).drop("stop")
    with pytest.raises(ValueError, match="нет столбцов"):
        run_portfolio(signals, two_markets(), CAPITAL)


def test_unsorted_bars_rejected():
    markets = {"a": market(FREE, trend_bars().reverse())}
    with pytest.raises(ValueError, match="по времени"):
        run_portfolio(make_signals([signal("a", 0, 95.0, 110.0)]), markets)


def test_capital_must_be_positive():
    with pytest.raises(ValueError, match="капитал"):
        run_portfolio(make_signals([signal("a", 0, 95.0, 110.0)]), two_markets(), 0.0)
