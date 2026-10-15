"""Проверки правил двух стратегий. Запуск: uv run pytest strategies/st01"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

import math
import random
from datetime import datetime, time, timedelta

import numpy as np
import polars as pl
import pytest
from rules import (
    BARS_PER_DAY,
    breakout_signals,
    frozen_breakout_signals,
    long_trend,
    reversion_signals,
    volatility_rank,
)

from backtest.markets import add_bar_sigma, add_session_marks, add_usual_volume
from backtest.portfolio import MarketData, run_portfolio
from backtest.position import HOLD_DAY, HOLD_OVERNIGHT, LONG, SHORT
from backtest.specs import ContractSpec

# Среда 09.09.2026, 10:00: до конца дня далеко, выходных рядом нет.
START = datetime(2026, 9, 9, 10)
SIGMA = 0.001
DAY_MOVE = SIGMA * math.sqrt(BARS_PER_DAY)


def make_bars(closes: list[float], spread: float = 0.5, **columns) -> pl.DataFrame:
    """Свечи по закрытиям: high и low на `spread` от закрытия."""
    n = len(closes)
    base = {
        "timestamp": [START + timedelta(minutes=15 * i) for i in range(n)],
        "high": [c + spread for c in closes],
        "low": [c - spread for c in closes],
        "close": closes,
        "price_adjustment": [0.0] * n,
        "bar_sigma": [SIGMA] * n,
        "is_day_end": [False] * n,
    }
    return pl.DataFrame({**base, **columns})


def at(signals: pl.DataFrame, bar: int) -> dict | None:
    rows = signals.filter(pl.col("timestamp") == START + timedelta(minutes=15 * bar))
    return rows.row(0, named=True) if rows.height else None


# ---------------------------------------------------------------- пробой канала


def test_breakout_long_levels_from_day_move():
    # Канал прошлых 4 свечей: 99.5–100.5. Закрытие 101 — выше.
    bars = make_bars([100.0, 100.0, 100.0, 100.0, 101.0])
    signals = breakout_signals(bars, lookback=4)
    assert signals.height == 1
    signal = at(signals, 4)
    assert signal["side"] == LONG
    assert signal["hold"] == HOLD_OVERNIGHT
    distance = 101.0 * DAY_MOVE
    assert signal["stop"] == pytest.approx(101.0 - distance)
    assert signal["target"] == pytest.approx(101.0 + 3 * distance)


def test_breakout_short_mirrors_long():
    bars = make_bars([100.0, 100.0, 100.0, 100.0, 99.0])
    signal = at(breakout_signals(bars, lookback=4, stop_days=2.0, reward=1.5), 4)
    assert signal["side"] == SHORT
    distance = 2.0 * 99.0 * DAY_MOVE
    assert signal["stop"] == pytest.approx(99.0 + distance)
    assert signal["target"] == pytest.approx(99.0 - 1.5 * distance)


def test_breakout_needs_close_beyond_channel_not_just_high():
    # high последней свечи 100.9 выше канала 100.5, но закрытие 100.4 — внутри.
    bars = make_bars([100.0, 100.0, 100.0, 100.0, 100.4])
    assert breakout_signals(bars, lookback=4).is_empty()


def test_breakout_waits_for_full_channel():
    bars = make_bars([100.0, 100.0, 100.0, 101.0, 102.0])
    signals = breakout_signals(bars, lookback=4)
    # На свече 3 в прошлом только 3 свечи — канала ещё нет.
    assert at(signals, 3) is None
    assert at(signals, 4)["side"] == LONG


def test_breakout_ignores_contract_roll_jump():
    # Новый контракт на 5 дороже: реальная цена прыгнула, непрерывная стоит.
    closes = [100.0, 100.0, 100.0, 100.0, 105.0, 106.5]
    bars = make_bars(closes, price_adjustment=[0.0, 0.0, 0.0, 0.0, -5.0, -5.0])
    signals = breakout_signals(bars, lookback=4)
    assert at(signals, 4) is None
    # Настоящий пробой на новом контракте: уровни — в его реальных ценах.
    signal = at(signals, 5)
    assert signal["side"] == LONG
    assert signal["stop"] == pytest.approx(106.5 * (1 - DAY_MOVE))


def test_no_signal_without_sigma_history():
    bars = make_bars(
        [100.0, 100.0, 100.0, 100.0, 101.0],
        bar_sigma=pl.Series([None] * 5, dtype=pl.Float64),
    )
    assert breakout_signals(bars, lookback=4).is_empty()


def test_no_signal_on_last_bar_of_day():
    bars = make_bars(
        [100.0, 100.0, 100.0, 100.0, 101.0],
        is_day_end=[False, False, False, False, True],
    )
    assert breakout_signals(bars, lookback=4).is_empty()


def test_no_signal_on_weekend_session():
    saturday = datetime(2026, 9, 12, 10)
    bars = make_bars([100.0, 100.0, 100.0, 100.0, 101.0]).with_columns(
        pl.Series("timestamp", [saturday + timedelta(minutes=15 * i) for i in range(5)])
    )
    assert breakout_signals(bars, lookback=4).is_empty()


def test_missing_columns_rejected():
    bars = make_bars([100.0, 100.0]).drop("is_day_end")
    with pytest.raises(ValueError, match="is_day_end"):
        breakout_signals(bars)


# ------------------------------------------------------- варианты пробоя


def test_breakout_without_target_carries_trailing_distance():
    bars = make_bars([100.0, 100.0, 100.0, 100.0, 101.0])
    signal = at(breakout_signals(bars, lookback=4, reward=None, trail_days=0.5), 4)
    assert signal["target"] is None
    assert signal["breakeven_r"] is None
    assert signal["trail"] == pytest.approx(0.5 * 101.0 * DAY_MOVE)
    # Начальный стоп тот же, что у правила шага 4.
    assert signal["stop"] == pytest.approx(101.0 * (1 - DAY_MOVE))


def test_breakout_carries_breakeven_threshold():
    bars = make_bars([100.0, 100.0, 100.0, 100.0, 101.0])
    signal = at(breakout_signals(bars, lookback=4, breakeven_r=1.0), 4)
    assert signal["breakeven_r"] == 1.0
    assert signal["trail"] is None
    assert signal["target"] == pytest.approx(101.0 * (1 + 3 * DAY_MOVE))


def bars_at_times(closes: list[float], times: list[str], **columns) -> pl.DataFrame:
    """Свечи по закрытиям с заданными метками времени «YYYY-MM-DDTHH:MM»."""
    return make_bars(closes, **columns).with_columns(
        pl.Series("timestamp", [datetime.fromisoformat(t) for t in times])
    )


def test_entry_hours_look_at_the_entry_bar():
    # Пробой на свече 18:45: вход — на свече 19:00, уже вечер.
    times = [f"2026-09-09T{t}" for t in ("18:00", "18:15", "18:30", "18:45", "19:00")]
    bars = bars_at_times([100.0, 100.0, 100.0, 101.0, 101.0], times)
    main = breakout_signals(bars, lookback=3, entry_hours=(time(10), time(19)))
    assert main.is_empty()
    no_morning = breakout_signals(bars, lookback=3, entry_hours=(time(10), None))
    assert no_morning["timestamp"].to_list() == [datetime(2026, 9, 9, 18, 45)]


def test_skip_break_days_drops_the_whole_day():
    # Чт 10.09 — обычная ночь; пт 11.09 кончается перерывом.
    times = [
        "2026-09-10T10:00",
        "2026-09-10T10:15",
        "2026-09-10T10:30",
        "2026-09-10T10:45",
        "2026-09-10T23:45",
        "2026-09-11T10:00",
        "2026-09-11T10:15",
        "2026-09-11T23:45",
    ]
    closes = [100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 102.0, 102.0]
    marks = {
        "is_day_end": [False, False, False, False, True, False, False, True],
        "is_before_break": [False] * 7 + [True],
    }
    bars = bars_at_times(closes, times, **marks)
    every = breakout_signals(bars, lookback=3)
    assert every["timestamp"].dt.day().to_list() == [10, 11]
    skipped = breakout_signals(bars, lookback=3, skip_break_days=True)
    assert skipped["timestamp"].dt.day().to_list() == [10]


def trend_bars(previous_close: float) -> pl.DataFrame:
    """Два прошлых дня закрылись на `previous_close`, сегодня пробой вверх на 101."""
    times = [
        "2026-09-07T10:00",
        "2026-09-07T10:15",
        "2026-09-08T10:00",
        "2026-09-08T10:15",
        "2026-09-09T10:00",
        "2026-09-09T10:15",
        "2026-09-09T10:30",
        "2026-09-09T10:45",
    ]
    p = previous_close
    return bars_at_times([p, p, p, p, 100.0, 100.0, 100.0, 101.0], times)


def test_long_trend_uses_only_finished_days():
    trend = long_trend(trend_bars(90.0), days=2)
    means = dict(trend.iter_rows())
    assert means[datetime(2026, 9, 9).date()] == pytest.approx(90.0)
    # 08.09: в прошлом один день — средней за два ещё нет.
    assert means[datetime(2026, 9, 8).date()] is None


@pytest.mark.parametrize(("previous_close", "kept"), [(90.0, True), (110.0, False)])
def test_trend_filter_keeps_only_breakouts_with_the_long_trend(previous_close, kept):
    signals = breakout_signals(trend_bars(previous_close), lookback=3, trend_days=2)
    # Сигнал последней свечи: лонг на 101 при средней 90 или 110.
    last = signals.filter(pl.col("timestamp") == datetime(2026, 9, 9, 10, 45))
    assert last.height == (1 if kept else 0)


def test_frozen_rules_skip_morning_entries_only_on_rouble_markets():
    # 250 свечей до 09:45 (канал — 240); пробой на свече 09:30, вход в 09:45 — утро.
    last = datetime(2026, 9, 9, 9, 45)
    times = [(last - timedelta(minutes=15 * (249 - i))).isoformat() for i in range(250)]
    bars = bars_at_times([100.0] * 248 + [101.0, 101.0], times)
    frozen = {
        name: frozen_breakout_signals(name, bars)
        for name in ("cnyrub", "si", "gold", "brent")
    }
    assert frozen["cnyrub"].is_empty()
    assert frozen["si"].is_empty()
    for name in ("gold", "brent"):
        signal = frozen[name].row(0, named=True)
        assert signal["timestamp"] == datetime(2026, 9, 9, 9, 30)
        assert frozen[name].height == 1
        # Стоп 0.75 дня, цель 3 стопа.
        assert signal["stop"] == pytest.approx(101.0 * (1 - 0.75 * DAY_MOVE))
        assert signal["target"] == pytest.approx(101.0 * (1 + 3 * 0.75 * DAY_MOVE))


def test_frozen_rules_use_channel_of_240_bars():
    # Всплеск до 102 за 150 свечей до пробоя: канал 120 его уже не видит,
    # канал 240 — видит, и закрытие на 101 пробоем не считается.
    last = datetime(2026, 9, 9, 15, 0)
    times = [(last - timedelta(minutes=15 * (249 - i))).isoformat() for i in range(250)]
    closes = [100.0] * 248 + [101.0, 101.0]
    closes[99] = 102.0
    bars = bars_at_times(closes, times)
    assert frozen_breakout_signals("gold", bars).is_empty()
    assert not breakout_signals(bars, lookback=120).is_empty()


# ------------------------------------------------ роль волатильности (5.3.1)


def test_fixed_pct_stop_changes_only_the_stop():
    bars = make_bars([100.0, 100.0, 100.0, 100.0, 101.0])
    main = breakout_signals(bars, lookback=4)
    fixed = breakout_signals(bars, lookback=4, stop_pct=0.02)
    assert fixed["timestamp"].to_list() == main["timestamp"].to_list()
    signal = at(fixed, 4)
    assert signal["stop"] == pytest.approx(101.0 * 0.98)
    assert signal["target"] == pytest.approx(101.0 + 3 * 101.0 * 0.02)


def test_fixed_pct_stop_skips_bars_without_sigma_like_main_rule():
    bars = make_bars(
        [100.0, 100.0, 100.0, 100.0, 101.0],
        bar_sigma=pl.Series([None] * 5, dtype=pl.Float64),
    )
    assert breakout_signals(bars, lookback=4, stop_pct=0.02).is_empty()


def test_frozen_rules_pass_fixed_pct_stop():
    last = datetime(2026, 9, 9, 15, 0)
    times = [(last - timedelta(minutes=15 * (249 - i))).isoformat() for i in range(250)]
    bars = bars_at_times([100.0] * 248 + [101.0, 101.0], times)
    signal = frozen_breakout_signals("gold", bars, stop_pct=0.01).row(0, named=True)
    assert signal["stop"] == pytest.approx(101.0 * 0.99)


def rank_bars(days: int) -> pl.DataFrame:
    """Две свечи в будний день: 10:00 — σ по кругу 1…10 (×0.001), 10:15 — 0.0055."""
    first = datetime(2025, 1, 6)
    timestamps, sigmas = [], []
    for k in range(days):
        day = first + timedelta(days=k)
        if day.weekday() >= 5:
            continue
        timestamps += [day.replace(hour=10), day.replace(hour=10, minute=15)]
        sigmas += [(k % 10 + 1) * 0.001, 0.0055]
    return make_bars([100.0] * len(timestamps)).with_columns(
        pl.Series("timestamp", timestamps), pl.Series("bar_sigma", sigmas)
    )


def test_volatility_rank_is_share_of_past_year_days_below():
    bars = rank_bars(500)
    ranked = volatility_rank(bars)
    moment = datetime(2026, 3, 2, 10, 15)
    # Сверка «вручную»: σ первых свечей дней за [дата − 365, дата).
    day = moment.date()
    past = bars.filter(
        (pl.col("timestamp").dt.hour() == 10)
        & (pl.col("timestamp").dt.minute() == 0)
        & (pl.col("timestamp").dt.date() >= day - timedelta(days=365))
        & (pl.col("timestamp").dt.date() < day)
    )["bar_sigma"]
    expected = (past < 0.0055).sum() / past.len()
    rank = ranked.filter(pl.col("timestamp") == moment)["vol_rank"].item()
    assert rank == pytest.approx(expected)
    assert 0.4 < rank < 0.6


def test_volatility_rank_does_not_count_the_day_itself():
    bars = rank_bars(500).with_columns(
        pl.when(pl.col("timestamp") == datetime(2026, 3, 2, 10))
        .then(1.0)
        .otherwise(pl.col("bar_sigma"))
        .alias("bar_sigma")
    )
    ranked = volatility_rank(bars)
    # Бурнее всего за год: если бы сам день входил в сравнение, было бы < 1.
    rank = ranked.filter(pl.col("timestamp") == datetime(2026, 3, 2, 10))["vol_rank"]
    assert rank.item() == 1.0


def test_volatility_rank_needs_a_year_of_history():
    ranked = volatility_rank(rank_bars(500))
    first_year = ranked.filter(pl.col("timestamp") < datetime(2026, 1, 6))
    assert first_year["vol_rank"].null_count() == first_year.height
    later = ranked.filter(pl.col("timestamp") >= datetime(2026, 1, 6))
    assert later["vol_rank"].null_count() == 0


def test_volatility_rank_future_does_not_change_past():
    full = volatility_rank(rank_bars(500))
    cut = volatility_rank(rank_bars(430))
    joined = cut.join(full, on="timestamp", suffix="_full")
    assert joined.height == cut.height
    assert joined["vol_rank"].to_list() == joined["vol_rank_full"].to_list()


# ------------------------------------------------------------ возврат к средней


def test_reversion_long_targets_the_mean():
    closes = [100.0, 100.2, 99.8, 100.1, 99.9, 98.0]
    bars = make_bars(closes)
    # В окне из 5 свечей уход больше 1.79 разброса невозможен, поэтому 1.5.
    signals = reversion_signals(bars, lookback=5, entry_sd=1.5)
    assert signals.height == 1
    signal = at(signals, 5)
    window = closes[1:]
    mean = sum(window) / 5
    assert signal["side"] == LONG
    assert signal["hold"] == HOLD_DAY
    # Цель — средняя, стоп — на таком же расстоянии с другой стороны.
    assert signal["target"] == pytest.approx(mean)
    assert signal["stop"] == pytest.approx(98.0 - (mean - 98.0))


def test_reversion_short_mirrors_long():
    closes = [100.0, 99.8, 100.2, 99.9, 100.1, 102.0]
    signal = at(reversion_signals(make_bars(closes), lookback=5, entry_sd=1.5), 5)
    mean = sum(closes[1:]) / 5
    assert signal["side"] == SHORT
    assert signal["target"] == pytest.approx(mean)
    assert signal["stop"] == pytest.approx(102.0 + (102.0 - mean))


def test_reversion_signals_once_per_excursion():
    # Цена ушла вниз и остаётся за границей: сигнал только на первой свече.
    closes = [100.0, 100.2, 99.8, 100.1, 99.9, 98.0, 96.0, 94.0]
    signals = reversion_signals(make_bars(closes), lookback=5, entry_sd=1.5)
    assert [row["timestamp"] for row in signals.iter_rows(named=True)] == [
        START + timedelta(minutes=75)
    ]


def test_reversion_flat_market_gives_no_signals():
    assert reversion_signals(make_bars([100.0] * 12), lookback=5).is_empty()


def test_reversion_skips_stop_closer_than_one_bar_move():
    # Рынок почти стоит: уход на 0.04 от средней — это 1.6 разброса, но меньше
    # обычного движения свечи 100 × 0.1% = 0.1.
    closes = [100.0, 100.01, 99.99, 100.01, 99.99, 99.95]
    bars = make_bars(closes)
    assert reversion_signals(bars, lookback=5, entry_sd=1.5).is_empty()
    assert (
        reversion_signals(bars, lookback=5, entry_sd=1.5, min_stop_sigmas=0.1).height
        == 1
    )


def test_reversion_uses_continuous_prices_across_roll():
    adjustment = [0.0, 0.0, 0.0, 0.0, 0.0, -5.0]
    # Реальная цена прыгнула на новый контракт, непрерывная (100.0) — у средней.
    flat = make_bars(
        [100.0, 100.2, 99.8, 100.1, 99.9, 105.0], price_adjustment=adjustment
    )
    assert reversion_signals(flat, lookback=5, entry_sd=1.5).is_empty()
    # Настоящий уход на новом контракте: непрерывная цена 98.0, реальная 103.0.
    moved = make_bars(
        [100.0, 100.2, 99.8, 100.1, 99.9, 103.0], price_adjustment=adjustment
    )
    signal = at(reversion_signals(moved, lookback=5, entry_sd=1.5), 5)
    mean = (100.2 + 99.8 + 100.1 + 99.9 + 98.0) / 5
    assert signal["side"] == LONG
    # Уровни — в реальных ценах нового контракта: до средней те же 1.6.
    assert signal["target"] == pytest.approx(103.0 + (mean - 98.0))
    assert signal["stop"] == pytest.approx(103.0 - (mean - 98.0))


# ---------------------------------------------------------- без подглядывания


def random_bars(rng: random.Random, n: int = 400) -> pl.DataFrame:
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(0.0, 0.004)))
    return make_bars(closes, spread=0.2)


@pytest.mark.parametrize(
    ("rule", "n", "cut"),
    [
        (lambda bars: breakout_signals(bars, lookback=20), 400, 250),
        # Долгий тренд на 2 дня: нужно несколько дней истории.
        (
            lambda bars: breakout_signals(
                bars,
                lookback=20,
                reward=None,
                trail_days=1.0,
                entry_hours=(time(10), time(19)),
                trend_days=2,
            ),
            2000,
            1500,
        ),
        (lambda bars: reversion_signals(bars, lookback=20), 400, 250),
    ],
)
def test_future_bars_do_not_change_past_signals(rule, n, cut):
    rng = random.Random(3)
    bars = random_bars(rng, n)
    moment = bars["timestamp"][cut]
    # Будущее после свечи cut заменено другим: сигналы до неё те же.
    changed = bars.with_row_index("_i").with_columns(
        [
            pl.when(pl.col("_i") > cut)
            .then(pl.col(column) * 1.5)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("high", "low", "close")
        ]
    )
    before = rule(bars).filter(pl.col("timestamp") <= moment)
    after = rule(changed.drop("_i")).filter(pl.col("timestamp") <= moment)
    assert before.height > 3
    assert before.equals(after)


@pytest.mark.parametrize("rule", [breakout_signals, reversion_signals])
def test_signals_are_valid_for_the_simulator(rule):
    bars = random_bars(random.Random(5), n=1500)
    signals = rule(bars).join(bars.select("timestamp", "close"), on="timestamp")
    assert signals.height > 3
    longs = signals.filter(pl.col("side") == LONG)
    shorts = signals.filter(pl.col("side") == SHORT)
    assert longs.height > 0
    assert shorts.height > 0
    assert (longs["stop"] < longs["close"]).all()
    assert (longs["close"] < longs["target"]).all()
    assert (shorts["target"] < shorts["close"]).all()
    assert (shorts["close"] < shorts["stop"]).all()


# ------------------------------------------------------ случайное блуждание


def random_walk_market(seed: int, days: int = 1300) -> MarketData:
    """Рынок без закономерностей и без издержек: будни, 60 свечей в день.

    Цена идёт минутными шагами, high и low свечи — из этого же пути.
    """
    rng = np.random.default_rng(seed)
    first_day = datetime(2021, 1, 4, 9)
    stamps = [
        first_day + timedelta(days=d, minutes=15 * i)
        for d in range(days)
        if (first_day + timedelta(days=d)).weekday() < 5
        for i in range(BARS_PER_DAY)
    ]
    n = len(stamps)
    steps = rng.normal(0.0, 0.002 / math.sqrt(15), size=(n, 15))
    path = 100.0 * np.exp(np.cumsum(steps.ravel())).reshape(n, 15)
    close = path[:, -1]
    open_ = np.concatenate([[100.0], close[:-1]])
    bars = pl.DataFrame(
        {
            "timestamp": stamps,
            "open": open_,
            "high": np.maximum(path.max(axis=1), open_),
            "low": np.minimum(path.min(axis=1), open_),
            "close": close,
        }
    ).with_columns(
        pl.lit(0.0).alias("price_adjustment"),
        pl.lit(False).alias("is_first_bar_after_roll"),
        pl.lit(10.0).alias("point_value"),
        pl.lit(10.0).alias("point_value_known"),
        pl.lit(0.15).alias("margin_rate"),
        pl.lit(300_000).alias("volume"),
        (pl.col("close") / 100.0).alias("nav"),
    )
    free = ContractSpec(
        asset="walk", tick_size=0.01, exchange_fee_rate=0.0, slippage_ticks=0.0
    )
    return MarketData(
        spec=free, bars=add_session_marks(add_bar_sigma(add_usual_volume(bars)))
    )


@pytest.mark.parametrize(
    ("rule", "tolerance"), [(breakout_signals, 0.1), (reversion_signals, 0.05)]
)
def test_random_walk_gives_no_edge(rule, tolerance):
    # На случайных ценах заработать нельзя: правила, портфель и симулятор
    # вместе должны дать около 0R на сделку. Подглядывание в будущее дало бы
    # заметный плюс. На 150 рядах: пробой +0.010 ± 0.005R (свеча из 15 минуток
    # чуть перепрыгивает стоп, а выход считается по уровню), возврат
    # +0.001 ± 0.002R. Здесь 6 рядов: погрешность 0.026R и 0.010R.
    results = []
    for seed in range(6):
        market = {"walk": random_walk_market(seed)}
        signals = rule(market["walk"].bars).with_columns(pl.lit("walk").alias("market"))
        trades = run_portfolio(signals, market, 1_000_000.0).trade_log()
        results.append(trades.filter(pl.col("exit_reason") != "end")["r_multiple"])
    r_multiple = pl.concat(results)
    assert r_multiple.len() > 2000
    assert abs(r_multiple.mean()) < tolerance


def breakeven_variant(signals: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """Вариант б: стоп в безубыток после +1R, цель прежняя."""
    return signals.with_columns(pl.lit(1.0).alias("breakeven_r"))


def trailing_variant(signals: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """Вариант в: скользящий стоп на расстоянии начального, без цели."""
    return (
        signals.join(bars.select("timestamp", "close"), on="timestamp")
        .with_columns(
            (pl.col("close") - pl.col("stop")).abs().alias("trail"),
            pl.lit(None, dtype=pl.Float64).alias("target"),
        )
        .drop("close")
    )


@pytest.mark.parametrize("variant", [breakeven_variant, trailing_variant])
def test_random_walk_gives_no_edge_with_moving_stops(variant):
    # Подвижный стоп тоже не должен зарабатывать на случайных ценах: стоп
    # двигается только по прошлым свечам, а действует со следующей.
    results = []
    for seed in range(6):
        market = {"walk": random_walk_market(seed)}
        bars = market["walk"].bars
        signals = variant(breakout_signals(bars), bars).with_columns(
            pl.lit("walk").alias("market")
        )
        trades = run_portfolio(signals, market, 1_000_000.0).trade_log()
        results.append(trades.filter(pl.col("exit_reason") != "end")["r_multiple"])
    r_multiple = pl.concat(results)
    assert r_multiple.len() > 2000
    assert abs(r_multiple.mean()) < 0.1
