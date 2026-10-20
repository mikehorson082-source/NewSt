"""Проверки фильтра входа шага 6. Запуск: uv run pytest strategies/st01

Тест CatBoost пропускается, если пакет не установлен (группа зависимостей ml):
uv run --group ml pytest strategies/st01
"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from catboost_filter import (
    CENTRAL_SHARE,
    LABEL_CLIP,
    SKIP_SHARES,
    TEST_YEARS,
    Fit,
    fit_catboost,
    fit_linear,
    kept,
    random_skip,
    skipped_by_year,
    target,
    threshold_column,
    walk_forward,
)
from entry_features import (
    FEATURES,
    label_signals,
    market_dataset,
    signal_features,
    training_rows,
)
from rules import frozen_breakout_signals
from test_rules import random_walk_market

from backtest.portfolio import MarketData, MultiRules, run_multi_portfolio
from backtest.position import LONG


@pytest.fixture(scope="module")
def walk() -> MarketData:
    # 2 года будней: хватает на год истории σ (vol_rank) и 200 дней тренда.
    return random_walk_market(seed=11, days=730)


@pytest.fixture(scope="module")
def costly_walk(walk: MarketData) -> MarketData:
    """Тот же рынок с проскальзыванием и сбором: метка должна их учесть."""
    spec = replace(walk.spec, exchange_fee_rate=0.0002, slippage_ticks=1.0)
    return replace(walk, spec=spec)


# ----------------------------------------------------------------- признаки


def test_features_do_not_change_when_future_changes(walk: MarketData):
    bars = walk.bars
    cut = bars.height - 3000
    moment = bars["timestamp"][cut]
    # Будущее после свечи cut заменено другим: признаки сигналов до неё те же.
    changed = (
        bars.with_row_index("_i")
        .with_columns(
            [
                pl.when(pl.col("_i") > cut)
                .then(pl.col(column) * 1.5)
                .otherwise(pl.col(column))
                .alias(column)
                for column in ("open", "high", "low", "close", "volume")
            ]
        )
        .drop("_i")
    )
    signals = frozen_breakout_signals("walk", bars).filter(
        pl.col("timestamp") <= moment
    )
    before = signal_features(signals, bars)
    after = signal_features(signals, changed)
    assert before.height > 50
    assert before["vol_rank"].drop_nulls().len() > 10
    assert before["trend"].drop_nulls().len() > 10
    assert before.equals(after)


def test_feature_signs_follow_the_trade(walk: MarketData):
    signals = frozen_breakout_signals("walk", walk.bars)
    features = signal_features(signals, walk.bars)
    # Закрытие за каналом: пробой всегда в сторону сделки, за день — тоже.
    assert (features["breakout_strength"] > 0).all()
    assert (features["move_1d"] > 0).all()
    assert (features["channel_width"] > 0).all()
    assert (features["hours_to_break"] > 0).all()
    assert features["weekday"].is_between(1, 5).all()
    assert set(features["side"].unique()) == {-1, 1}


def test_hours_to_break_counts_to_friday_close():
    # Среда 10:00 → выход перед выходными по закрытию пятничной 23:45.
    start = datetime(2026, 9, 9, 10)
    stamps = []
    day = start.date()
    for d in range(4):  # ср, чт, пт, пн
        current = day + timedelta(days=d if d < 3 else 5)
        first = datetime(current.year, current.month, current.day, 10)
        stamps += [first + timedelta(minutes=15 * i) for i in range(56)]  # до 23:45
    n = len(stamps)
    bars = pl.DataFrame(
        {
            "timestamp": stamps,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "price_adjustment": [0.0] * n,
            "bar_sigma": [0.001] * n,
            "volume": [10] * n,
            "usual_volume": [10.0] * n,
        }
    )
    from backtest.markets import add_session_marks

    bars = add_session_marks(bars)
    signals = pl.DataFrame(
        {
            "timestamp": [stamps[0]],
            "side": [LONG],
            "stop": [99.0],
            "target": [103.0],
            "hold": ["overnight"],
            "breakeven_r": [None],
            "trail": [None],
        },
        schema_overrides={"breakeven_r": pl.Float64, "trail": pl.Float64},
    )
    features = signal_features(signals, bars).row(0, named=True)
    # Сигнал закрылся в ср 10:15, пятничная свеча 23:45 закроется в сб 00:00.
    assert features["hours_to_break"] == pytest.approx(2 * 24 + 13.75)
    assert features["entry_hour"] == 10
    assert features["weekday"] == 3
    assert features["volume_ratio"] == pytest.approx(0.0)


# -------------------------------------------------------------------- метка


def test_label_equals_portfolio_r_multiple(costly_walk: MarketData):
    signals = frozen_breakout_signals("walk", costly_walk.bars)
    labels = label_signals(costly_walk, signals)
    result = run_multi_portfolio(
        signals.with_columns(pl.lit("walk").alias("market")),
        {"walk": costly_walk},
        1_000_000.0,
        MultiRules(risk_fractions={"walk": 0.01}),
    )
    trades = result.trade_log().select(
        pl.col("signal_time").alias("timestamp"), "r_multiple", "exit_time"
    )
    joined = trades.join(labels, on="timestamp")
    assert joined.height == trades.height > 50
    # Издержки на контракт от размера не зависят: R одной сделки тот же.
    assert np.allclose(joined["r"], joined["r_multiple"], atol=1e-9)
    assert (joined["exit_time"] == joined["exit_time_right"]).all()


def test_dataset_has_label_for_every_signal_with_next_bar(walk: MarketData):
    dataset = market_dataset("walk", walk)
    signals = frozen_breakout_signals("walk", walk.bars)
    # Сигнал на последней свече ряда пропускается: входить не на чем.
    last = walk.bars["timestamp"][-1]
    assert dataset.height == signals.filter(pl.col("timestamp") < last).height
    assert set(FEATURES) <= set(dataset.columns)
    assert dataset["r"].null_count() == 0


def test_training_rows_skip_trades_closed_after_start():
    start = datetime(2024, 1, 1)
    dataset = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 12, 20, 12),
                datetime(2023, 12, 28, 12),
                datetime(2023, 12, 29, 12),
            ],
            "exit_time": [
                datetime(2023, 12, 22, 23, 45),
                datetime(2024, 1, 3, 11),  # метка заглядывает в 2024 год
                datetime(2023, 12, 29, 23, 45),
            ],
            "exit_reason": ["break", "stop", "end"],
        }
    )
    rows = training_rows(dataset, start)
    assert rows["timestamp"].to_list() == [datetime(2023, 12, 20, 12)]


# -------------------------------------------------------------- walk-forward


def toy_dataset() -> pl.DataFrame:
    """По 100 сигналов на каждый год 2022–2026, сделка закрывается через неделю.

    Последний сигнал года (28–29.12) закрывается уже в январе.
    """
    rows = []
    for year in range(2022, 2027):
        for i in range(100):
            moment = datetime(year, 1, 2) + timedelta(days=3.65 * i)
            rows.append(
                {
                    "market": "walk",
                    "timestamp": moment,
                    "exit_time": moment + timedelta(days=7),
                    "exit_reason": "stop",
                    "r": float(i % 7) - 3.0,
                    "score": float(i),
                }
            )
    return pl.DataFrame(rows)


def test_walk_forward_trains_only_on_the_past():
    dataset = toy_dataset()
    seen: dict[int, tuple[datetime, int]] = {}

    def fake(train: pl.DataFrame, test: pl.DataFrame, y: np.ndarray) -> Fit:
        year = test["timestamp"].dt.year()[0]
        seen[year] = (train["exit_time"].max(), train.height)
        return Fit(np.arange(train.height, dtype=float), test["score"].to_numpy(), {})

    run = walk_forward(dataset, fake)
    for year in TEST_YEARS:
        last_exit, rows = seen[year]
        assert last_exit < datetime(year, 1, 1)
        # Сигнал конца декабря закрывается в январе — в обучение не идёт.
        assert rows < (year - 2022) * 100
    first = run.predictions.filter(pl.col("year") == TEST_YEARS[0])
    train_rows = int(first["train_rows"][0])
    assert first[threshold_column(CENTRAL_SHARE)][0] == pytest.approx(
        np.quantile(np.arange(train_rows), CENTRAL_SHARE)
    )


def test_threshold_does_not_depend_on_test_predictions():
    dataset = toy_dataset()

    def fitter(shift: float):
        def fake(train: pl.DataFrame, test: pl.DataFrame, y: np.ndarray) -> Fit:
            return Fit(
                np.arange(train.height, dtype=float),
                test["score"].to_numpy() + shift,
                {},
            )

        return fake

    low = walk_forward(dataset, fitter(0.0)).predictions
    high = walk_forward(dataset, fitter(1_000.0)).predictions
    for share in SKIP_SHARES:
        column = threshold_column(share)
        assert low[column].equals(high[column])
    # Прогнозы на проверке выросли — пропусков меньше, порог тот же.
    assert kept(high, CENTRAL_SHARE).height == high.height
    assert kept(low, CENTRAL_SHARE).height < low.height


def test_shuffled_target_keeps_values_and_clips():
    train = pl.DataFrame({"r": [-4.0, -1.0, 0.5, 2.9, 5.0]})
    plain = target(train)
    assert plain.tolist() == [LABEL_CLIP[0], -1.0, 0.5, 2.9, LABEL_CLIP[1]]
    shuffled = target(train, np.random.default_rng(1))
    assert sorted(shuffled.tolist()) == sorted(plain.tolist())
    assert shuffled.tolist() != plain.tolist()


def test_random_skip_removes_same_count_per_year():
    dataset = toy_dataset()
    predictions = dataset.with_columns(
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("score").alias("prediction"),
        pl.lit(30.0).alias(threshold_column(CENTRAL_SHARE)),
    )
    counts = skipped_by_year(predictions, CENTRAL_SHARE)
    assert counts == {year: 30 for year in range(2022, 2027)}
    rest = random_skip(dataset, counts, np.random.default_rng(2))
    left = rest.group_by(pl.col("timestamp").dt.year().alias("year")).len()
    assert left["len"].to_list() == [70] * 5
    again = random_skip(dataset, counts, np.random.default_rng(2))
    assert rest.equals(again)


# ------------------------------------------------------------------- модели


def model_data() -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """R растёт с силой пробоя; у части сигналов нет места волатильности."""
    rng = np.random.default_rng(4)
    n = 600

    def frame(size: int, hour_shift: int = 0) -> pl.DataFrame:
        strength = rng.uniform(0, 1, size)
        return pl.DataFrame(
            {
                "side": rng.choice([-1, 1], size),
                "breakout_strength": strength,
                "channel_width": rng.uniform(1, 6, size),
                "vol_rank": np.where(rng.uniform(size=size) < 0.3, np.nan, 0.5),
                "trend": rng.normal(0, 5, size),
                "move_1d": rng.uniform(0, 3, size),
                "move_5d": rng.normal(2, 2, size),
                "hours_to_break": rng.uniform(1, 100, size),
                "entry_hour": rng.integers(10, 20, size) + hour_shift,
                "weekday": rng.integers(1, 6, size),
                "volume_ratio": rng.normal(0, 1, size),
                "repeat_signals": rng.integers(0, 20, size).astype(float),
                "cost_share": rng.uniform(0.01, 0.05, size),
                "r": 3 * strength - 1 + rng.normal(0, 0.3, size),
            }
        ).with_columns(pl.col("vol_rank").fill_nan(None))

    train = frame(n)
    # В проверке есть час 20+, которого не было в обучении.
    test = frame(200, hour_shift=5)
    return train, test, train["r"].to_numpy()


def test_linear_handles_missing_values_and_new_hours():
    train, test, y = model_data()
    fit = fit_linear(train, test, y)
    assert np.isfinite(fit.test_pred).all()
    assert np.corrcoef(fit.test_pred, test["r"])[0, 1] > 0.8
    # Сила пробоя — главный признак, с плюсом.
    assert max(fit.importance, key=lambda k: abs(fit.importance[k])) == (
        "breakout_strength"
    )
    assert fit.importance["breakout_strength"] > 0


def test_catboost_learns_a_simple_rule():
    pytest.importorskip("catboost")
    train, test, y = model_data()
    fit = fit_catboost(train, test, y)
    assert np.isfinite(fit.test_pred).all()
    assert np.corrcoef(fit.test_pred, test["r"])[0, 1] > 0.8
    assert max(fit.importance, key=fit.importance.get) == "breakout_strength"
