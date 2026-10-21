"""Проверки ADX, VHF и сжатия полос Боллинджера. Запуск: uv run pytest strategies/st01"""

# бары в тестах — naive timestamp MSK, как в data/continuous_15m.
# ruff: noqa: DTZ001

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from check_trend_indicators import (
    INDICATORS,
    THIRDS,
    adx,
    indicator_bars,
    past_rank,
    vhf,
    with_thirds,
)

START = datetime(2024, 1, 1)


def bars_from(
    closes: np.ndarray, adjustment: np.ndarray | None = None, step=timedelta(minutes=15)
) -> pl.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pl.DataFrame(
        {
            "timestamp": [START + step * i for i in range(n)],
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "price_adjustment": np.zeros(n) if adjustment is None else adjustment,
        }
    )


def zigzag(n: int) -> np.ndarray:
    return 100 + np.arange(n) % 2


def last(bars: pl.DataFrame, expr: pl.Expr) -> float:
    return bars.select(expr)[-1, 0]


def test_adx_of_steady_trend_is_100():
    # Каждая свеча на шаг выше прошлой: движение только вверх, DX = 100.
    assert last(bars_from(100 + np.arange(200)), adx(14)) == pytest.approx(100)


def test_adx_of_zigzag_is_low():
    # Вверх-вниз на шаг: +DI и −DI почти равны, направления нет.
    assert last(bars_from(zigzag(400)), adx(14)) < 10


def test_vhf_one_way_and_zigzag():
    assert last(bars_from(100 + np.arange(100)), vhf(28)) == pytest.approx(1.0)
    assert last(bars_from(zigzag(100)), vhf(28)) == pytest.approx(1 / 28)


def test_vhf_stays_between_zero_and_one():
    rng = np.random.default_rng(1)
    values = bars_from(100 + rng.normal(size=500).cumsum()).select(vhf(28))
    assert values.drop_nulls().to_series().is_between(0, 1).all()


def test_roll_jump_does_not_move_adx_and_vhf():
    # Новый контракт дороже на 10, поправка ряда это гасит: непрерывные цены те же.
    rng = np.random.default_rng(2)
    smooth = 100 + rng.normal(size=300).cumsum()
    jump = np.where(np.arange(300) >= 150, 10.0, 0.0)
    plain = bars_from(smooth)
    rolled = bars_from(smooth + jump, adjustment=-jump)
    for expr in (adx(14), vhf(28)):
        assert np.allclose(
            plain.select(expr).to_numpy(),
            rolled.select(expr).to_numpy(),
            equal_nan=True,
        )


def test_past_rank_uses_only_past_days():
    # Окно 3 дня: место дня 3 — среди дней 0–2, дня 4 — среди 1–3, сам день не входит.
    days = np.array(
        [np.datetime64("2024-01-01T12:00") + np.timedelta64(i, "D") for i in range(6)]
    )
    ranks = past_rank(days, np.array([3.0, 1.0, 2.0, 5.0, 4.0, 0.0]), 3)
    assert np.isnan(ranks[:3]).all()
    assert ranks[3:] == pytest.approx([1.0, 2 / 3, 0.0])


def test_indicators_do_not_look_ahead():
    # Свечи раз в день, чтобы набралось полгода истории для сжатия полос.
    rng = np.random.default_rng(3)
    closes = 100 + rng.normal(size=400).cumsum()
    changed = closes.copy()
    changed[350:] += 25 * rng.normal(size=50)
    step = timedelta(days=1)
    before = indicator_bars(bars_from(closes, step=step)).head(350)
    after = indicator_bars(bars_from(changed, step=step)).head(350)
    assert before["squeeze"].drop_nulls().len() > 100
    for column in INDICATORS:
        assert np.allclose(
            before[column].to_numpy(), after[column].to_numpy(), equal_nan=True
        )


def test_thirds_are_split_by_each_market():
    # У рынка b индикатор в 100 раз больше, но трети считаются внутри рынка.
    trades = pl.DataFrame(
        {
            "market": ["a"] * 9 + ["b"] * 9,
            "x": [float(v) for v in range(1, 10)] + [100.0 * v for v in range(1, 10)],
        }
    )
    grouped = with_thirds(trades, "x")
    counts = grouped.group_by("market", "треть").len()
    assert counts["len"].to_list() == [3] * 6
    lows = grouped.filter(pl.col("треть") == THIRDS[0]).sort("x")
    assert lows["x"].to_list() == [1.0, 2.0, 3.0, 100.0, 200.0, 300.0]
