"""Проверки расчёта просадок (plan.md, 5.7). Запуск: uv run pytest strategies/st01"""

from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import polars as pl
import pytest
from check_drawdowns import Weeks, drawdowns, sample_path, simulate, weekly


def test_drawdown_depth_and_length_by_weeks():
    # Капитал (лог): 0.10, −0.10, −0.05, 0.15, 0.05. Максимум 0.10 в неделе 0,
    # под ним — недели 0 и 1; новый максимум 0.15 в неделе 2, под ним — неделя 3.
    log_returns = np.array([0.10, -0.20, 0.05, 0.20, -0.10])
    weeks = np.array([0, 0, 1, 2, 3])
    depth, longest = drawdowns(log_returns, weeks)
    assert depth == pytest.approx(np.expm1(-0.20))
    assert longest == 1.0


def test_drawdown_from_the_start_counts_from_first_week():
    depth, longest = drawdowns(np.array([-0.1, -0.1]), np.array([0, 2]))
    assert depth == pytest.approx(np.expm1(-0.2))
    assert longest == 2.0


def test_no_drawdown_when_capital_only_grows():
    assert drawdowns(np.array([0.01, 0.02]), np.array([0, 1])) == (0.0, 0.0)


def test_weekly_splits_days_by_monday():
    # Пн 05.10.2026 … вс 11.10 — одна неделя, пн 12.10 — следующая.
    days = [date(2026, 10, 5) + timedelta(days=i) for i in range(8)]
    daily = pl.DataFrame({"date": days, "return": [0.01] * 8})
    weeks = weekly(daily)
    assert weeks.starts.tolist() == [0, 7]
    assert weeks.ends.tolist() == [7, 8]
    assert np.allclose(weeks.returns, np.log1p(0.01))


def toy_weeks() -> Weeks:
    # 10 недель по 5 дней, в неделе w все дни дают w.
    returns = np.repeat(np.arange(10, dtype=float), 5)
    starts = np.arange(0, 50, 5)
    return Weeks(returns, starts, starts + 5)


def test_sample_path_takes_whole_consecutive_weeks():
    weeks = toy_weeks()
    days, week_ids = sample_path(
        weeks, block=4, horizon=9, rng=np.random.default_rng(3)
    )
    assert len(days) == 9 * 5
    assert week_ids.tolist() == np.repeat(np.arange(9), 5).tolist()
    picked = weeks.returns[days].reshape(9, 5)[:, 0]
    # Внутри блока недели идут подряд, по кругу: 7, 8, 9, 0…
    for block_start in (0, 4):
        part = picked[block_start : block_start + 4]
        assert all((b - a) % 10 == 1 for a, b in pairwise(part))


def test_half_edge_halves_the_mean():
    weeks = toy_weeks()
    full = simulate(weeks, 1.0, block=1, runs=5, seed=1)
    flat = Weeks(weeks.returns - weeks.returns.mean(), weeks.starts, weeks.ends)
    zero = simulate(flat, 1.0, block=1, runs=5, seed=1)
    half = simulate(weeks, 0.5, block=1, runs=5, seed=1)
    # Те же истории: без перевеса просадки длиннее, чем с половиной, а с
    # половиной — не короче, чем с полным.
    assert (zero.longest >= half.longest).all()
    assert (half.longest >= full.longest).all()
