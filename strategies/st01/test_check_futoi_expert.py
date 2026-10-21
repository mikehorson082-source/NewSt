"""Проверки правил идей эксперта по FUTOI. Запуск: uv run pytest strategies/st01"""

# сигналы в тестах — naive timestamp MSK, как в данных.
# ruff: noqa: DTZ001

from datetime import datetime

import numpy as np
import polars as pl
from check_futoi_expert import (
    AGAINST_CROWD,
    WITH_CROWD,
    crowd_scale,
    shuffled_scale,
    skipped_counts,
    with_crowd_only,
)


def signals(weeks: list[float | None], years: list[int] | None = None) -> pl.DataFrame:
    years = years or [2024] * len(weeks)
    return pl.DataFrame(
        {
            "timestamp": [datetime(y, 1, 1 + i) for i, y in enumerate(years)],
            "fiz_week": weeks,
        },
        schema={"timestamp": pl.Datetime("us"), "fiz_week": pl.Float64},
    )


def test_crowd_scale_by_weekly_flow():
    # С нами — ×1.5, против — ×0.5, ровно 0 или нет данных — ×1.
    scaled = crowd_scale(signals([3.0, -2.0, 0.0, None]))
    assert scaled["risk_scale"].to_list() == [WITH_CROWD, AGAINST_CROWD, 1.0, 1.0]


def test_overlay_drops_only_signals_against_crowd():
    kept = with_crowd_only(signals([3.0, -2.0, 0.0, None]))
    assert kept["fiz_week"].to_list() == [3.0, 0.0, None]


def test_skipped_counts_by_year():
    all_signals = signals([1.0, -1.0, -1.0, 1.0], years=[2023, 2023, 2024, 2024])
    assert skipped_counts(all_signals, with_crowd_only(all_signals)) == {
        2023: 1,
        2024: 1,
    }


def test_shuffled_scale_keeps_the_same_multipliers():
    scaled = crowd_scale(signals([3.0, -2.0, 1.0, -1.0, None]))
    shuffled = shuffled_scale(scaled, np.random.default_rng(1))
    assert sorted(shuffled["risk_scale"]) == sorted(scaled["risk_scale"])
