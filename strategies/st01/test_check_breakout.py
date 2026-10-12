"""Проверки расчётов скрипта check_breakout. Запуск: uv run pytest strategies/st01"""

# сделки в тестах — naive timestamp MSK, как в журнале сделок.
# ruff: noqa: DTZ001

from datetime import datetime, timedelta

import polars as pl
import pytest
from check_breakout import bootstrap_expectancy, trades_by

MONDAY = datetime(2025, 1, 6, 12)


def trades(rows: list[tuple[int, float]], side: int = 1) -> pl.DataFrame:
    """Сделки (номер недели, результат в R): по одной на строку."""
    return pl.DataFrame(
        {
            "exit_time": [MONDAY + timedelta(weeks=week) for week, _ in rows],
            "r_multiple": [r for _, r in rows],
            "net_pnl": [r * 10_000 for _, r in rows],
            "side": [side] * len(rows),
            "exit_reason": ["stop"] * len(rows),
        }
    )


def test_bootstrap_of_only_winning_weeks_never_drops_to_zero():
    stats = bootstrap_expectancy(trades([(w, 0.5 + 0.1 * (w % 3)) for w in range(30)]))
    assert stats["weeks"] == 30
    assert stats["trades"] == 30
    assert stats["not_positive"] == 0.0
    assert 0.5 <= stats["low"] <= stats["expectancy"] <= stats["high"] <= 0.7


def test_bootstrap_of_coin_flip_is_centered_on_zero():
    rows = [(w, 1.0 if w % 2 else -1.0) for w in range(200)]
    stats = bootstrap_expectancy(trades(rows))
    assert stats["expectancy"] == pytest.approx(0.0)
    assert stats["low"] < 0 < stats["high"]
    assert stats["not_positive"] == pytest.approx(0.5, abs=0.05)


def test_bootstrap_keeps_trades_of_one_week_together():
    # Одна неделя с десятью сделками по +1R и девять недель по одной −1R.
    # Сделки перемешивались бы почти без разброса вокруг +0.05R; недели —
    # нет: без «удачной» недели история целиком убыточна.
    rows = [(0, 1.0)] * 10 + [(w, -1.0) for w in range(1, 10)]
    stats = bootstrap_expectancy(trades(rows))
    assert stats["weeks"] == 10
    assert stats["trades"] == 19
    assert stats["low"] == pytest.approx(-1.0)
    assert stats["not_positive"] > 0.3


def test_bootstrap_is_reproducible_and_skips_end_trades():
    frame = trades([(w, 0.2 * (w % 5) - 0.3) for w in range(50)])
    ended = frame.head(1).with_columns(
        pl.lit("end").alias("exit_reason"), pl.lit(100.0).alias("r_multiple")
    )
    first = bootstrap_expectancy(pl.concat([frame, ended]))
    assert first == bootstrap_expectancy(frame)
    assert first["trades"] == 50


def test_trades_by_side():
    frame = pl.concat(
        [trades([(0, 2.0), (1, -1.0)], side=1), trades([(0, -1.0)], side=-1)]
    )
    table = trades_by(frame, "сторона")
    long_row, short_row = table.rows(named=True)
    assert long_row == {
        "сторона": "лонг",
        "сделок": 2,
        "выигрышных %": 50.0,
        "expectancy, R": 0.5,
        "сумма, R": 1.0,
    }
    assert short_row["сторона"] == "шорт"
    assert short_row["сумма, R"] == -1.0
