"""Графики отчёта рисуются без ошибок. Запуск: uv run pytest src/backtest/test_charts.py"""

import polars as pl

from backtest.charts import save_cost_grid_chart, save_report_chart
from backtest.test_report import three_day_report


def test_report_chart_is_saved(tmp_path):
    _, report = three_day_report()
    path = save_report_chart(report, tmp_path / "report" / "chart.png", "Проверка")
    assert path.is_file()
    assert path.stat().st_size > 10_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cost_grid_chart_is_saved(tmp_path):
    grid = pl.DataFrame(
        {
            "slippage_ticks": [0.0, 1.0, 2.0] * 2,
            "fee_multiplier": [1.0] * 3 + [2.0] * 3,
            "expectancy_r": [0.1, 0.0, -0.1, 0.05, -0.05, -0.15],
        }
    )
    path = save_cost_grid_chart(grid, tmp_path / "grid.png", "Проверка")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
