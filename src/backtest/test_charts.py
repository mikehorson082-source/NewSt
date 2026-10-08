"""Графики отчёта рисуются без ошибок. Запуск: uv run pytest src/backtest/test_charts.py"""

from backtest.charts import save_report_chart
from backtest.test_report import three_day_report


def test_report_chart_is_saved(tmp_path):
    _, report = three_day_report()
    path = save_report_chart(report, tmp_path / "report" / "chart.png", "Проверка")
    assert path.is_file()
    assert path.stat().st_size > 10_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
