"""Какие просадки ждать от портфеля paper mode (plan.md, 5.7).

Вопрос пользователя: полгода без новых максимумов психологически тяжело —
как часто такое бывает и насколько глубоко, при нынешнем риске и при
половинном. Правила не меняются: это ожидания, а не подбор.

Портфель — как в paper mode (5.6): cnyrub, si, euro по 0.67% риска (группа
2%), gold 1%, ГО до 50%. «Половина» — 0.33% на пару и 0.5% на золото.
Прогон по всей истории.

Метод — блочный бутстрап. Дневные результаты портфеля режутся на недели,
недели — на блоки по 4 подряд (около месяца). Из блоков с возвращением
собирается 10 000 историй по 5 лет, в каждой — самая глубокая и самая долгая
просадка. Блоки, а не отдельные дни: так сохраняются серии стопов внутри
месяца. Пример: в настоящей истории одна полугодовая просадка; в
перемешанных историях она может выпасть дважды или ни разу — бутстрап
покажет, как часто.

Два сценария:
- как в бэктесте;
- перевес вдвое меньше: из каждого дня вычтена половина среднего дневного
  результата. Цифры бэктеста завышены подбором (cnyrub, gold) и отбором
  рынков, вживую перевес обычно меньше.

Соседние длины блока — 1 и 13 недель — только для устойчивости: если ответ
сильно зависит от длины блока, у рынка есть долгие режимы, и короткие блоки
их недооценивают.

Длина просадки — от недели последнего максимума до последней недели под ним,
в месяцах (неделя = 7 / 30.44 месяца). Глубина — по закрытиям дней; отчёт
портфеля считает по свечам, внутри дня бывает чуть глубже. Сверка на
настоящей истории — в первой таблице.

Ничего не записывает, кроме графика.
Запуск: uv run python strategies/st01/check_drawdowns.py
→ strategies/st01/reports/drawdowns.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from rules import frozen_breakout_signals
from run_baseline import CAPITAL
from run_portfolio import RISK_FRACTIONS, RULES
from tune_breakout import REPORTS, free_path

from backtest.charts import BLUE, INK_SECONDARY, ORANGE, SURFACE, _style
from backtest.portfolio import MarketData, MultiRules, load_markets, run_multi_portfolio
from backtest.report import Report, build_report, format_value, market_timeline

RISK_LEVELS: dict[str, dict[str, float]] = {
    "сейчас": dict(RISK_FRACTIONS),
    "половина": {name: share / 2 for name, share in RISK_FRACTIONS.items()},
}
# Доля перевеса бэктеста, которая остаётся в сценарии.
SCENARIOS = {"как в бэктесте": 1.0, "перевес вдвое меньше": 0.5}
HORIZON_WEEKS = 261  # 5 лет
FIRST_YEAR_WEEKS = 52
BLOCK_WEEKS = (4, 1, 13)  # центральная длина блока — первая
RUNS = 10_000
SEED = 1
MONTHS_PER_WEEK = 7 / 30.44
LONG_MONTHS = (6, 12)
DEEP = (-0.2, -0.3)
WORST_PERCENTILE = 10  # «худшие 10% историй»
EPS = 1e-12

COLORS = {"сейчас": BLUE, "половина": ORANGE}
STYLES = {"как в бэктесте": "-", "перевес вдвое меньше": "--"}


@dataclass(frozen=True)
class Weeks:
    """Дневные лог-доходности по порядку и границы недель в них."""

    returns: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

    @property
    def count(self) -> int:
        return len(self.starts)

    def week_ids(self) -> np.ndarray:
        """Номер недели каждого дня настоящей истории."""
        return np.repeat(np.arange(self.count), self.ends - self.starts)


def weekly(daily: pl.DataFrame) -> Weeks:
    """Дни отчёта (`Report.daily`) по неделям с понедельника."""
    frame = daily.sort("date").with_columns(
        pl.col("return").log1p().alias("_log"),
        pl.col("date").dt.truncate("1w").alias("_week"),
    )
    counts = (
        frame.group_by("_week", maintain_order=True)
        .len()["len"]
        .cast(pl.Int64)
        .to_numpy()
    )
    ends = np.cumsum(counts)
    return Weeks(frame["_log"].to_numpy(), ends - counts, ends)


def drawdowns(log_returns: np.ndarray, weeks: np.ndarray) -> tuple[float, float]:
    """Самая глубокая просадка (доля) и самая долгая (недель) на одном пути.

    `weeks[i]` — номер недели дня i, по неубыванию. Капитал стартует на
    максимуме в первой неделе пути. Пример: максимум в неделе 3, капитал под
    ним до недели 30 включительно — просадка 27 недель, около 6 месяцев.
    """
    equity = np.cumsum(log_returns)
    peak = np.maximum.accumulate(np.maximum(equity, 0.0))
    depth = float(np.expm1((equity - peak).min()))
    at_peak = equity >= peak - EPS
    peak_week = np.maximum.accumulate(np.where(at_peak, weeks, weeks[0]))
    below = ~at_peak
    longest = float((weeks - peak_week)[below].max()) if below.any() else 0.0
    return min(depth, 0.0), longest


def sample_path(
    weeks: Weeks, block: int, horizon: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Одна перемешанная история: номера дней и номер недели каждого дня.

    Блоки по `block` недель подряд, начало — случайная неделя; история
    закольцована (за последней неделей идёт первая).
    """
    blocks = -(-horizon // block)
    first = rng.integers(0, weeks.count, size=blocks)
    order = ((first[:, None] + np.arange(block)) % weeks.count).ravel()[:horizon]
    days = np.concatenate([np.arange(weeks.starts[w], weeks.ends[w]) for w in order])
    week_ids = np.repeat(np.arange(horizon), weeks.ends[order] - weeks.starts[order])
    return days, week_ids


@dataclass(frozen=True)
class Simulation:
    depth: np.ndarray  # самая глубокая просадка за 5 лет, доля
    longest: np.ndarray  # самая долгая за 5 лет, месяцев
    first_year: np.ndarray  # самая долгая за первый год, месяцев


def simulate(weeks: Weeks, edge: float, block: int, runs: int, seed: int) -> Simulation:
    """`runs` историй по 5 лет; `edge` — доля среднего дневного результата."""
    rng = np.random.default_rng(seed)
    shift = (1.0 - edge) * weeks.returns.mean()
    depth = np.empty(runs)
    longest = np.empty(runs)
    first_year = np.empty(runs)
    for i in range(runs):
        days, week_ids = sample_path(weeks, block, HORIZON_WEEKS, rng)
        path = weeks.returns[days] - shift
        depth[i], weeks_long = drawdowns(path, week_ids)
        longest[i] = weeks_long * MONTHS_PER_WEEK
        first = week_ids < FIRST_YEAR_WEEKS
        first_year[i] = drawdowns(path[first], week_ids[first])[1] * MONTHS_PER_WEEK
    return Simulation(depth, longest, first_year)


def simulation_row(label: dict[str, Any], sim: Simulation) -> dict[str, Any]:
    row = {
        **label,
        "Max DD: медиана": f"{np.median(sim.depth) * 100:.1f}%",
        f"Max DD: худшие {WORST_PERCENTILE}%": (
            f"{np.percentile(sim.depth, WORST_PERCENTILE) * 100:.1f}%"
        ),
    }
    for level in DEEP:
        row[f"глубже {level:.0%}, % историй"] = round(
            float((sim.depth < level).mean() * 100), 1
        )
    row["самая долгая: медиана, мес."] = round(float(np.median(sim.longest)), 1)
    for months in LONG_MONTHS:
        row[f"≥ {months} мес., % историй"] = round(
            float((sim.longest >= months).mean() * 100), 1
        )
    row[f"за первый год ≥ {LONG_MONTHS[0]} мес., %"] = round(
        float((sim.first_year >= LONG_MONTHS[0]).mean() * 100), 1
    )
    return row


def run_level(markets: dict[str, MarketData], risk: dict[str, float]) -> Report:
    """Портфель paper mode по всей истории с этим риском на сделку."""
    signals = pl.concat(
        [
            frozen_breakout_signals(name, data.bars).with_columns(
                pl.lit(name).alias("market")
            )
            for name, data in markets.items()
        ],
        how="diagonal_relaxed",
    )
    rules = MultiRules(
        risk_fractions=risk,
        max_margin_fraction=RULES.max_margin_fraction,
        max_volume_share=RULES.max_volume_share,
    )
    result = run_multi_portfolio(signals, markets, CAPITAL, rules)
    return build_report(result, market_timeline(d.bars for d in markets.values()))


def history_row(level: str, report: Report, weeks: Weeks) -> dict[str, Any]:
    s = report.summary
    depth, longest = drawdowns(weeks.returns, weeks.week_ids())
    risks = ", ".join(f"{share:.2%}" for share in RISK_LEVELS[level].values())
    return {
        "риск": level,
        "риск сделки: пары…, золото": risks,
        "CAGR": format_value("cagr", s["cagr"]),
        "Sharpe": round(s["sharpe"], 2),
        "Max DD, по свечам": format_value("max_drawdown", s["max_drawdown"]),
        "Max DD, по дням": f"{depth * 100:.1f}%",
        "долгая, отчёт, мес.": round(s["longest_drawdown_days"] / 30.44, 1),
        "долгая, по неделям, мес.": round(longest * MONTHS_PER_WEEK, 1),
    }


def save_chart(sims: dict[tuple[str, str], Simulation]) -> Path:
    fig = Figure(figsize=(14, 10), dpi=110, facecolor=SURFACE)
    panels = (
        ("longest", "Самая долгая просадка за 5 лет, месяцев", 1.0),
        ("depth", "Самая глубокая просадка за 5 лет, %", 100.0),
    )
    for i, (field, title, scale) in enumerate(panels, 1):
        ax = fig.add_subplot(2, 1, i)
        _style(ax, title)
        for (level, scenario), sim in sims.items():
            ax.hist(
                getattr(sim, field) * scale,
                bins=60,
                histtype="step",
                lw=1.6,
                color=COLORS[level],
                ls=STYLES[scenario],
                label=f"риск {level}, {scenario}",
            )
        if field == "longest":
            for months in LONG_MONTHS:
                ax.axvline(months, color=INK_SECONDARY, lw=0.6)
        ax.legend(fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
    fig.suptitle(
        "Просадки портфеля paper mode: 10 000 историй по 5 лет (блоки по 4 недели)",
        x=0.06,
        ha="left",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = free_path(REPORTS / "drawdowns.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return path


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets(RISK_FRACTIONS)
    reports = {level: run_level(markets, risk) for level, risk in RISK_LEVELS.items()}
    weeks = {level: weekly(report.daily) for level, report in reports.items()}
    dates = reports["сейчас"].daily["date"]
    first, last = dates.item(0), dates.item(-1)
    print(
        f"\nНастоящая история {first:%d.%m.%Y}–{last:%d.%m.%Y}, портфель paper mode "
        f"({', '.join(markets)}). Сверка: расчёт по дням и неделям против отчёта."
    )
    print(pl.DataFrame([history_row(lv, reports[lv], weeks[lv]) for lv in reports]))

    central = BLOCK_WEEKS[0]
    sims: dict[tuple[str, str], Simulation] = {}
    rows = []
    for level in RISK_LEVELS:
        for scenario, edge in SCENARIOS.items():
            sim = simulate(weeks[level], edge, central, RUNS, SEED)
            sims[level, scenario] = sim
            rows.append(simulation_row({"риск": level, "сценарий": scenario}, sim))
    print(
        f"\n{RUNS} историй по 5 лет, блоки по {central} недели. "
        "«% историй» — в скольких из них такое было хотя бы раз."
    )
    print(pl.DataFrame(rows))

    neighbor_rows = []
    for block in BLOCK_WEEKS:
        sim = simulate(weeks["сейчас"], 1.0, block, RUNS, SEED)
        neighbor_rows.append(simulation_row({"блок, недель": block}, sim))
    print("\nУстойчивость: риск сейчас, как в бэктесте, разная длина блока.")
    print(pl.DataFrame(neighbor_rows))

    print(f"\nГрафик: {save_chart(sims)}")


if __name__ == "__main__":
    main()
