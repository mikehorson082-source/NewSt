"""Подбор пробоя канала на cnyrub, mix, brent, gold (plan.md, 5.2).

Каждый рынок — в одиночку, вся история, как `solo` в check_breakout.py.
Сравнение — по expectancy в R и Sharpe каждого рынка; рубли рынков не
складываются. Соседние значения параметров — только для проверки
устойчивости: центральные значения заданы заранее и остаются.

Запуски:
0. `--stage stops` — выбор стопа: «а» со стопом 0.25, 0.5, 0.75 и 1 день,
   каждый при 1 и 2 шагах проскальзывания.
1. `--stage exits` — выход а, б, в при прежнем входе. Выход выбирает
   пользователь, с записью причины.
2. `--stage entries --exit fixed|breakeven|trail` — фильтры входа г, д, е
   поверх выбранного выхода (fixed — а, breakeven — б, trail — в) со стопом
   `CHOSEN_STOP_DAYS` (0.75 дня — выбор пользователя 14.10.2026).
   - г: вход только в основную сессию; соседние — без утра, без вечера.
   - д: диагностика — сделки, открытые в день, который кончается перерывом,
     против остальных; плюс прогон с правилом для сведения. Правило нужно,
     если вместе по четырём рынкам такие входы хуже больше чем на одну
     погрешность разницы.
   - е: только по направлению долгого тренда (средняя 200 дней, соседние 100
     и 300) на золоте (решение), mix и brent (для сведения). Все варианты е и
     прогон без фильтра — на одном периоде: с даты, когда появилась средняя
     на 300 дней.

Графики: strategies/st01/reports/tuning_<рынок>_<этап>.png — кривые капитала
всех вариантов вместе. Файл уже есть — новый получает суффикс _2, _3…

Запуск:
  uv run python strategies/st01/tune_breakout.py --stage stops
  uv run python strategies/st01/tune_breakout.py --stage exits
  uv run python strategies/st01/tune_breakout.py --stage entries --exit fixed
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from datetime import date, time
from pathlib import Path
from typing import Any

import polars as pl
from matplotlib.figure import Figure
from rules import break_days, breakout_signals, long_trend
from run_baseline import CAPITAL

from backtest.charts import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    SURFACE,
    _dates,
    _rub,
    _style,
)
from backtest.portfolio import MarketData, load_markets, run_portfolio
from backtest.position import LONG
from backtest.report import Report, build_report, format_value, market_timeline

REPORTS = Path(__file__).resolve().parent / "reports"
MARKETS = ("cnyrub", "mix", "brent", "gold")

# Выход, поверх которого проверяются фильтры входа.
EXITS: dict[str, dict[str, Any]] = {
    "fixed": {},
    "breakeven": {"breakeven_r": 1.0},
    "trail": {"reward": None, "trail_days": 1.0},
}
EXIT_LETTERS = {"fixed": "а", "breakeven": "б", "trail": "в"}
# Стоп для этапа входов: выбран пользователем 14.10.2026 по этапу stops.
CHOSEN_STOP_DAYS = 0.75

MAIN_SESSION = (time(10), time(19))
# Долгий тренд: решение — по золоту, mix и brent — для сведения, на cnyrub
# фильтр не применяем (там зарабатывают обе стороны).
TREND_DECISION = "gold"
TREND_MARKETS = ("gold", "mix", "brent")
TREND_DAYS = (200, 100, 300)  # центральное значение — первое

COLORS = {
    "а": BLUE,
    "б": ORANGE,
    "в": AQUA,
    "г": BLUE,
    "д": ORANGE,
    "е": AQUA,
    "1 шаг": BLUE,
    "2 шага": ORANGE,
}
# Соседние значения одной группы различаются видом пунктира.
NEIGHBOR_STYLES = ("--", ":", "-.", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)))
STOP_DAYS = (1.0, 0.25, 0.5, 0.75)
REFERENCE_COLOR = MUTED


@dataclass(frozen=True)
class Variant:
    """Вариант правила: подпись, группа (а…е), центральный или соседний."""

    label: str
    group: str
    central: bool
    params: dict[str, Any] = field(default_factory=dict)
    slippage_ticks: float | None = None  # None — как в паспорте рынка (1 шаг)


def exit_variants() -> list[Variant]:
    """а, б, в и их соседние значения. Начальный стоп у всех — 1 день."""
    return [
        Variant("а", "а", True),
        Variant("а · канал 60", "а", False, {"lookback": 60}),
        Variant("а · канал 240", "а", False, {"lookback": 240}),
        Variant("а · стоп 0.5 дня", "а", False, {"stop_days": 0.5}),
        Variant("а · стоп 2 дня", "а", False, {"stop_days": 2.0}),
        Variant("а · цель 2R", "а", False, {"reward": 2.0}),
        Variant("а · цель 4R", "а", False, {"reward": 4.0}),
        Variant("б · 1R", "б", True, {"breakeven_r": 1.0}),
        Variant("б · 0.5R", "б", False, {"breakeven_r": 0.5}),
        Variant("б · 1.5R", "б", False, {"breakeven_r": 1.5}),
        Variant("в · 1 день", "в", True, {"reward": None, "trail_days": 1.0}),
        Variant("в · 0.5 дня", "в", False, {"reward": None, "trail_days": 0.5}),
        Variant("в · 1.5 дня", "в", False, {"reward": None, "trail_days": 1.5}),
    ]


def stop_variants() -> list[Variant]:
    """«а» со стопом 0.25–1 день при 1 и 2 шагах проскальзывания."""
    variants = []
    for ticks, group in ((1.0, "1 шаг"), (2.0, "2 шага")):
        for days in STOP_DAYS:
            variants.append(
                Variant(
                    f"стоп {days:g} {'день' if days == 1 else 'дня'} · {group}",
                    group,
                    days in (1.0, 0.5),
                    {"stop_days": days},
                    ticks,
                )
            )
    return variants


def chosen_exit(exit_key: str) -> tuple[str, dict[str, Any]]:
    """Подпись и параметры выбранного выхода со выбранным стопом."""
    label = f"{EXIT_LETTERS[exit_key]} · стоп {CHOSEN_STOP_DAYS:g}"
    return label, {**EXITS[exit_key], "stop_days": CHOSEN_STOP_DAYS}


def entry_variants(exit_key: str) -> list[Variant]:
    """Выход без фильтра, г и его соседние, д — поверх выбранного выхода."""
    label, base = chosen_exit(exit_key)
    return [
        Variant(f"{label} без фильтра", "ref", True, base),
        Variant("г · 10:00–19:00", "г", True, {**base, "entry_hours": MAIN_SESSION}),
        Variant("г · без утра", "г", False, {**base, "entry_hours": (time(10), None)}),
        Variant(
            "г · без вечера", "г", False, {**base, "entry_hours": (None, time(19))}
        ),
        Variant(
            "д · без дней перед перерывом", "д", True, {**base, "skip_break_days": True}
        ),
    ]


def trend_variants(exit_key: str) -> list[Variant]:
    label, base = chosen_exit(exit_key)
    return [Variant(f"{label} без фильтра", "ref", True, base)] + [
        Variant(
            f"е · {days} дней", "е", days == TREND_DAYS[0], {**base, "trend_days": days}
        )
        for days in TREND_DAYS
    ]


def run(
    name: str, data: MarketData, variant: Variant, start: date | None = None
) -> Report:
    """Пробой одного рынка в одиночку. `start` — сигналы и кривая с этой даты."""
    if variant.slippage_ticks is not None:
        data = replace(
            data, spec=replace(data.spec, slippage_ticks=variant.slippage_ticks)
        )
    signals = breakout_signals(data.bars, **variant.params).with_columns(
        pl.lit(name).alias("market")
    )
    bars = data.bars
    if start is not None:
        signals = signals.filter(pl.col("timestamp").dt.date() >= start)
        bars = bars.filter(pl.col("timestamp").dt.date() >= start)
    result = run_portfolio(signals, {name: data}, CAPITAL)
    return build_report(result, market_timeline([bars]))


def run_all(
    name: str, data: MarketData, variants: list[Variant], start: date | None = None
) -> dict[str, Report]:
    reports = {}
    for i, variant in enumerate(variants, 1):
        print(f"  {name}: {i}/{len(variants)} {variant.label}", flush=True)
        reports[variant.label] = run(name, data, variant, start)
    return reports


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None or math.isnan(value) else round(value, digits)


def metrics_table(reports: dict[str, Report]) -> pl.DataFrame:
    rows = []
    for label, report in reports.items():
        s = report.summary
        rows.append(
            {
                "вариант": label,
                "сделок": s["trades"],
                "expectancy, R": _round(s["expectancy_r"], 3),
                "Sharpe": _round(s["sharpe"], 2),
                "±": _round(s["sharpe_se"], 2),
                "CAGR": format_value("cagr", s["cagr"]),
                "Max DD": format_value("max_drawdown", s["max_drawdown"]),
                "в позиции": format_value("exposure", s["exposure"]),
                "лонгов": report.trades.filter(pl.col("side") == LONG).height,
            }
        )
    return pl.DataFrame(rows)


def matrix(
    results: dict[str, dict[str, Report]], key: str, digits: int
) -> pl.DataFrame:
    """Вариант × рынок: одна метрика каждого прогона."""
    labels = list(next(iter(results.values())))
    table: dict[str, list[Any]] = {"вариант": labels}
    for name, reports in results.items():
        table[name] = [_round(reports[label].summary[key], digits) for label in labels]
    return pl.DataFrame(table)


def exit_reasons(
    results: dict[str, dict[str, Report]], labels: list[str]
) -> pl.DataFrame:
    rows = []
    for name, reports in results.items():
        for label in labels:
            trades = reports[label].trades
            counts = dict(trades.group_by("exit_reason").len().iter_rows())
            rows.append({"рынок": name, "вариант": label, **counts})
    return pl.DataFrame(rows).fill_null(0)


def free_path(path: Path) -> Path:
    """Путь, который не затирает прошлые запуски: _2, _3…"""
    if not path.exists():
        return path
    n = 2
    while path.with_stem(f"{path.stem}_{n}").exists():
        n += 1
    return path.with_stem(f"{path.stem}_{n}")


def _curves(
    ax, reports: dict[str, Report], variants: list[Variant], title: str
) -> None:
    _style(ax, title)
    neighbors: dict[str, int] = {}
    for variant in variants:
        curve = reports[variant.label].curve
        color = COLORS.get(variant.group, REFERENCE_COLOR)
        style: Any = "-"
        if not variant.central:
            k = neighbors.get(variant.group, 0)
            neighbors[variant.group] = k + 1
            style = NEIGHBOR_STYLES[k % len(NEIGHBOR_STYLES)]
        ax.plot(
            curve["timestamp"].to_numpy(),
            curve["equity"].to_numpy(),
            color=color,
            lw=1.8 if variant.central else 0.9,
            ls=style,
            alpha=1.0 if variant.central else 0.75,
            label=variant.label,
        )
    ax.axhline(CAPITAL, color=MUTED, lw=0.6)
    ax.yaxis.set_major_formatter(_rub)
    _dates(ax)
    ax.legend(fontsize=8, frameon=False, ncol=3, labelcolor=INK_SECONDARY)


def save_chart(
    name: str,
    stage: str,
    panels: list[tuple[str, dict[str, Report], list[Variant]]],
) -> Path:
    fig = Figure(figsize=(14, 6.5 * len(panels)), dpi=110, facecolor=SURFACE)
    for i, (title, reports, variants) in enumerate(panels, 1):
        _curves(fig.add_subplot(len(panels), 1, i), reports, variants, title)
    fig.suptitle(f"Подбор пробоя: {name}", x=0.06, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = free_path(REPORTS / f"tuning_{name}_{stage}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return path


def stage_stops(markets: dict[str, MarketData]) -> None:
    variants = stop_variants()
    results = {name: run_all(name, data, variants) for name, data in markets.items()}
    for name, reports in results.items():
        print(f"\n{name}:")
        print(metrics_table(reports))
    print("\nSharpe: вариант × рынок (погрешность одного прогона около ±0.47).")
    print(matrix(results, "sharpe", 2))
    print("\nExpectancy в R: вариант × рынок.")
    print(matrix(results, "expectancy_r", 3))
    print("\nИздержки на сделку в R: вариант × рынок.")
    print(matrix(results, "costs_per_trade_r", 3))
    for name, reports in results.items():
        title = f"{name}: стоп 0.25–1 день, 1 и 2 шага проскальзывания"
        path = save_chart(name, "stops", [(title, reports, variants)])
        print(f"График: {path}")


def stage_exits(markets: dict[str, MarketData]) -> None:
    variants = exit_variants()
    results = {}
    for name, data in markets.items():
        results[name] = run_all(name, data, variants)
    for name, reports in results.items():
        print(f"\n{name}:")
        print(metrics_table(reports))
    print("\nSharpe: вариант × рынок (погрешность одного прогона около ±0.47).")
    print(matrix(results, "sharpe", 2))
    print("\nExpectancy в R: вариант × рынок.")
    print(matrix(results, "expectancy_r", 3))
    print("\nКак закрылись сделки центральных вариантов:")
    central = [v.label for v in variants if v.central]
    print(exit_reasons(results, central))
    for name, reports in results.items():
        path = save_chart(
            name, "exits", [(f"{name}: выход а, б, в", reports, variants)]
        )
        print(f"График: {path}")


def break_day_split(
    markets: dict[str, MarketData], reports: dict[str, Report]
) -> pl.DataFrame:
    """Сделки дня, который кончается перерывом, против остальных."""
    frames = []
    for name, data in markets.items():
        days = break_days(data.bars).implode()
        frames.append(
            reports[name]
            .trades.filter(pl.col("exit_reason") != "end")
            .select(
                pl.lit(name).alias("рынок"),
                pl.col("entry_time").dt.date().is_in(days).alias("перед перерывом"),
                "r_multiple",
            )
        )
    return pl.concat(frames)


def split_row(label: str, trades: pl.DataFrame) -> dict[str, Any]:
    """Разница expectancy «остальные − перед перерывом» и её погрешность."""
    before = trades.filter(pl.col("перед перерывом"))["r_multiple"]
    other = trades.filter(~pl.col("перед перерывом"))["r_multiple"]
    row: dict[str, Any] = {
        "рынок": label,
        "сделок перед перерывом": before.len(),
        "expectancy перед перерывом, R": _round(before.mean(), 3),
        "сделок в другие дни": other.len(),
        "expectancy в другие дни, R": _round(other.mean(), 3),
        "разница, R": None,
        "± погрешность": None,
        "хуже больше погрешности": None,
    }
    if before.len() >= 2 and other.len() >= 2:
        difference = other.mean() - before.mean()
        se = math.sqrt(before.var() / before.len() + other.var() / other.len())
        row["разница, R"] = round(difference, 3)
        row["± погрешность"] = round(se, 3)
        row["хуже больше погрешности"] = difference > se
    return row


def stage_entries(markets: dict[str, MarketData], exit_key: str) -> None:
    variants = entry_variants(exit_key)
    results = {name: run_all(name, data, variants) for name, data in markets.items()}
    for name, reports in results.items():
        print(f"\n{name}:")
        print(metrics_table(reports))
    print("\nSharpe: вариант × рынок.")
    print(matrix(results, "sharpe", 2))
    print("\nExpectancy в R: вариант × рынок.")
    print(matrix(results, "expectancy_r", 3))

    reference = {name: reports[variants[0].label] for name, reports in results.items()}
    split = break_day_split(markets, reference)
    print(
        "\nд. Входы в день, который кончается перерывом, против остальных "
        f"({variants[0].label}). Решение — по строке «вместе»:"
    )
    rows = [split_row(name, split.filter(pl.col("рынок") == name)) for name in markets]
    rows.append(split_row("вместе", split))
    print(pl.DataFrame(rows))

    trend_results = {}
    for name in (m for m in TREND_MARKETS if m in markets):
        data = markets[name]
        start = (
            long_trend(data.bars, max(TREND_DAYS))
            .filter(pl.col("trend_mean").is_not_null())["date"]
            .min()
        )
        print(
            f"\nе. {name}: с {start:%d.%m.%Y}, когда появилась средняя на {max(TREND_DAYS)} дней."
        )
        trend_results[name] = (
            start,
            run_all(name, data, trend_variants(exit_key), start),
        )
    for name, (start, reports) in trend_results.items():
        role = "решение" if name == TREND_DECISION else "для сведения"
        print(f"\nе. {name} ({role}), с {start:%d.%m.%Y}:")
        print(metrics_table(reports))

    for name, reports in results.items():
        panels = [(f"{name}: фильтры г и д", reports, variants)]
        if name in trend_results:
            start, trend_reports = trend_results[name]
            panels.append(
                (
                    f"{name}: фильтр е, с {start:%d.%m.%Y}",
                    trend_reports,
                    trend_variants(exit_key),
                )
            )
        print(f"График: {save_chart(name, 'entries', panels)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подбор пробоя канала (plan.md, 5.2).")
    parser.add_argument("--stage", choices=("stops", "exits", "entries"), required=True)
    parser.add_argument(
        "--exit",
        choices=list(EXITS),
        help="Выход для этапа entries: fixed — а, breakeven — б, trail — в.",
    )
    args = parser.parse_args()
    if args.stage == "entries" and args.exit is None:
        parser.error("для --stage entries нужен --exit: fixed, breakeven или trail")
    return args


def main() -> None:
    args = parse_args()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets(MARKETS)
    if args.stage == "stops":
        stage_stops(markets)
    elif args.stage == "exits":
        stage_exits(markets)
    else:
        stage_entries(markets, args.exit)


if __name__ == "__main__":
    main()
