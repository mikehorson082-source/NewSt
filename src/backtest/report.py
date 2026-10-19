"""Отчёт по прогону портфеля: кривая капитала, метрики, разбивка по годам.

Кривая капитала — на закрытии каждой 15-минутной свечи общего расписания
всех рынков: между сделками капитал стоит, внутри сделки идёт по её пути
(`ClosedTrade.path`). Max Drawdown считается по этой кривой — по одним
закрытиям сделок он занижен.

Sharpe, Sortino и Calmar — по дневной доходности (капитал на конец дня),
ставка 0: ставка уже в цене фьючерса. Перевод в годовые — по фактическому
числу торговых дней в году: с 2025 года есть сессии выходного дня, и 252
не подходит. Рядом с Sharpe — его погрешность (Mertens, с учётом асимметрии
и толстых хвостов) и вероятность, что настоящий Sharpe > 0 (PSR, ML4T 16.7):
на четырёх годах дневных данных Sharpe 1 бывает и случайностью.

Сделки, закрытые концом данных (`end`), рынок не закрывал: в кривой
капитала они есть (деньги были в риске), а в проценте выигрышных, Profit
Factor и expectancy — нет (ML4T 16, `09_performance_reporting`).

Запас по издержкам — во сколько раз издержки могли бы вырасти, прежде чем
результат до издержек обнулится. Путь сделок при этом тот же: при больших
издержках часть сделок не открылась бы, поэтому это верхняя граница.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from scipy import stats

from backtest.portfolio import PortfolioResult

DAYS_PER_YEAR = 365.25
# Выход по стопу, потому что порядок касаний в баре неизвестен.
PESSIMISTIC_EXITS = ("same_minute", "no_minutes", "minutes_mismatch")

# Подписи метрик для таблиц и графиков, в порядке вывода.
LABELS = {
    "start_capital": "Капитал в начале, ₽",
    "end_capital": "Капитал в конце, ₽",
    "total_return": "Доходность",
    "cagr": "CAGR",
    "max_drawdown": "Max Drawdown",
    "max_drawdown_rub": "Max Drawdown, ₽",
    "longest_drawdown_days": "Самая долгая просадка, дней",
    "sharpe": "Sharpe",
    "sharpe_se": "Погрешность Sharpe (±)",
    "psr": "Вероятность Sharpe > 0",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "exposure": "Доля времени в позиции",
    "trades": "Сделок",
    "win_rate": "Выигрышных",
    "profit_factor": "Profit Factor",
    "expectancy_rub": "Expectancy, ₽",
    "expectancy_r": "Expectancy, R",
    "avg_bars_held": "Свечей в сделке, среднее",
    "pnl_before_costs": "Результат до издержек, ₽",
    "costs": "Издержки, ₽",
    "net_pnl": "Результат после издержек, ₽",
    "costs_per_trade_r": "Издержки на сделку, R",
    "cost_headroom": "Запас по издержкам, раз",
    "pessimistic_stop_share": "Стоп по пессимизму",
    "winners_mae_r": "MAE выигрышных, R (медиана)",
    "losers_mfe_r": "MFE проигрышных, R (медиана)",
}
# Доли, которые показываются в процентах.
PERCENT = {
    "total_return",
    "cagr",
    "max_drawdown",
    "psr",
    "exposure",
    "win_rate",
    "pessimistic_stop_share",
}


@dataclass(frozen=True)
class Report:
    # timestamp, equity, in_position, positions — на закрытии каждой свечи расписания.
    curve: pl.DataFrame
    # date, equity, return — капитал на конец дня и доходность дня.
    daily: pl.DataFrame
    summary: dict[str, float | int | None]
    # Те же метрики по календарным годам; сделки — по году выхода.
    by_year: pl.DataFrame
    trades: pl.DataFrame  # PortfolioResult.trade_log()


def market_timeline(bars: Iterable[pl.DataFrame]) -> pl.Series:
    """Общее расписание: все метки свечей всех рынков, по времени."""
    return (
        pl.concat([frame.select("timestamp") for frame in bars])
        .unique()
        .sort("timestamp")["timestamp"]
    )


def equity_curve(result: PortfolioResult, timeline: pl.Series) -> pl.DataFrame:
    """Капитал на закрытии каждой свечи расписания.

    Метка свечи — её начало, как в рядах; капитал — на её закрытии. Свечи
    других рынков внутри сделки берут последнюю оценку сделки. Капитал —
    стартовый плюс приращения путей всех сделок: так кривая верна и при
    нескольких открытых позициях (plan.md, 5.5). `positions` — сколько
    позиций открыто на свече.
    """
    frame = pl.DataFrame({"timestamp": timeline}).sort("timestamp")
    steps = []
    for item in result.trades:
        previous = 0.0
        for timestamp, value in item.trade.path:
            steps.append((timestamp, value - previous))
            previous = value
    if not steps:
        return frame.with_columns(
            pl.lit(result.initial_capital).alias("equity"),
            pl.lit(False).alias("in_position"),
            pl.lit(0, dtype=pl.Int64).alias("positions"),
        )
    marks = pl.DataFrame(
        steps, schema=["timestamp", "step"], orient="row"
    ).with_columns(pl.col("timestamp").cast(frame["timestamp"].dtype))
    missing = marks.join(frame, on="timestamp", how="anti")
    if not missing.is_empty():
        raise ValueError(
            f"{missing.height} свечей сделок нет в расписании "
            f"(первая {missing['timestamp'][0]})"
        )
    marks = (
        marks.group_by("timestamp")
        .agg(pl.col("step").sum())
        .sort("timestamp")
        .select(
            "timestamp",
            (result.initial_capital + pl.col("step").cum_sum()).alias("equity"),
        )
    )
    # Сделка открыта на свечах от входа до выхода включительно.
    times = frame["timestamp"].to_numpy()
    entries = np.sort(
        np.array([item.trade.entry_time for item in result.trades], dtype=times.dtype)
    )
    exits = np.sort(
        np.array([item.trade.exit_time for item in result.trades], dtype=times.dtype)
    )
    positions = np.searchsorted(entries, times, side="right") - np.searchsorted(
        exits, times, side="left"
    )
    return (
        frame.join_asof(marks, on="timestamp", strategy="backward")
        .with_columns(
            pl.col("equity").fill_null(result.initial_capital),
            pl.Series("positions", positions, dtype=pl.Int64),
        )
        .with_columns((pl.col("positions") > 0).alias("in_position"))
        .select("timestamp", "equity", "in_position", "positions")
    )


def daily_equity(curve: pl.DataFrame, initial_capital: float) -> pl.DataFrame:
    """Капитал на конец каждого дня расписания и доходность дня."""
    daily = (
        curve.group_by(pl.col("timestamp").dt.date().alias("date"))
        .agg(pl.col("equity").last())
        .sort("date")
    )
    previous = pl.col("equity").shift(1).fill_null(initial_capital)
    return daily.with_columns((pl.col("equity") / previous - 1.0).alias("return"))


def _drawdown(curve: pl.DataFrame, start_equity: float) -> tuple[float, float, float]:
    """Max Drawdown в долях и рублях и самая долгая просадка в днях.

    Длина просадки — время ниже прошлого максимума: от последней свечи, на
    которой капитал был на максимуме, до последней свечи под ним. Капитал,
    стоящий на максимуме между сделками, в просадке не считается.
    """
    peak = pl.max_horizontal(pl.col("equity").cum_max(), pl.lit(start_equity))
    # Момент последнего максимума: от него отсчитывается длина просадки.
    frame = curve.with_columns(peak.alias("_peak")).with_columns(
        pl.when(pl.col("equity") >= pl.col("_peak"))
        .then(pl.col("timestamp"))
        .forward_fill()
        .fill_null(curve["timestamp"][0])
        .alias("_peak_time")
    )
    worst = frame.select(
        (pl.col("equity") / pl.col("_peak") - 1.0).min().alias("pct"),
        (pl.col("equity") - pl.col("_peak")).min().alias("rub"),
        (pl.col("timestamp") - pl.col("_peak_time")).max().alias("duration"),
    ).row(0, named=True)
    days = worst["duration"].total_seconds() / 86_400 if worst["duration"] else 0.0
    return min(worst["pct"], 0.0), min(worst["rub"], 0.0), days


def _sharpe(
    returns: np.ndarray, periods_per_year: float
) -> tuple[float | None, float | None, float | None]:
    """Годовой Sharpe, его погрешность и вероятность, что настоящий Sharpe > 0.

    Погрешность — Mertens (2002): дисперсия дневного Sharpe
    (1 − асимметрия × SR + (эксцесс − 1) / 4 × SR²) / (T − 1), эксцесс
    обычный (у нормального 3). Переводится в годовую как и сам Sharpe.
    Автокорреляцию дневных доходностей не учитывает.
    """
    if returns.size < 3 or returns.std(ddof=1) == 0:
        return None, None, None
    sr = returns.mean() / returns.std(ddof=1)
    skew = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns, fisher=False))
    variance = (1 - skew * sr + (kurtosis - 1) / 4 * sr**2) / (returns.size - 1)
    scale = math.sqrt(periods_per_year)
    if variance <= 0:
        return sr * scale, None, None
    se = math.sqrt(variance)
    return sr * scale, se * scale, float(stats.norm.cdf(sr / se))


def _sortino(returns: np.ndarray, periods_per_year: float) -> float | None:
    """Как Sharpe, но в знаменателе только дни с убытком (цель 0)."""
    if returns.size < 3:
        return None
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2)))
    if downside == 0:
        return None
    return returns.mean() / downside * math.sqrt(periods_per_year)


def _median(series: pl.Series) -> float | None:
    return None if series.is_empty() else series.median()


def period_metrics(
    curve: pl.DataFrame,
    daily: pl.DataFrame,
    trades: pl.DataFrame,
    start_equity: float,
) -> dict[str, float | int | None]:
    """Метрики одного периода: весь прогон или один год."""
    end_equity = float(curve["equity"][-1])
    first: date = daily["date"][0]
    last: date = daily["date"][-1]
    years = ((last - first).days + 1) / DAYS_PER_YEAR
    growth = end_equity / start_equity
    cagr = growth ** (1 / years) - 1 if growth > 0 else -1.0
    max_dd, max_dd_rub, longest = _drawdown(curve, start_equity)

    returns = daily["return"].to_numpy()
    periods_per_year = daily.height / years
    sharpe, sharpe_se, psr = _sharpe(returns, periods_per_year)

    closed = trades.filter(pl.col("exit_reason") != "end")
    wins = closed.filter(pl.col("net_pnl") > 0)
    losses = closed.filter(pl.col("net_pnl") < 0)
    gross_loss = -losses["net_pnl"].sum()
    before_costs = trades["pnl_before_costs"].sum()
    costs = trades["costs"].sum()
    return {
        "start_capital": start_equity,
        "end_capital": end_equity,
        "total_return": growth - 1,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "max_drawdown_rub": max_dd_rub,
        "longest_drawdown_days": longest,
        "sharpe": sharpe,
        "sharpe_se": sharpe_se,
        "psr": psr,
        "sortino": _sortino(returns, periods_per_year),
        "calmar": cagr / -max_dd if max_dd < 0 else None,
        "exposure": curve["in_position"].mean(),
        "trades": closed.height,
        "win_rate": wins.height / closed.height if closed.height else None,
        "profit_factor": wins["net_pnl"].sum() / gross_loss if gross_loss > 0 else None,
        "expectancy_rub": closed["net_pnl"].mean(),
        "expectancy_r": closed["r_multiple"].mean(),
        "avg_bars_held": closed["bars_held"].mean(),
        "pnl_before_costs": before_costs,
        "costs": costs,
        "net_pnl": trades["net_pnl"].sum(),
        "costs_per_trade_r": (trades["costs"] / trades["risk"]).mean(),
        "cost_headroom": before_costs / costs
        if costs > 0 and before_costs > 0
        else None,
        "pessimistic_stop_share": (
            closed["exit_resolution"].is_in(PESSIMISTIC_EXITS).mean()
            if closed.height
            else None
        ),
        "winners_mae_r": _median(wins["mae_r"]),
        "losers_mfe_r": _median(losses["mfe_r"]),
    }


def _check_reconciliation(result: PortfolioResult, curve: pl.DataFrame) -> None:
    """Сверка: сумма сделок = изменение капитала = конец кривой."""
    total = sum(item.trade.net_pnl for item in result.trades)
    change = result.final_capital - result.initial_capital
    if not math.isclose(total, change, rel_tol=1e-9, abs_tol=1e-6):
        raise AssertionError(f"Сумма сделок {total} ≠ изменение капитала {change}")
    if not math.isclose(curve["equity"][-1], result.final_capital, abs_tol=1e-6):
        raise AssertionError("Конец кривой капитала ≠ капитал после сделок")


def build_report(result: PortfolioResult, timeline: pl.Series) -> Report:
    """Кривая капитала, метрики за весь прогон и по годам."""
    curve = equity_curve(result, timeline)
    _check_reconciliation(result, curve)
    daily = daily_equity(curve, result.initial_capital)
    trades = result.trade_log()
    if trades.is_empty():
        trades = pl.DataFrame(
            schema={
                "exit_time": curve["timestamp"].dtype,
                "exit_reason": pl.String,
                "exit_resolution": pl.String,
                "net_pnl": pl.Float64,
                "r_multiple": pl.Float64,
                "bars_held": pl.Int64,
                "pnl_before_costs": pl.Float64,
                "costs": pl.Float64,
                "risk": pl.Float64,
                "mae_r": pl.Float64,
                "mfe_r": pl.Float64,
            }
        )
    summary = period_metrics(curve, daily, trades, result.initial_capital)

    rows = []
    start_equity = result.initial_capital
    for year in daily["date"].dt.year().unique().sort():
        curve_year = curve.filter(pl.col("timestamp").dt.year() == year)
        rows.append(
            {
                "year": year,
                **period_metrics(
                    curve_year,
                    daily.filter(pl.col("date").dt.year() == year),
                    trades.filter(pl.col("exit_time").dt.year() == year),
                    start_equity,
                ),
            }
        )
        start_equity = float(curve_year["equity"][-1])
    by_year = pl.DataFrame(rows, infer_schema_length=None)
    return Report(
        curve=curve, daily=daily, summary=summary, by_year=by_year, trades=trades
    )


def format_value(key: str, value: float | None) -> str:
    """Значение метрики для таблицы: проценты, рубли с пробелами, R."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if key in PERCENT:
        return f"{value * 100:.1f}%"
    if key == "trades":
        return str(value)
    if key == "longest_drawdown_days":
        return f"{value:.0f}"
    if abs(value) >= 1_000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.2f}"
