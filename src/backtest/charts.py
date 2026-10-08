"""Графики отчёта: одна картинка PNG с шестью панелями.

1. Капитал на закрытии каждой свечи.
2. Просадка от прошлого максимума — сколько и как долго был под водой.
3. Sharpe в скользящем окне полгода — ровный ли результат или сделан
   одним отрезком.
4. Доходность по месяцам — какие месяцы сделали результат.
5. Распределение дневных доходностей — у стратегии, которая большую часть
   времени вне рынка, много нулевых дней.
6. MAE против MFE каждой сделки в R — насколько сделки уходили против нас
   и в нашу пользу (для стопов и целей шага 4).

Панели 1–5 — книга ML4T, глава 16 `09_performance_reporting`; панель 6 —
глава 19 `03_position_sizing_mae_mfe`. Цвета — проверенная палитра навыка
dataviz (синий и оранжевый проходят проверку на дальтонизм).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure

from backtest.report import Report, format_value

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"  # категория 1
ORANGE = "#eb6834"  # категория 2
AQUA = "#1baf7a"  # категория 3
RED = "#e34948"  # категория 8: просадка
# Первые три категории палитры проходят проверку на дальтонизм попарно.
SERIES = (BLUE, ORANGE, AQUA)
# Расходящаяся шкала: синий — плюс, красный — минус, серый — ноль.
DIVERGING = LinearSegmentedColormap.from_list(
    "returns", ["#e34948", "#f0efec", "#2a78d6"]
)
ROLLING_DAYS = 126  # полгода торговых дней
# Рамка графика MAE/MFE в R: редкие гэпы через стоп (−5R) сжали бы облако
# сделок в узкую полосу. Сделки за рамкой считаются в заголовке.
MAE_FLOOR = -2.5
MFE_CEILING = 5.0
MONTHS = [
    "янв",
    "фев",
    "мар",
    "апр",
    "май",
    "июн",
    "июл",
    "авг",
    "сен",
    "окт",
    "ноя",
    "дек",
]


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def _rub(value: float, _position=None) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _date_label(value: float, _position=None) -> str:
    """Подпись даты по-русски: год на январе, иначе месяц."""
    moment = mdates.num2date(value)
    if moment.month == 1 and moment.day == 1:
        return str(moment.year)
    return MONTHS[moment.month - 1]


def _dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    ax.xaxis.set_major_formatter(_date_label)


def _equity(ax, report: Report) -> None:
    curve = report.curve
    ax.plot(
        curve["timestamp"].to_numpy(), curve["equity"].to_numpy(), color=BLUE, lw=1.4
    )
    ax.axhline(report.summary["start_capital"], color=BASELINE, lw=1, ls="--")
    ax.yaxis.set_major_formatter(_rub)
    _dates(ax)
    _style(ax, "Капитал на закрытии каждой свечи, ₽ (пунктир — стартовый)")


def _drawdown(ax, report: Report) -> None:
    curve = report.curve
    peak = np.maximum.accumulate(
        np.maximum(curve["equity"].to_numpy(), report.summary["start_capital"])
    )
    drawdown = (curve["equity"].to_numpy() / peak - 1) * 100
    times = curve["timestamp"].to_numpy()
    ax.fill_between(times, drawdown, 0, color=RED, alpha=0.25, linewidth=0)
    ax.plot(times, drawdown, color=RED, lw=1)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    _dates(ax)
    s = report.summary
    _style(
        ax,
        f"Просадка от прошлого максимума: худшая {format_value('max_drawdown', s['max_drawdown'])}, "
        f"самая долгая {s['longest_drawdown_days']:.0f} дн.",
    )


def _rolling_sharpe(ax, report: Report) -> None:
    daily = report.daily
    years = ((daily["date"][-1] - daily["date"][0]).days + 1) / 365.25
    scale = math.sqrt(daily.height / years)
    rolling = daily.select(
        "date",
        (
            pl.col("return").rolling_mean(ROLLING_DAYS)
            / pl.col("return").rolling_std(ROLLING_DAYS)
            * scale
        ).alias("sharpe"),
    ).drop_nulls()
    rolling = rolling.filter(pl.col("sharpe").is_finite())
    if not rolling.is_empty():
        ax.plot(
            rolling["date"].to_numpy(), rolling["sharpe"].to_numpy(), color=BLUE, lw=1.4
        )
    ax.axhline(0, color=BASELINE, lw=1)
    _dates(ax)
    _style(ax, f"Sharpe в окне {ROLLING_DAYS} торговых дней (годовой)")


def _daily_histogram(ax, report: Report) -> None:
    returns = report.daily["return"].to_numpy() * 100
    flat = int(np.sum(returns == 0))
    moved = returns[returns != 0]
    if moved.size:
        ax.hist(moved, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    _style(
        ax,
        f"Дневная доходность: {moved.size} дней с движением, {flat} без (вне рынка)",
    )
    ax.set_ylabel("дней", color=MUTED, fontsize=8.5)


def _monthly(ax, report: Report) -> None:
    monthly = (
        report.daily.group_by(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .agg(pl.col("equity").last(), pl.col("date").max())
        .sort("date")
        .with_columns(
            (
                pl.col("equity")
                / pl.col("equity").shift(1).fill_null(report.summary["start_capital"])
                - 1
            ).alias("return")
        )
    )
    years = monthly["year"].unique().sort().to_list()
    grid = np.full((len(years), 12), np.nan)
    for year, month, value in monthly.select("year", "month", "return").iter_rows():
        grid[years.index(year), month - 1] = value * 100
    limit = max(np.nanmax(np.abs(grid)), 0.1) if np.isfinite(grid).any() else 1.0
    ax.imshow(
        grid,
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    for i in range(len(years)):
        for j in range(12):
            if np.isfinite(grid[i, j]):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=INK,
                )
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(len(years)), [str(year) for year in years])
    _style(ax, "Доходность по месяцам, % (синий — плюс, красный — минус)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _excursions(ax, report: Report) -> None:
    trades = report.trades
    closed = trades.filter(pl.col("exit_reason") != "end")
    wins = closed.filter(pl.col("net_pnl") > 0)
    losses = closed.filter(pl.col("net_pnl") <= 0)
    ended = trades.filter(pl.col("exit_reason") == "end")
    style = {"s": 10, "alpha": 0.5, "linewidths": 0}
    # Сделки, закрытые концом данных («купил и держи»), — серым, крупнее:
    # рынок их не закрывал, в процент выигрышных они не входят.
    if not ended.is_empty():
        ax.scatter(
            ended["mae_r"],
            ended["mfe_r"],
            color=MUTED,
            s=60,
            label=f"закрыты концом данных ({ended.height})",
            zorder=4,
        )
    ax.scatter(
        losses["mae_r"],
        losses["mfe_r"],
        color=ORANGE,
        label=f"проигрышные ({losses.height})",
        **style,
    )
    ax.scatter(
        wins["mae_r"],
        wins["mfe_r"],
        color=BLUE,
        label=f"выигрышные ({wins.height})",
        **style,
    )
    ax.axvline(-1, color=MUTED, lw=1, ls="--")
    if not trades.is_empty():
        ax.set_xlim(max(MAE_FLOOR, trades["mae_r"].min() - 0.1), 0.1)
        ax.set_ylim(-0.1, min(MFE_CEILING, trades["mfe_r"].max() + 0.2))
    outside = trades.filter(
        (pl.col("mae_r") < MAE_FLOOR) | (pl.col("mfe_r") > MFE_CEILING)
    ).height
    ax.set_xlabel(
        "MAE, R — насколько уходила против нас (пунктир — стоп, −1R)",
        color=MUTED,
        fontsize=8.5,
    )
    ax.set_ylabel("MFE, R — насколько в нашу пользу", color=MUTED, fontsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    # Легенда под осью: облако сделок может занять любой угол графика.
    legend = ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=len(handles),
        fontsize=8.5,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _style(
        ax,
        f"MAE и MFE сделок в R (за рамкой {outside})"
        if outside
        else "MAE и MFE сделок в R",
    )


def _trades_text(report: Report) -> str:
    """Сделки для заголовка: закрытые рынком и закрытые концом данных."""
    s = report.summary
    ended = report.trades.filter(pl.col("exit_reason") == "end").height
    text = f"сделок {s['trades']}"
    if ended:
        text += f" (+{ended} закрыта концом данных)"
    if s["expectancy_r"] is not None:
        text += f"  ·  expectancy {format_value('expectancy_r', s['expectancy_r'])}R"
    return text


def save_report_chart(report: Report, path: Path, title: str) -> Path:
    """Рисует шесть панелей отчёта в один PNG и возвращает путь."""
    fig = Figure(figsize=(14, 17), dpi=110, facecolor=SURFACE)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[3, 1.6, 2.4, 2.6],
        hspace=0.42,
        wspace=0.18,
        top=0.93,
        bottom=0.04,
    )
    _equity(fig.add_subplot(grid[0, :]), report)
    _drawdown(fig.add_subplot(grid[1, :]), report)
    _rolling_sharpe(fig.add_subplot(grid[2, 0]), report)
    _daily_histogram(fig.add_subplot(grid[2, 1]), report)
    _monthly(fig.add_subplot(grid[3, 0]), report)
    _excursions(fig.add_subplot(grid[3, 1]), report)

    s = report.summary
    se = s["sharpe_se"]
    sharpe = format_value("sharpe", s["sharpe"]) + (f" ± {se:.2f}" if se else "")
    fig.suptitle(title, x=0.06, y=0.978, ha="left", fontsize=15, color=INK)
    fig.text(
        0.06,
        0.962,
        f"Доходность {format_value('total_return', s['total_return'])}  ·  "
        f"CAGR {format_value('cagr', s['cagr'])}  ·  "
        f"Max DD {format_value('max_drawdown', s['max_drawdown'])}  ·  "
        f"Sharpe {sharpe}  ·  P(Sharpe > 0) {format_value('psr', s['psr'])}  ·  "
        f"{_trades_text(report)}",
        fontsize=10,
        color=INK_SECONDARY,
        va="top",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return path


def save_cost_grid_chart(
    grid: pl.DataFrame,
    path: Path,
    title: str,
    current: tuple[float, float] = (1.0, 1.0),
) -> Path:
    """Expectancy после издержек против проскальзывания, линия на каждый сбор.

    В R на сделку, а не в рублях: рубли упираются в «дно», когда капитал
    кончается, и линии сливаются. `grid` — `sensitivity.cost_grid`, не больше
    трёх множителей сбора.
    `current` — нынешнее предположение (шаги, множитель сбора), отмечено точкой.
    """
    if "impact" in grid.columns:
        grid = grid.filter(pl.col("impact") == 0.0)
    fees = grid["fee_multiplier"].unique().sort().to_list()
    if len(fees) > len(SERIES):
        raise ValueError(f"Не больше {len(SERIES)} множителей сбора на графике")
    fig = Figure(figsize=(9, 5.2), dpi=110, facecolor=SURFACE)
    ax = fig.add_subplot()
    for color, fee in zip(SERIES, fees, strict=False):
        line = grid.filter(pl.col("fee_multiplier") == fee).sort("slippage_ticks")
        x = line["slippage_ticks"].to_numpy()
        y = line["expectancy_r"].to_numpy()
        label = f"сбор ×{fee:g}"
        ax.plot(x, y, color=color, lw=1.8, marker="o", ms=5, label=label)
        ax.annotate(
            label,
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    now = grid.filter(
        (pl.col("slippage_ticks") == current[0])
        & (pl.col("fee_multiplier") == current[1])
    )
    if not now.is_empty():
        ax.scatter(
            [current[0]],
            [now["expectancy_r"][0]],
            s=90,
            facecolor="none",
            edgecolor=INK,
            linewidths=1.2,
            zorder=5,
        )
        ax.annotate(
            "сейчас в бэктесте",
            (current[0], now["expectancy_r"][0]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    ax.axhline(0, color=BASELINE, lw=1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.2f}R" if v else "0")
    ax.set_xlabel("проскальзывание, шагов цены на сторону", color=MUTED, fontsize=9)
    ax.set_ylabel("expectancy после издержек, R на сделку", color=MUTED, fontsize=9)
    legend = ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _style(ax, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return path
