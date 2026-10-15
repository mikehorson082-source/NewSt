"""Роль волатильности в пробое (plan.md, 5.3.1).

Замороженные правила 5.3 на cnyrub, mix, brent, gold, каждый рынок в одиночку,
вся история. Правила не меняются: это проверки.

1. `--part stop` — нужна ли волатильность в стопе. Стоп 0.75 дневного движения
   против неподвижного стопа в процентах цены. Неподвижный стоп — медиана
   нынешнего стопа этого рынка по всем сигналам истории (медиана знает всю
   историю: для правила это подглядывание, для проверки допустимо). Сигналы
   те же, меняется только стоп. 1 и 2 шага проскальзывания.
   Ориентир: неподвижный стоп хуже по Sharpe на 3 рынках из 4 — волатильность
   в стопе нужна, оставляем как есть.
2. `--part regimes` — тихий рынок: входить или нет. Сделки по месту σ свечи
   сигнала среди значений рынка за прошлый год (`rules.volatility_rank`):
   тихо — нижняя треть, обычно, бурно — верхняя треть. Сделки первого года
   данных — без группы. Результат в R после и до издержек.
   Ориентир — вместе по четырём рынкам: «тихо» хуже остальных больше чем на
   одну погрешность разницы и после издержек, и до — тогда прогон правила с
   пустышкой. «Бурно» хуже остальных так же — запись в идеи шага 6.

Ничего не записывает.
Запуск: uv run python strategies/st01/check_volatility.py [--part stop|regimes]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from typing import Any

import polars as pl
from check_breakout import closed
from rules import FROZEN_STOP_DAYS, frozen_breakout_signals, volatility_rank
from run_baseline import CAPITAL
from tune_breakout import MARKETS, _round

from backtest.portfolio import MarketData, load_markets, run_portfolio
from backtest.report import Report, build_report, format_value, market_timeline

SLIPPAGE_TICKS = (1.0, 2.0)
# Границы групп по месту σ за год: нижняя и верхняя треть.
QUIET, STORMY = 1 / 3, 2 / 3
GROUPS = ("тихо", "обычно", "бурно")
NO_HISTORY = "нет года истории"
MAIN_STOP = f"стоп {FROZEN_STOP_DAYS:g} дня"
FIXED_STOP = "стоп в % цены"


def run(
    name: str, data: MarketData, signals: pl.DataFrame, ticks: float = 1.0
) -> Report:
    """Сигналы одного рынка через портфель при `ticks` шагах проскальзывания."""
    data = replace(data, spec=replace(data.spec, slippage_ticks=ticks))
    signals = signals.with_columns(pl.lit(name).alias("market"))
    result = run_portfolio(signals, {name: data}, CAPITAL)
    return build_report(result, market_timeline([data.bars]))


def median_stop_pct(data: MarketData, signals: pl.DataFrame) -> float:
    """Медиана расстояния до стопа в долях цены закрытия по всем сигналам."""
    joined = signals.join(data.bars.select("timestamp", "close"), on="timestamp")
    return float(((joined["close"] - joined["stop"]).abs() / joined["close"]).median())


def regime(rank: pl.Expr) -> pl.Expr:
    return (
        pl.when(rank.is_null())
        .then(pl.lit(NO_HISTORY))
        .when(rank < QUIET)
        .then(pl.lit(GROUPS[0]))
        .when(rank < STORMY)
        .then(pl.lit(GROUPS[1]))
        .otherwise(pl.lit(GROUPS[2]))
    )


def with_regime(name: str, ranks: pl.DataFrame, report: Report) -> pl.DataFrame:
    """Закрытые сделки с группой по месту σ свечи сигнала за прошлый год."""
    return (
        closed(report.trades)
        .join(ranks, on="signal_time", how="left")
        .select(
            pl.lit(name).alias("рынок"),
            regime(pl.col("vol_rank")).alias("группа"),
            "vol_rank",
            "r_multiple",
            (pl.col("pnl_before_costs") / pl.col("risk")).alias("r_before"),
            (pl.col("costs") / pl.col("risk")).alias("costs_r"),
        )
    )


def signal_ranks(data: MarketData) -> pl.DataFrame:
    return volatility_rank(data.bars).select(
        pl.col("timestamp").alias("signal_time"), "vol_rank"
    )


def _order(column: str, values: tuple[str, ...]) -> pl.Expr:
    return pl.col(column).replace_strict(
        {v: i for i, v in enumerate(values)}, default=len(values)
    )


def group_table(trades: pl.DataFrame, by: tuple[str, ...]) -> pl.DataFrame:
    """Сделок, R после и до издержек, издержки в R по группам."""
    return (
        trades.group_by(*by)
        .agg(
            pl.len().alias("сделок"),
            pl.col("r_multiple").mean().round(3).alias("R после издержек"),
            pl.col("r_before").mean().round(3).alias("R до издержек"),
            pl.col("costs_r").mean().round(3).alias("издержки, R"),
        )
        .sort(
            *[_order(c, MARKETS + ("вместе",) if c == "рынок" else GROUPS) for c in by]
        )
    )


def _difference(group: pl.Series, rest: pl.Series) -> tuple[float, float] | None:
    """«Остальные − группа» и погрешность разницы. Плюс — группа хуже."""
    if group.len() < 2 or rest.len() < 2:
        return None
    se = math.sqrt(group.var() / group.len() + rest.var() / rest.len())
    return rest.mean() - group.mean(), se


def compare_row(label: str, trades: pl.DataFrame, group: str) -> dict[str, Any]:
    """Группа против остальных сделок с известным местом σ."""
    ranked = trades.filter(pl.col("группа") != NO_HISTORY)
    inside = ranked.filter(pl.col("группа") == group)
    rest = ranked.filter(pl.col("группа") != group)
    after = _difference(inside["r_multiple"], rest["r_multiple"])
    before = _difference(inside["r_before"], rest["r_before"])
    row: dict[str, Any] = {
        "рынок": label,
        "группа": group,
        "сделок": inside.height,
        "остальных": rest.height,
        "хуже на, R (после)": None,
        "± (после)": None,
        "хуже на, R (до)": None,
        "± (до)": None,
        "хуже больше погрешности": None,
    }
    if after is not None and before is not None:
        row["хуже на, R (после)"] = round(after[0], 3)
        row["± (после)"] = round(after[1], 3)
        row["хуже на, R (до)"] = round(before[0], 3)
        row["± (до)"] = round(before[1], 3)
        row["хуже больше погрешности"] = after[0] > after[1] and before[0] > before[1]
    return row


def part_stop(markets: dict[str, MarketData]) -> None:
    print("\n1. Нужна ли волатильность в стопе")
    rows = []
    by_regime = []
    sharpe: dict[str, dict[str, float]] = {}
    for name, data in markets.items():
        main = frozen_breakout_signals(name, data.bars)
        pct = median_stop_pct(data, main)
        fixed = frozen_breakout_signals(name, data.bars, stop_pct=pct)
        ranks = signal_ranks(data)
        print(f"  {name}: неподвижный стоп {pct * 100:.2f}% цены", flush=True)
        for ticks in SLIPPAGE_TICKS:
            for label, signals in ((MAIN_STOP, main), (FIXED_STOP, fixed)):
                report = run(name, data, signals, ticks)
                s = report.summary
                trades = closed(report.trades)
                rows.append(
                    {
                        "рынок": name,
                        "шагов": ticks,
                        "стоп": label,
                        "% цены": f"{pct * 100:.2f}" if label == FIXED_STOP else "по σ",
                        "сделок": s["trades"],
                        "expectancy, R": _round(s["expectancy_r"], 3),
                        "Sharpe": _round(s["sharpe"], 2),
                        "±": _round(s["sharpe_se"], 2),
                        "Max DD": format_value("max_drawdown", s["max_drawdown"]),
                        "по стопу, %": round(
                            (trades["exit_reason"] == "stop").mean() * 100, 1
                        ),
                        "издержки, R": _round(s["costs_per_trade_r"], 3),
                    }
                )
                if ticks == SLIPPAGE_TICKS[0]:
                    sharpe.setdefault(name, {})[label] = s["sharpe"]
                    by_regime.append(
                        with_regime(name, ranks, report).with_columns(
                            pl.lit(label).alias("стоп")
                        )
                    )
    print(pl.DataFrame(rows))

    trades = pl.concat(by_regime).filter(pl.col("группа") != NO_HISTORY)
    print(
        "\nExpectancy в R по месту σ за год, 1 шаг, вместе по четырём рынкам "
        "(сделки первого года — без группы):"
    )
    print(
        trades.group_by("группа", "стоп")
        .agg(pl.len().alias("сделок"), pl.col("r_multiple").mean().round(3).alias("R"))
        .pivot(on="стоп", index="группа", values=["R", "сделок"])
        .sort(_order("группа", GROUPS))
    )

    worse = [n for n, v in sharpe.items() if v[FIXED_STOP] < v[MAIN_STOP]]
    print(
        f"\nНеподвижный стоп хуже по Sharpe (1 шаг) на {len(worse)} рынках из "
        f"{len(sharpe)}: {', '.join(worse) or '—'}."
    )
    if len(worse) >= 3:
        print("Ориентир: волатильность в стопе нужна — оставляем как есть.")
    else:
        print(
            "Ориентир: неподвижный стоп не хуже на 3 рынках из 4 — решает пользователь."
        )


def part_regimes(markets: dict[str, MarketData]) -> None:
    print("\n2. Тихий рынок: входить или нет (1 шаг проскальзывания)")
    frames = []
    for name, data in markets.items():
        report = run(name, data, frozen_breakout_signals(name, data.bars))
        frames.append(with_regime(name, signal_ranks(data), report))
    trades = pl.concat(frames)

    print("\nСделки по группам:")
    print(
        pl.concat(
            [
                group_table(trades, ("рынок", "группа")),
                group_table(
                    trades.with_columns(pl.lit("вместе").alias("рынок")),
                    ("рынок", "группа"),
                ),
            ]
        )
    )

    print("\nГруппа против остальных. Решение — по строкам «вместе»:")
    rows = []
    for group in (GROUPS[0], GROUPS[2]):
        for name in markets:
            rows.append(
                compare_row(name, trades.filter(pl.col("рынок") == name), group)
            )
        rows.append(compare_row("вместе", trades, group))
    table = pl.DataFrame(rows)
    print(table)

    verdict = {
        row["группа"]: row["хуже больше погрешности"]
        for row in table.filter(pl.col("рынок") == "вместе").iter_rows(named=True)
    }
    if verdict[GROUPS[0]]:
        print(
            "\n«Тихо» хуже остальных больше погрешности — нужен прогон правила с пустышкой."
        )
    else:
        print("\n«Тихо» не хуже остальных больше погрешности — правило не вводим.")
    if verdict[GROUPS[2]]:
        print("«Бурно» хуже остальных больше погрешности — в идеи шага 6.")
    else:
        print("«Бурно» не хуже остальных больше погрешности.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Роль волатильности (plan.md, 5.3.1).")
    parser.add_argument(
        "--part",
        choices=("stop", "regimes", "all"),
        default="all",
        help="stop — пункт 1, regimes — пункт 2, all — оба (по умолчанию).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets(MARKETS)
    if args.part in ("stop", "all"):
        part_stop(markets)
    if args.part in ("regimes", "all"):
        part_regimes(markets)


if __name__ == "__main__":
    main()
