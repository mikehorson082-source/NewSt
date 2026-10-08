"""«А что если без курса»: ошибка прибыли, если рубли за пункт — одно число.

Одни и те же сделки считаются тремя способами:
- «по свечам» — рубли за пункт каждой свечи, как биржа начисляет
  вариационную маржу. Так считает симулятор;
- «одно число» — рубли за пункт последней свечи ряда на всю историю.
  Так считал бы паспорт со стоимостью пункта из сегодняшнего снимка биржи;
- «курс входа» — рубли за пункт, известные в момент сигнала, на всю сделку.

Сделки условные, без стратегии: каждую неделю лонг одного контракта
с первой свечи недели до закрытия последней. Стоп и цель далеко (±50%
от цены), поэтому выход по времени. У CNY и MIX все три способа должны
совпасть: рубли за пункт у них постоянны. Ничего не записывает.
Запуск: uv run python -m backtest.checks.point_value
"""

from __future__ import annotations

import polars as pl

from backtest.markets import MARKETS, load_market
from backtest.position import LONG, Signal, simulate_position
from backtest.specs import ContractSpec

COLUMNS = [
    "timestamp",
    "contract",
    "open",
    "high",
    "low",
    "close",
    "price_adjustment",
    "is_first_bar_after_roll",
    "point_value",
    "point_value_known",
]


def weekly_trades(bars: pl.DataFrame) -> pl.DataFrame:
    """Бар сигнала и число баров удержания: лонг на каждую неделю ряда.

    Сигнал — последняя свеча прошлой недели, вход по открытию первой
    свечи недели, выход по закрытию её последней свечи.
    """
    return (
        bars.with_row_index("bar")
        .group_by(pl.col("timestamp").dt.truncate("1w").alias("week"))
        .agg(pl.col("bar").min().alias("first_bar"), pl.len().alias("bars"))
        .filter(pl.col("first_bar") > 0)
        .sort("first_bar")
        .select((pl.col("first_bar") - 1).alias("signal_bar"), "bars")
    )


def net_pnl(
    spec: ContractSpec,
    window: pl.DataFrame,
    max_bars: int,
    point_value: float | None = None,
) -> float:
    """Результат сделки; `point_value` — одно число на всю сделку вместо свечей."""
    if point_value is not None:
        window = window.with_columns(pl.lit(point_value).alias("point_value"))
    close = float(window["close"][0])
    signal = Signal(side=LONG, stop=close * 0.5, target=close * 1.5, max_bars=max_bars)
    return simulate_position(spec, window, 0, signal).net_pnl


def rub(value: float) -> str:
    return f"{value:+,.0f} ₽".replace(",", " ")


def compare(name: str) -> None:
    params, bars = load_market(name)
    spec = params.contract_spec()
    bars = bars.select(COLUMNS)
    single = float(bars["point_value"][-1])

    rows = []
    for signal_bar, hold in weekly_trades(bars).iter_rows():
        window = bars.slice(signal_bar, hold + 1)
        rows.append(
            {
                "week": window["timestamp"][1].date(),
                "pv_first": window["point_value"][1],
                "pv_last": window["point_value"][-1],
                "by_bar": net_pnl(spec, window, hold),
                "single": net_pnl(spec, window, hold, single),
                "entry": net_pnl(
                    spec, window, hold, float(window["point_value_known"][0])
                ),
            }
        )
    trades = pl.DataFrame(rows).with_columns(
        (pl.col("single") - pl.col("by_bar")).abs().alias("err_single"),
        (pl.col("entry") - pl.col("by_bar")).abs().alias("err_entry"),
    )

    summary = [
        pl.len().alias("сделок"),
        pl.col("by_bar").sum().round(0).alias("по свечам ₽"),
        pl.col("single").sum().round(0).alias("одно число ₽"),
        pl.col("entry").sum().round(0).alias("курс входа ₽"),
        pl.col("err_single").mean().round(0).alias("ош. одно число ₽"),
        pl.col("err_entry").mean().round(0).alias("ош. курс входа ₽"),
    ]
    by_year = (
        trades.group_by(pl.col("week").dt.year().cast(pl.String).alias("год"))
        .agg(summary)
        .sort("год")
    )
    total = trades.select(pl.lit("всего").alias("год"), *summary)

    last_day = bars["timestamp"][-1].date()
    print(
        f"\n{name}: одно число = {single:.2f} ₽ за пункт (последняя свеча {last_day})"
    )
    print(pl.concat([by_year, total]))

    typical = trades["by_bar"].abs().mean()
    if typical:
        print(
            "Средняя ошибка сделки от среднего размера её результата: "
            f"одно число {trades['err_single'].mean() / typical:.1%}, "
            f"курс входа {trades['err_entry'].mean() / typical:.1%}"
        )
    worst = trades.sort("err_entry", descending=True).row(0, named=True)
    if worst["err_entry"] > 0:
        print(
            f"Самая большая ошибка «курс входа»: неделя с {worst['week']}, "
            f"₽ за пункт {worst['pv_first']:.2f} → {worst['pv_last']:.2f}, "
            f"по свечам {rub(worst['by_bar'])}, по курсу входа {rub(worst['entry'])}"
        )


def main() -> None:
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_thousands_separator(" ")
    print(
        "Лонг 1 контракта на каждую неделю, результат после издержек.\n"
        "ош. — средняя по модулю ошибка одной сделки против расчёта по свечам."
    )
    for name in MARKETS:
        compare(name)


if __name__ == "__main__":
    main()
