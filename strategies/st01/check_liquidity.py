"""Ликвидность рынков экзамена до экзамена (plan.md, 5.4).

Результатов стратегии здесь нет: ни сделок, ни прибыли. Только объёмы и
размер позиции по сигналам замороженных правил 5.3.

По каждому рынку, данные до `--before`:
- дневной оборот — медиана по дням, объём × цена × рубли за пункт;
- обычный объём свечи входа — медиана `usual_volume` по сигналам пробоя;
- доля сигналов без объёма — 0 контрактов по лимиту 10% обычного объёма
  (или обычный объём неизвестен: мало истории);
- доля сигналов, урезанных объёмом, — контрактов больше 0, но меньше, чем
  позволили бы риск и ГО.

Размер — от стартового капитала 1 000 000 ₽ по каждому сигналу, как будто
позиция свободна: сделок здесь нет, и какие сигналы взял бы портфель,
неизвестно. Сигнал на последней свече ряда пропускается: свечи входа нет.

Рынок идёт на экзамен, если сигналов без объёма не больше 10%. Рынки
подбора (cnyrub, mix, brent, gold) — для сравнения, их не отбираем.

Ничего не записывает.
Запуск: uv run python strategies/st01/check_liquidity.py [--before 2026-01-01]
"""

from __future__ import annotations

import argparse
from datetime import date

import polars as pl
from rules import frozen_breakout_signals
from run_baseline import CAPITAL, cut

from backtest.markets import EXAM_MARKETS, MARKETS
from backtest.portfolio import MarketData, load_markets
from backtest.position import HOLD_ANY, Signal
from backtest.sizing import VOLUME, size_position

EXAM_BEFORE = date(2026, 1, 1)
MAX_NO_VOLUME_SHARE = 0.10


def signal_sizes(name: str, data: MarketData) -> pl.DataFrame:
    """Размер позиции по каждому сигналу пробоя: контракты, лимит, обычный объём."""
    bars = data.bars
    index = {ts: i for i, ts in enumerate(bars["timestamp"])}
    rows = []
    for row in frozen_breakout_signals(name, bars).iter_rows(named=True):
        bar = index[row["timestamp"]]
        if bar == bars.height - 1:
            continue
        signal = Signal(
            side=row["side"],
            stop=row["stop"],
            target=row["target"],
            max_bars=row.get("max_bars"),
            hold=row.get("hold") or HOLD_ANY,
            breakeven_r=row.get("breakeven_r"),
            trail=row.get("trail"),
        )
        size = size_position(data.spec, bars, bar, signal, CAPITAL)
        rows.append(
            {
                "contracts": size.contracts,
                "limit": size.limit,
                "usual_volume": size.usual_volume,
                "without_volume": min(size.contracts_by_risk, size.contracts_by_margin),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "contracts": pl.Int64,
            "limit": pl.String,
            "usual_volume": pl.Float64,
            "without_volume": pl.Int64,
        },
    )


def daily_turnover(data: MarketData) -> float:
    """Медиана дневного оборота в рублях."""
    return (
        data.bars.group_by(pl.col("timestamp").dt.date())
        .agg((pl.col("volume") * pl.col("close") * pl.col("point_value")).sum())
        .select(pl.col("volume").median())
        .item()
    )


def market_row(name: str, data: MarketData) -> dict[str, object]:
    sizes = signal_sizes(name, data)
    by_volume = sizes.filter(pl.col("limit") == VOLUME)
    no_volume = by_volume.filter(pl.col("contracts") == 0).height
    trimmed = by_volume.filter(
        (pl.col("contracts") > 0) & (pl.col("contracts") < pl.col("without_volume"))
    ).height
    share = no_volume / sizes.height if sizes.height else None
    exam = name in EXAM_MARKETS
    if not exam:
        verdict = "рынок подбора"
    elif share is None:
        verdict = "нет сигналов"
    else:
        verdict = "на экзамен" if share <= MAX_NO_VOLUME_SHARE else "не идёт"
    return {
        "рынок": name,
        "с": data.bars["timestamp"].min().date(),
        "по": data.bars["timestamp"].max().date(),
        "оборот в день, млн ₽": round(daily_turnover(data) / 1e6, 1),
        "сигналов": sizes.height,
        "обычный объём входа": sizes["usual_volume"].median(),
        "контрактов (медиана)": sizes.filter(pl.col("contracts") > 0)[
            "contracts"
        ].median(),
        "без объёма, %": None if share is None else round(share * 100, 2),
        "урезано объёмом, %": round(trimmed / sizes.height * 100, 2)
        if sizes.height
        else None,
        "итог": verdict,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--before",
        type=date.fromisoformat,
        default=EXAM_BEFORE,
        help="Данные только до этой даты, YYYY-MM-DD (по умолчанию 2026-01-01).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    names = [*EXAM_MARKETS, *MARKETS]
    markets = cut(load_markets(names), args.before)
    rows = [market_row(name, data) for name, data in markets.items()]
    print(
        f"\nЛиквидность до {args.before:%d.%m.%Y}. На экзамен — без объёма "
        f"не больше {MAX_NO_VOLUME_SHARE:.0%} сигналов."
    )
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
