"""Размер позиции на реальных данных: сколько контрактов и какой лимит сработал.

Капитал 1 000 000 ₽, риск 1% (10 000 ₽), ГО позиции не больше 50% капитала.
Для каждого рынка — лонг по закрытию первой и последней свечи ряда со
стопом на 0.25%, 0.5%, 1% и 2% ниже закрытия. Близкий стоп даёт много
контрактов, и тогда ограничивает уже ГО, а не риск. Ничего не записывает.
Запуск: uv run python st01/check_sizing.py
"""

from __future__ import annotations

import polars as pl

from backtest.markets import MARKETS, load_market
from backtest.position import LONG, Signal
from backtest.sizing import size_position

CAPITAL = 1_000_000.0
STOP_DISTANCES = (0.0025, 0.005, 0.01, 0.02)
LIMIT_NAMES = {"risk": "риск", "margin": "ГО"}


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_thousands_separator(" ")

    rows = []
    for name in MARKETS:
        params, bars = load_market(name)
        spec = params.contract_spec()
        for signal_bar in (0, bars.height - 1):
            bar = bars.row(signal_bar, named=True)
            close = bar["close"]
            for distance in STOP_DISTANCES:
                signal = Signal(
                    side=LONG, stop=close * (1 - distance), target=close * 2
                )
                size = size_position(spec, bars, signal_bar, signal, CAPITAL)
                rows.append(
                    {
                        "рынок": name,
                        "дата": bar["timestamp"].date(),
                        "₽ за пункт": round(bar["point_value_known"], 2),
                        "ГО контракта ₽": round(size.margin_per_contract),
                        "стоп %": distance * 100,
                        "риск контракта ₽": round(size.risk_per_contract),
                        "по риску": size.contracts_by_risk,
                        "по ГО": size.contracts_by_margin,
                        "контрактов": size.contracts,
                        "лимит": LIMIT_NAMES[size.limit],
                        "риск ₽": round(size.risk),
                        "ГО % капитала": round(size.margin / CAPITAL * 100, 1),
                    }
                )
    capital = f"{CAPITAL:,.0f}".replace(",", " ")
    print(
        f"Капитал {capital} ₽, риск 1%, ГО не больше 50%. "
        "Лонг по закрытию свечи, стоп ниже закрытия."
    )
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
