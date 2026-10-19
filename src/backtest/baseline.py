"""«Купил и держи»: сверка симулятора с NAV склейки и стоимость переноса.

NAV склейки (`build_continuous_futures.add_etf_nav`) — основа признаков и
таргетов, симулятор — то, что платит рубли. Если они расходятся, модель
учится на одном, а бэктест платит по другому. Поэтому один контракт держится
с первой свечи до последней без издержек, и на каждой свече сравнивается:

    прибыль симулятора за свечу ÷ (контракты × рубли за пункт × база)
    = доходность NAV за свечу,

где база — закрытие прошлой свечи в ценах текущего контракта
(close_adjusted[t−1] − price_adjustment[t]). Внутри контракта это прошлое
закрытие, на первой свече нового контракта — закрытие старого, пересчитанное
поправкой: так NAV не видит базиса, а ночное движение рынка оставляет.
Первая свеча сделки не сравнивается: симулятор входит по её открытию, а NAV
считает от прошлого закрытия.

NAV склейки собран без издержки переноса (`--roll-cost-bps 0`). Симулятор
знает её из спецификации: `roll_costs_bps` — стоимость каждого переноса в
б.п. стоимости контракта, её можно подставить в сборку NAV.

Базовая линия для стратегий (`buy_and_hold_result`): на весь капитал
покупается столько контрактов, чтобы их стоимость равнялась капиталу
(плечо 1×), и держится до конца — через тот же симулятор, с теми же
издержками переноса (ML4T 16 `09`: базовая линия проходит те же условия).
Результат — `PortfolioResult`, к нему подходят отчёт и графики. Стопа нет,
поэтому 1R — весь номинал позиции на входе: MAE −0.55R — позиция уходила
на 55% номинала против нас.
"""

from __future__ import annotations

import math
from dataclasses import replace

import polars as pl

from backtest.costs import commission, roll_cost
from backtest.minutes import RawMinutes
from backtest.portfolio import PortfolioResult, PortfolioTrade
from backtest.position import LONG, ClosedTrade, Signal, simulate_position
from backtest.sizing import PositionSize
from backtest.specs import ContractSpec

NOTIONAL = "notional"  # размер «купил и держи» задан номиналом, а не риском

DAYS_PER_YEAR = 365.25


def free_spec(spec: ContractSpec) -> ContractSpec:
    """Тот же контракт без проскальзывания и сборов."""
    return replace(spec, exchange_fee_rate=0.0, broker_fee_rate=0.0, slippage_ticks=0.0)


def buy_and_hold(
    spec: ContractSpec,
    bars: pl.DataFrame,
    contracts: int = 1,
    minutes: RawMinutes | None = None,
) -> ClosedTrade:
    """Лонг с закрытия первой свечи до конца ряда, без стопа и цели.

    Выход — концом данных; любой другой выход — ошибка, а не результат.
    Стопа нет совсем, а не «стоп у нуля»: у газа аддитивная поправка уводит
    adjusted-цены ниже нуля (−2.3 к октябрю 2025), и такой стоп срабатывал.
    """
    signal = Signal(side=LONG, stop=None, target=None, contracts=contracts)
    trade = simulate_position(spec, bars, 0, signal, minutes=minutes)
    if trade.exit_reason != "end":
        raise AssertionError(f"«Купил и держи» закрылся по {trade.exit_reason}")
    return trade


def buy_and_hold_result(
    spec: ContractSpec, bars: pl.DataFrame, capital: float = 1_000_000.0
) -> PortfolioResult:
    """«Купил и держи» на весь капитал с плечом 1× как результат портфеля.

    Контрактов — сколько помещается в капитал по стоимости на закрытии
    первой свечи (рубли за пункт — известные в тот момент). Держатся до
    конца ряда, переносы платят проскальзывание и сборы.
    """
    if not capital > 0:
        raise ValueError("Капитал должен быть > 0")
    first = bars.sort("timestamp").row(0, named=True)
    close = float(first["close"])
    point_value = float(first.get("point_value_known") or first["point_value"])
    notional = close * point_value
    contracts = math.floor(capital / notional)
    if contracts < 1:
        raise ValueError("Капитала не хватает на один контракт")
    trade = buy_and_hold(spec, bars, contracts=contracts)
    size = PositionSize(
        contracts=contracts,
        limit=NOTIONAL,
        capital=capital,
        # 1R — весь номинал контракта: стопа нет.
        risk_per_contract=notional,
        margin_per_contract=close
        * point_value
        * float(first.get("margin_rate") or 0.0),
        contracts_by_risk=contracts,
        contracts_by_margin=contracts,
        costs_per_contract=2 * spec.slippage * point_value
        + 2 * commission(spec, close, point_value),
    )
    # Объём свечи входа — следующей после сигнальной.
    volume = bars.sort("timestamp").row(1, named=True).get("volume")
    return PortfolioResult(
        trades=[
            PortfolioTrade(
                trade=trade,
                size=size,
                capital_before=capital,
                entry_volume=None if volume is None else int(volume),
            )
        ],
        signals=pl.DataFrame(),
        initial_capital=capital,
    )


def nav_comparison(trade: ClosedTrade, bars: pl.DataFrame) -> pl.DataFrame:
    """Доходность каждой свечи сделки: по симулятору и по NAV.

    Сделка — без издержек (`free_spec`): иначе в прибыли свечи переноса
    сидит стоимость переноса, которой нет в NAV.
    """
    ordered = bars.sort("timestamp").with_columns(
        (
            pl.col("close").shift(1)
            + pl.col("price_adjustment").shift(1)
            - pl.col("price_adjustment")
        ).alias("_base"),
        (pl.col("nav") / pl.col("nav").shift(1) - 1.0).alias("nav_return"),
    )
    path = pl.DataFrame(
        {
            "timestamp": [ts for ts, _ in trade.path],
            "_pnl": [value for _, value in trade.path],
        }
    ).with_columns(pl.col("timestamp").cast(ordered["timestamp"].dtype))
    joined = path.join(ordered, on="timestamp", how="left").sort("timestamp")
    return (
        joined.with_columns(
            (
                pl.col("_pnl").diff()
                / (trade.contracts * pl.col("point_value") * pl.col("_base"))
            ).alias("sim_return")
        )
        .slice(1)
        .select(
            "timestamp",
            "is_first_bar_after_roll",
            "sim_return",
            "nav_return",
            (pl.col("sim_return") - pl.col("nav_return")).abs().alias("difference"),
        )
    )


def roll_costs_bps(spec: ContractSpec, bars: pl.DataFrame) -> pl.DataFrame:
    """Стоимость каждого переноса ряда в б.п. стоимости нового контракта.

    Перенос — закрытие старого контракта по прошлому закрытию и открытие
    нового по открытию свечи: проскальзывание двух ног и два сбора (`roll_cost`).
    """
    ordered = bars.sort("timestamp").with_columns(
        pl.col("close").shift(1).alias("_old_close")
    )
    rolls = ordered.filter(pl.col("is_first_bar_after_roll"))
    rows = []
    for row in rolls.iter_rows(named=True):
        cost = roll_cost(spec, row["_old_close"], row["open"], row["point_value"])
        rows.append(
            {
                "timestamp": row["timestamp"],
                "contract": row.get("contract"),
                "cost_rub": cost,
                "cost_bps": cost / (row["open"] * row["point_value"]) * 10_000,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "timestamp": ordered["timestamp"].dtype,
            "contract": pl.String,
            "cost_rub": pl.Float64,
            "cost_bps": pl.Float64,
        },
    )


def nav_summary(bars: pl.DataFrame) -> dict[str, float]:
    """Что дало бы просто держать рынок: рост NAV, CAGR и худшая просадка."""
    ordered = bars.sort("timestamp")
    nav = ordered["nav"]
    days = (ordered["timestamp"][-1] - ordered["timestamp"][0]).days + 1
    growth = float(nav[-1] / nav[0])
    drawdown = (nav / nav.cum_max() - 1.0).min()
    return {
        "total_return": growth - 1.0,
        "cagr": growth ** (DAYS_PER_YEAR / days) - 1.0,
        "max_drawdown": float(drawdown),
    }
