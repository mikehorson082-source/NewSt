"""Портфель st01: на всех рынках одновременно открыта максимум одна позиция.

Портфель сам цены не двигает: размер считает `size_position`, сделку
проводит `simulate_position`. Он решает, какой сигнал исполнить, и пишет
в журнал, что стало с остальными.

Правила:
- Пока позиция открыта, новые сигналы не исполняются (`busy`).
- Новый сигнал принимается не раньше свечи, на которой закрылась прошлая
  сделка. Сигнал на самой свече выхода исполняется: сделка закрылась внутри
  свечи или по её закрытию, а сигнал считается по закрытию.
- Сигналы на одной свече разбираются правилом «дешевле по издержкам»:
  берётся сигнал с меньшей долей издержек в риске контракта
  (`PositionSize.cost_share`). При равной доле — порядок рынков в `markets`.
  Если у кандидата 0 контрактов, берётся следующий.
- Размер считается от капитала на момент сигнала. Позиция в это время
  закрыта, поэтому капитал = стартовый + результат закрытых сделок.

Пример: в 10:00 сигналы по Brent и CNY. Издержки CNY — 4% риска, Brent — 9%:
берётся CNY, Brent получает `lost_tie`. CNY закрылась по стопу на свече 13:30.
Сигнал по золоту в 12:00 — `busy`, сигнал по MIX в 13:30 исполняется от
капитала, уменьшенного на убыток CNY.

Новый вход не пересекается с прошлым выходом: сигнал считается по закрытию
свечи выхода, а вход — по открытию следующей.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import polars as pl

from backtest.markets import MARKETS, load_market
from backtest.minutes import RawMinutes
from backtest.position import ClosedTrade, Signal, simulate_position
from backtest.sizing import PositionSize, SizingRules, size_position
from backtest.specs import ContractSpec

TAKEN = "taken"  # исполнен
BUSY = "busy"  # в этот момент была открыта позиция
LOST_TIE = "lost_tie"  # на той же свече исполнен сигнал дешевле по издержкам
ZERO_SIZE = "zero_size"  # 0 контрактов, причина в столбце limit
NO_NEXT_BAR = "no_next_bar"  # сигнал на последней свече ряда: входить не на чем
NO_CAPITAL = "no_capital"  # капитал кончился, торговля остановлена

REQUIRED_SIGNAL_COLUMNS = {"market", "timestamp", "side", "stop", "target"}


@dataclass(frozen=True)
class MarketData:
    """Всё, что портфелю нужно об одном рынке."""

    spec: ContractSpec
    bars: (
        pl.DataFrame
    )  # ряд из load_market: point_value, point_value_known, margin_rate
    minutes: RawMinutes | None = None


@dataclass(frozen=True)
class PortfolioTrade:
    """Исполненная сделка вместе с тем, как выбран её размер."""

    trade: ClosedTrade
    size: PositionSize
    capital_before: float
    # Объём свечи входа в контрактах; None, если в барах нет столбца volume.
    entry_volume: int | None

    @property
    def capital_after(self) -> float:
        return self.capital_before + self.trade.net_pnl

    @property
    def r_multiple(self) -> float:
        """Результат после издержек в долях планового риска (1R)."""
        return self.trade.net_pnl / self.size.risk

    @property
    def mae_r(self) -> float:
        """Насколько сделка уходила против нас, в R: −0.6 — до 60% риска."""
        return self.trade.mae / self.size.risk

    @property
    def mfe_r(self) -> float:
        """Насколько сделка уходила в нашу пользу, в R."""
        return self.trade.mfe / self.size.risk

    @property
    def volume_share(self) -> float | None:
        """Доля сделки в объёме свечи входа.

        11 контрактов при объёме свечи 50 — 22% рынка: проскальзывание
        в 1 шаг для такой сделки слишком оптимистично.
        """
        if not self.entry_volume:
            return None
        return self.size.contracts / self.entry_volume


@dataclass(frozen=True)
class PortfolioResult:
    trades: list[PortfolioTrade]
    # Журнал: каждый входной сигнал со статусом. trade_id — номер сделки в
    # trades: исполненной (taken), мешавшей (busy) или выигравшей выбор (lost_tie).
    signals: pl.DataFrame
    initial_capital: float

    @property
    def final_capital(self) -> float:
        if not self.trades:
            return self.initial_capital
        return self.trades[-1].capital_after

    def trade_log(self) -> pl.DataFrame:
        """Сделки одной таблицей: по строке на сделку."""
        rows = []
        for trade_id, item in enumerate(self.trades):
            trade = item.trade
            rows.append(
                {
                    "trade_id": trade_id,
                    "market": trade.asset,
                    "side": trade.side,
                    "contracts": trade.contracts,
                    "limit": item.size.limit,
                    "signal_time": trade.signal_time,
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "exit_reason": trade.exit_reason,
                    "exit_resolution": trade.exit_resolution,
                    "bars_held": trade.bars_held,
                    "pnl_before_costs": trade.pnl_before_costs,
                    "slippage": trade.slippage,
                    "commissions": trade.commissions,
                    "roll_costs": trade.roll_costs,
                    "costs": trade.costs,
                    "net_pnl": trade.net_pnl,
                    "mae": trade.mae,
                    "mfe": trade.mfe,
                    "mae_r": item.mae_r,
                    "mfe_r": item.mfe_r,
                    "risk": item.size.risk,
                    "r_multiple": item.r_multiple,
                    "cost_share": item.size.cost_share,
                    "entry_volume": item.entry_volume,
                    "usual_volume": item.size.usual_volume,
                    "volume_share": item.volume_share,
                    "capital_before": item.capital_before,
                    "capital_after": item.capital_after,
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None)


@dataclass
class _Candidate:
    row: int  # номер сигнала в журнале
    market: str
    bar: int
    signal: Signal
    size: PositionSize


def load_markets(names: Iterable[str] = MARKETS) -> dict[str, MarketData]:
    """Ряды, спецификации и минутки рынков st01 в порядке `MARKETS`."""
    result = {}
    for name in names:
        params, bars = load_market(name)
        result[name] = MarketData(
            spec=params.contract_spec(),
            bars=bars,
            minutes=RawMinutes.for_market(name),
        )
    return result


def _bar_index(name: str, bars: pl.DataFrame) -> dict[datetime, int]:
    timestamps = bars["timestamp"]
    if not timestamps.is_sorted() or timestamps.n_unique() != bars.height:
        raise ValueError(f"{name}: свечи должны идти по времени без повторов")
    return {timestamp: i for i, timestamp in enumerate(timestamps)}


def _prepare_signals(
    signals: pl.DataFrame, markets: dict[str, MarketData]
) -> pl.DataFrame:
    """Проверяет сигналы и сортирует по времени, внутри свечи — по порядку рынков."""
    missing = REQUIRED_SIGNAL_COLUMNS - set(signals.columns)
    if missing:
        raise ValueError(f"В сигналах нет столбцов: {', '.join(sorted(missing))}")
    if "max_bars" not in signals.columns:
        signals = signals.with_columns(pl.lit(None, dtype=pl.Int64).alias("max_bars"))
    unknown = set(signals["market"].unique()) - set(markets)
    if unknown:
        raise ValueError(f"Нет данных рынков: {', '.join(sorted(unknown))}")
    if signals.select("market", "timestamp").is_duplicated().any():
        raise ValueError("Два сигнала одного рынка на одной свече")
    order = {name: i for i, name in enumerate(markets)}
    return (
        signals.with_columns(
            pl.col("market")
            .replace_strict(order, return_dtype=pl.Int64)
            .alias("_order")
        )
        .sort("timestamp", "_order")
        .drop("_order")
    )


def _volume(bars: pl.DataFrame, bar: int) -> int | None:
    if "volume" not in bars.columns:
        return None
    value = bars["volume"][bar]
    return None if value is None else int(value)


def run_portfolio(
    signals: pl.DataFrame,
    markets: dict[str, MarketData],
    capital: float = 1_000_000.0,
    rules: SizingRules | None = None,
) -> PortfolioResult:
    """Прогоняет сигналы всех рынков через портфель с одной позицией.

    `signals` — таблица стратегии: market, timestamp (свеча, по закрытию
    которой посчитан сигнал), side, stop, target и необязательный max_bars.
    Число контрактов считает портфель. `markets` — данные рынков
    (`load_markets`); их порядок решает ничью по издержкам.
    """
    if not capital > 0:
        raise ValueError("Стартовый капитал должен быть > 0")
    rules = rules or SizingRules()
    log = _prepare_signals(signals, markets)
    indexes = {name: _bar_index(name, data.bars) for name, data in markets.items()}

    n = log.height
    status: list[str | None] = [None] * n
    contracts: list[int | None] = [None] * n
    limits: list[str | None] = [None] * n
    cost_shares: list[float | None] = [None] * n
    trade_ids: list[int | None] = [None] * n

    trades: list[PortfolioTrade] = []
    current = capital
    free_from: datetime | None = None  # свеча, с которой принимаются сигналы

    rows = log.iter_rows(named=True)
    groups: list[list[tuple[int, dict[str, Any]]]] = []
    for i, row in enumerate(rows):
        if groups and groups[-1][0][1]["timestamp"] == row["timestamp"]:
            groups[-1].append((i, row))
        else:
            groups.append([(i, row)])

    for group in groups:
        timestamp = group[0][1]["timestamp"]
        if current <= 0:
            for i, _ in group:
                status[i] = NO_CAPITAL
            continue
        if free_from is not None and timestamp < free_from:
            for i, _ in group:
                status[i] = BUSY
                trade_ids[i] = len(trades) - 1
            continue

        candidates: list[_Candidate] = []
        for i, row in group:
            name = row["market"]
            data = markets[name]
            bar = indexes[name].get(timestamp)
            if bar is None:
                raise ValueError(f"{name}: нет свечи сигнала {timestamp}")
            if bar == data.bars.height - 1:
                status[i] = NO_NEXT_BAR
                continue
            signal = Signal(
                side=row["side"],
                stop=row["stop"],
                target=row["target"],
                max_bars=row["max_bars"],
            )
            size = size_position(data.spec, data.bars, bar, signal, current, rules)
            contracts[i] = size.contracts
            limits[i] = size.limit
            cost_shares[i] = size.cost_share
            candidates.append(_Candidate(i, name, bar, signal, size))

        # sort устойчивый: при равной доле издержек остаётся порядок рынков.
        candidates.sort(key=lambda c: c.size.cost_share)
        chosen: _Candidate | None = None
        for candidate in candidates:
            if candidate.size.skipped:
                status[candidate.row] = ZERO_SIZE
            elif chosen is None:
                chosen = candidate
                status[candidate.row] = TAKEN
            else:
                status[candidate.row] = LOST_TIE
        if chosen is None:
            continue

        data = markets[chosen.market]
        trade = simulate_position(
            data.spec,
            data.bars,
            chosen.bar,
            replace(chosen.signal, contracts=chosen.size.contracts),
            minutes=data.minutes,
        )
        trade_id = len(trades)
        for candidate in candidates:
            if status[candidate.row] in (TAKEN, LOST_TIE):
                trade_ids[candidate.row] = trade_id
        trades.append(
            PortfolioTrade(
                trade=trade,
                size=chosen.size,
                capital_before=current,
                entry_volume=_volume(data.bars, chosen.bar + 1),
            )
        )
        current += trade.net_pnl
        free_from = trade.exit_time

    log = log.with_columns(
        pl.Series("status", status, dtype=pl.String),
        pl.Series("contracts", contracts, dtype=pl.Int64),
        pl.Series("limit", limits, dtype=pl.String),
        pl.Series("cost_share", cost_shares, dtype=pl.Float64),
        pl.Series("trade_id", trade_ids, dtype=pl.Int64),
    )
    return PortfolioResult(trades=trades, signals=log, initial_capital=capital)
