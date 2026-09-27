"""Симуляция одной позиции одного контракта.

Протокол: сигнал считается по закрытию бара `signal_bar`, вход — по открытию
следующего бара с проскальзыванием. Внутри позиции приоритет у стопа, затем
цель, затем выход по времени. Если позиция живёт через перенос контракта,
она переносится по открытию первого бара нового контракта, платит стоимость
переноса, а разрыв между контрактами в P&L не попадает (логика ETF Trick).

Уровни стопа и цели задаются в реальных ценах бара сигнала. Внутри они
хранятся в непрерывном (adjusted) пространстве через `price_adjustment`,
поэтому при переносе не «прыгают». P&L считается по adjusted-ценам: для
рынков с постоянным `point_value` это в точности рубли.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from backtest.costs import commission, fill_price, roll_cost
from backtest.specs import ContractSpec

LONG = 1
SHORT = -1

REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "price_adjustment",
    "is_first_bar_after_roll",
}


@dataclass(frozen=True)
class Signal:
    """Намерение открыть позицию по закрытию текущего бара."""

    side: int
    stop: float
    target: float
    contracts: int = 1
    max_bars: int | None = None

    def __post_init__(self) -> None:
        if self.side not in (LONG, SHORT):
            raise ValueError("Сторона сигнала должна быть +1 или -1")
        if self.stop <= 0 or self.target <= 0:
            raise ValueError("Стоп и цель должны быть > 0")
        if self.contracts < 1:
            raise ValueError("Число контрактов должно быть >= 1")
        if self.max_bars is not None and self.max_bars < 1:
            raise ValueError("Лимит удержания должен быть >= 1 бару")


@dataclass(frozen=True)
class ClosedTrade:
    """Закрытая сделка с разложением результата в рублях."""

    asset: str
    side: int
    contracts: int
    signal_time: datetime
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    bars_held: int
    gross_pnl: float
    commissions: float
    roll_costs: float

    @property
    def net_pnl(self) -> float:
        """Результат после всех издержек."""
        return self.gross_pnl - self.commissions - self.roll_costs


def _validate_bars(bars: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(bars.columns)
    if missing:
        raise ValueError(f"В барах нет столбцов: {', '.join(sorted(missing))}")
    if bars.height < 2:
        raise ValueError("Для симуляции нужно минимум два бара")


def _validate_levels(signal: Signal, close_signal: float) -> None:
    if signal.side == LONG and not signal.stop < close_signal < signal.target:
        raise ValueError("Для лонга требуется stop < close сигнала < target")
    if signal.side == SHORT and not signal.target < close_signal < signal.stop:
        raise ValueError("Для шорта требуется target < close сигнала < stop")


def _intrabar_exit(
    side: int,
    open_adj: float,
    high_adj: float,
    low_adj: float,
    stop_adj: float,
    target_adj: float,
) -> tuple[float | None, str | None]:
    """Выход внутри бара: стоп раньше цели; при гэпе фил по открытию."""
    if side == LONG:
        if low_adj <= stop_adj:
            return min(open_adj, stop_adj), "stop"
        if high_adj >= target_adj:
            return max(open_adj, target_adj), "target"
    else:
        if high_adj >= stop_adj:
            return max(open_adj, stop_adj), "stop"
        if low_adj <= target_adj:
            return min(open_adj, target_adj), "target"
    return None, None


def simulate_position(
    spec: ContractSpec,
    bars: pl.DataFrame,
    signal_bar: int,
    signal: Signal,
    exit_at_end: bool = True,
) -> ClosedTrade:
    """Прогоняет одну позицию по барам начиная с сигнального бара."""
    _validate_bars(bars)
    if not 0 <= signal_bar < bars.height - 1:
        raise ValueError("Сигнал должен иметь следующий бар для входа")

    signal_row = bars.row(signal_bar, named=True)
    close_signal = float(signal_row["close"])
    _validate_levels(signal, close_signal)
    adjustment_signal = float(signal_row["price_adjustment"])
    stop_adj = signal.stop + adjustment_signal
    target_adj = signal.target + adjustment_signal

    side = signal.side
    quantity = signal.contracts
    point_value = spec.point_value

    entry_adj: float | None = None
    entry_time: datetime | None = None
    entry_price: float | None = None
    mark = 0.0
    prev_close = 0.0
    bars_held = 0
    gross_pnl = 0.0
    commissions = 0.0
    roll_costs = 0.0

    for i in range(signal_bar + 1, bars.height):
        row = bars.row(i, named=True)
        adjustment = float(row["price_adjustment"])
        open_adj = float(row["open"]) + adjustment
        high_adj = float(row["high"]) + adjustment
        low_adj = float(row["low"]) + adjustment
        close_adj = float(row["close"]) + adjustment

        if entry_adj is None:
            entry_adj = fill_price(spec, open_adj, side, quantity)
            entry_time = row["timestamp"]
            entry_price = entry_adj - adjustment
            commissions += commission(spec, entry_price, quantity)
            mark = entry_adj
            bars_held = 1
        else:
            bars_held += 1
            if bool(row["is_first_bar_after_roll"]):
                roll_costs += roll_cost(spec, prev_close, float(row["open"]), quantity)
                mark = open_adj

        exit_adj, reason = _intrabar_exit(
            side, open_adj, high_adj, low_adj, stop_adj, target_adj
        )
        if (
            exit_adj is None
            and signal.max_bars is not None
            and bars_held >= signal.max_bars
        ):
            exit_adj, reason = close_adj, "time"
        if exit_adj is None and exit_at_end and i == bars.height - 1:
            exit_adj, reason = close_adj, "end"

        if exit_adj is not None:
            exit_fill_adj = fill_price(spec, exit_adj, -side, quantity)
            gross_pnl += side * (exit_fill_adj - mark) * point_value * quantity
            exit_price = exit_fill_adj - adjustment
            commissions += commission(spec, exit_price, quantity)
            return ClosedTrade(
                asset=spec.asset,
                side=side,
                contracts=quantity,
                signal_time=signal_row["timestamp"],
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=row["timestamp"],
                exit_price=exit_price,
                exit_reason=reason,
                bars_held=bars_held,
                gross_pnl=gross_pnl,
                commissions=commissions,
                roll_costs=roll_costs,
            )

        gross_pnl += side * (close_adj - mark) * point_value * quantity
        mark = close_adj
        prev_close = float(row["close"])

    raise ValueError("Позиция осталась открытой к концу данных")
