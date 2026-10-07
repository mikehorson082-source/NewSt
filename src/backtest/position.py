"""Симуляция одной позиции одного контракта.

Протокол: сигнал считается по закрытию бара `signal_bar`, вход — по открытию
следующего бара с проскальзыванием. Внутри бара сначала проверяется гэп
открытия за уровнем, затем касание стопа или цели, затем выход по времени.
Если бар задел и стоп, и цель, порядок касаний берётся из минуток того же
контракта (`RawMinutes`). Без минуток или при касании обоих уровней в одной
минутке выход пессимистично по стопу.

Если позиция живёт через перенос контракта, она переносится по открытию
первого бара нового контракта и платит стоимость переноса. Разрыв между
контрактами (базис) в P&L не попадает — его убирает поправка
`price_adjustment`, а движение рынка за ночь до открытия нового контракта
попадает, как у реальной позиции (логика ETF Trick).

Уровни стопа и цели задаются в реальных ценах бара сигнала. Внутри они
хранятся в непрерывном (adjusted) пространстве через `price_adjustment`,
поэтому при переносе не «прыгают». Заявки исполняются в реальных ценах:
аддитивная поправка может увести adjusted-цены ниже нуля.

P&L начисляется по барам, как биржа начисляет вариационную маржу:
изменение цены за бар × рубли за пункт этого бара (столбец `point_value`,
`markets.add_point_value`). У Brent и золота рубли за пункт идут за курсом
клиринга USD/RUB: лонг, +$1 при курсе 80 и ещё +$1 при курсе 90 даёт
1 × 10 × 80 + 1 × 10 × 90 = 1 700 ₽, а не 2 × 10 × 90 = 1 800 ₽. Изменения
берутся по adjusted-ценам: внутри контракта они равны изменениям реальных
цен. Комиссии и стоимость переноса тоже по рублям за пункт своего бара.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from backtest.costs import commission, fill_price, roll_cost
from backtest.minutes import RawMinutes
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
    "point_value",
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
    # Как найден выход: "bar" — хватило 15-минутного бара; "minutes" — бар
    # задел и стоп, и цель, порядок взят из минуток. Порядок неизвестен,
    # выход по стопу: "same_minute" — оба уровня в одной минутке,
    # "no_minutes" — минутки не переданы, "minutes_mismatch" — минутки не
    # совпали с баром.
    exit_resolution: str
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


def _point_value(row: dict[str, Any]) -> float:
    """Рубли за пункт бара: без них прибыль бара не посчитать."""
    value = row["point_value"]
    if value is None or not value > 0:
        raise ValueError(f"Нет рублей за пункт point_value на баре {row['timestamp']}")
    return float(value)


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
    """Выход внутри бара: гэп проверяется первым, затем касание уровня.

    Бар, открывшийся между уровнями и задевший оба, симулятор сюда не передаёт:
    порядок касаний в нём разбирается по минуткам (`_exit_by_minutes`).
    """
    if side == LONG:
        if open_adj <= stop_adj:
            return open_adj, "stop"
        if open_adj >= target_adj:
            return open_adj, "target"

        if low_adj <= stop_adj:
            return stop_adj, "stop"
        if high_adj >= target_adj:
            return target_adj, "target"
    else:
        if open_adj >= stop_adj:
            return open_adj, "stop"
        if open_adj <= target_adj:
            return open_adj, "target"

        if high_adj >= stop_adj:
            return stop_adj, "stop"
        if low_adj <= target_adj:
            return target_adj, "target"
    return None, None


def _hits_both_levels(
    side: int,
    open_adj: float,
    high_adj: float,
    low_adj: float,
    stop_adj: float,
    target_adj: float,
) -> bool:
    """Бар открылся между уровнями и задел оба: по нему порядок не узнать."""
    if side == LONG:
        return (
            stop_adj < open_adj < target_adj
            and low_adj <= stop_adj
            and high_adj >= target_adj
        )
    return (
        target_adj < open_adj < stop_adj
        and high_adj >= stop_adj
        and low_adj <= target_adj
    )


def _minutes_match_bar(window: pl.DataFrame, bar: dict[str, Any]) -> bool:
    """Минутки описывают именно этот бар: те же open, high, low и close."""
    if window.is_empty():
        return False
    return (
        window["open"][0] == bar["open"]
        and window["high"].max() == bar["high"]
        and window["low"].min() == bar["low"]
        and window["close"][-1] == bar["close"]
    )


def _exit_by_minutes(
    minutes: RawMinutes | None,
    bar: dict[str, Any],
    side: int,
    stop_adj: float,
    target_adj: float,
) -> tuple[float, str, str]:
    """Бар задел и стоп, и цель: порядок касаний берётся из минуток бара.

    Минутки проходятся тем же правилом, что и бары: гэп открытия минутки за
    уровнем — выход по её открытию, иначе касание — выход по уровню. Если
    порядок узнать нельзя, выход пессимистично по стопу.
    """
    if minutes is None:
        return stop_adj, "stop", "no_minutes"
    window = minutes.bar_minutes(bar["contract"], bar["timestamp"])
    if not _minutes_match_bar(window, bar):
        return stop_adj, "stop", "minutes_mismatch"

    adjustment = float(bar["price_adjustment"])
    for open_, high, low in window.select("open", "high", "low").iter_rows():
        open_adj = open_ + adjustment
        high_adj = high + adjustment
        low_adj = low + adjustment
        if _hits_both_levels(side, open_adj, high_adj, low_adj, stop_adj, target_adj):
            return stop_adj, "stop", "same_minute"
        exit_adj, reason = _intrabar_exit(
            side, open_adj, high_adj, low_adj, stop_adj, target_adj
        )
        if exit_adj is not None and reason is not None:
            return exit_adj, reason, "minutes"
    raise AssertionError("Минутки совпали с баром, но не задели ни одного уровня")


def simulate_position(
    spec: ContractSpec,
    bars: pl.DataFrame,
    signal_bar: int,
    signal: Signal,
    exit_at_end: bool = True,
    minutes: RawMinutes | None = None,
) -> ClosedTrade:
    """Прогоняет одну позицию по барам начиная с сигнального бара.

    В барах нужны рубли за пункт `point_value` — ряд из `markets.load_market`.
    `minutes` — минутки рынка для баров, задевших и стоп, и цель. Без них
    такой бар закрывается по стопу.
    """
    _validate_bars(bars)
    if minutes is not None and "contract" not in bars.columns:
        raise ValueError("Для разбора по минуткам в барах нужен столбец contract")
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
        point_value = _point_value(row)
        adjustment = float(row["price_adjustment"])
        open_adj = float(row["open"]) + adjustment
        high_adj = float(row["high"]) + adjustment
        low_adj = float(row["low"]) + adjustment
        close_adj = float(row["close"]) + adjustment

        if entry_adj is None:
            entry_price = fill_price(spec, float(row["open"]), side, quantity)
            entry_adj = entry_price + adjustment
            entry_time = row["timestamp"]
            commissions += commission(spec, entry_price, point_value, quantity)
            mark = entry_adj
            bars_held = 1
        else:
            bars_held += 1
            if bool(row["is_first_bar_after_roll"]):
                # mark не сбрасывается на open: базис уже убран поправкой,
                # а движение рынка за ночь — настоящий P&L позиции.
                roll_costs += roll_cost(
                    spec, prev_close, float(row["open"]), point_value, quantity
                )

        resolution = "bar"
        if _hits_both_levels(side, open_adj, high_adj, low_adj, stop_adj, target_adj):
            exit_adj, reason, resolution = _exit_by_minutes(
                minutes, row, side, stop_adj, target_adj
            )
        else:
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
            assert entry_time is not None
            assert entry_price is not None
            assert reason is not None
            exit_price = fill_price(spec, exit_adj - adjustment, -side, quantity)
            exit_fill_adj = exit_price + adjustment
            gross_pnl += side * (exit_fill_adj - mark) * point_value * quantity
            commissions += commission(spec, exit_price, point_value, quantity)
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
                exit_resolution=resolution,
                bars_held=bars_held,
                gross_pnl=gross_pnl,
                commissions=commissions,
                roll_costs=roll_costs,
            )

        gross_pnl += side * (close_adj - mark) * point_value * quantity
        mark = close_adj
        prev_close = float(row["close"])

    raise ValueError("Позиция осталась открытой к концу данных")
