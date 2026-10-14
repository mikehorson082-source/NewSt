"""Симуляция одной позиции одного контракта.

Протокол: сигнал считается по закрытию бара `signal_bar`, вход — по открытию
следующего бара с проскальзыванием. Внутри бара сначала проверяется гэп
открытия за уровнем, затем касание стопа или цели, затем выход по времени
(`max_bars`) и по расписанию (`hold`).
Если бар задел и стоп, и цель, порядок касаний берётся из минуток того же
контракта (`RawMinutes`). Без минуток или при касании обоих уровней в одной
минутке выход пессимистично по стопу. Если бар задел один уровень, цену
исполнения тоже уточняют минутки: минутка, открывшаяся за уровнем, исполняет
по своему открытию. Пример: стоп 99.00, бар открылся на 99.50, а минутка
10:07 — сразу на 98.90: выход по 98.90, а не по 99.00. Без минуток — по уровню.

Если позиция живёт через перенос контракта, она переносится по открытию
первого бара нового контракта и платит стоимость переноса. Разрыв между
контрактами (базис) в P&L не попадает — его убирает поправка
`price_adjustment`, а движение рынка за ночь до открытия нового контракта
попадает, как у реальной позиции (логика ETF Trick).

Выход по расписанию (`Signal.hold`): "day" — позиция живёт только внутри
дня и закрывается по закрытию его последнего бара; "overnight" — обычные
ночи держит, но закрывается по закрытию последнего бара перед выходными
или праздником. Какой бар последний, говорят отметки `is_day_end` и
`is_before_break` (`markets.add_session_marks`). Пример: шорт по золоту,
вход в пятницу в 21:00, hold="overnight" — выход в пятницу по закрытию
бара 23:45, гэп понедельника в 08:45 позицию уже не застанет. Стоп и цель
внутри последнего бара срабатывают раньше, чем выход по расписанию.

Подвижный стоп. Цели может не быть (`target=None`): выход только по стопу,
времени или расписанию. Стоп подтягивается двумя способами:
- безубыток (`breakeven_r`): цена коснулась входа + `breakeven_r` × расстояние
  до начального стопа — стоп переносится на цену входа (открытие бара входа,
  без проскальзывания: так выход не зависит от издержек);
- скользящий (`trail`, в пунктах): стоп = лучшее закрытие сделки − `trail`
  для лонга, + `trail` для шорта.
Порядок внутри бара неизвестен: новый максимум мог быть до отката или после.
Поэтому внутри бара стоп не двигается — новое значение считается по бару
целиком и действует со следующего бара. Стоп только подтягивается, назад не
отходит. Пример: лонг от 100, стоп 99, безубыток после +1R. Бар: максимум
101.2, минимум 99.3 — в нём стоп ещё 99, со следующего — 100. Выход по
подвинутому стопу — причина `trail` (если задан `trail`) или `breakeven`.

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

Для отчёта сделка отдаёт путь — результат после издержек на закрытии
каждого бара (`path`, из него кривая капитала и просадка внутри сделки),
MAE и MFE — худший и лучший незакрытый результат, и проскальзывание
отдельной строкой: результат по ценам рынка − проскальзывание − сборы −
переносы = net_pnl, каждая издержка ровно один раз.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from backtest.costs import commission, fill_price, impact_points, roll_cost
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
# Нужны, только если включено влияние на рынок (`ContractSpec.impact`).
IMPACT_COLUMNS = {"volume", "bar_sigma"}

# Сколько позиция может жить по расписанию (`Signal.hold`).
HOLD_ANY = "any"  # без ограничения: ночи, выходные и праздники
HOLD_OVERNIGHT = "overnight"  # обычные ночи; перед выходными и праздниками выход
HOLD_DAY = "day"  # только внутри дня
# Отметка бара, по закрытию которого позиция закрывается, и причина выхода.
HOLD_EXITS = {
    HOLD_OVERNIGHT: ("is_before_break", "break"),
    HOLD_DAY: ("is_day_end", "day_end"),
}

# Выход по подвинутому стопу (`Signal.breakeven_r`, `Signal.trail`).
BREAKEVEN = "breakeven"
TRAIL = "trail"
# Выходы по уровню внутри бара: что было в баре после них, неизвестно.
LEVEL_EXITS = ("stop", "target", BREAKEVEN, TRAIL)


@dataclass(frozen=True)
class Signal:
    """Намерение открыть позицию по закрытию текущего бара."""

    side: int
    stop: float
    target: float | None  # None — без цели
    contracts: int = 1
    max_bars: int | None = None
    hold: str = HOLD_ANY
    # Безубыток: после касания +breakeven_r × (расстояние до начального стопа).
    breakeven_r: float | None = None
    # Скользящий стоп: расстояние от лучшего закрытия сделки, в пунктах цены.
    trail: float | None = None

    def __post_init__(self) -> None:
        if self.side not in (LONG, SHORT):
            raise ValueError("Сторона сигнала должна быть +1 или -1")
        if self.stop <= 0 or (self.target is not None and self.target <= 0):
            raise ValueError("Стоп и цель должны быть > 0")
        if self.breakeven_r is not None and not self.breakeven_r > 0:
            raise ValueError("Порог безубытка breakeven_r должен быть > 0")
        if self.trail is not None and not self.trail > 0:
            raise ValueError("Расстояние скользящего стопа trail должно быть > 0")
        if self.contracts < 1:
            raise ValueError("Число контрактов должно быть >= 1")
        if self.max_bars is not None and self.max_bars < 1:
            raise ValueError("Лимит удержания должен быть >= 1 бару")
        if self.hold != HOLD_ANY and self.hold not in HOLD_EXITS:
            raise ValueError(
                f"hold должен быть {HOLD_ANY}, {HOLD_OVERNIGHT} или {HOLD_DAY}"
            )


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
    # Как найден выход: "bar" — хватило 15-минутного бара; "minutes" — цена
    # или порядок касаний взяты из минуток бара. Порядок неизвестен,
    # выход по стопу: "same_minute" — оба уровня в одной минутке,
    # "no_minutes" — минутки не переданы, "minutes_mismatch" — минутки не
    # совпали с баром.
    exit_resolution: str
    bars_held: int
    # По ценам исполнения: проскальзывание входа и выхода уже внутри.
    gross_pnl: float
    commissions: float
    roll_costs: float  # проскальзывание и сборы переносов
    # Проскальзывание входа и выхода в рублях, отдельной строкой.
    slippage: float
    # Худший и лучший незакрытый результат от цены исполнения входа, без
    # сборов и переносов: MAE <= 0 <= MFE. По high/low баров; в баре выхода по стопу
    # или цели — только открытие и уровень выхода: что было после выхода,
    # неизвестно.
    mae: float
    mfe: float
    # Результат после издержек на закрытии каждого бара сделки (в баре
    # выхода — на выходе): (метка бара, рубли). Последняя точка = net_pnl.
    path: tuple[tuple[datetime, float], ...]
    # Влияние на рынок входа и выхода в рублях (`ContractSpec.impact`), сверх
    # обычного проскальзывания. Переносы его не платят: они редки и идут в
    # ликвидное время.
    impact: float = 0.0

    @property
    def net_pnl(self) -> float:
        """Результат после всех издержек."""
        return self.gross_pnl - self.commissions - self.roll_costs

    @property
    def pnl_before_costs(self) -> float:
        """Результат по ценам рынка, до проскальзывания, сборов и переносов."""
        return self.gross_pnl + self.slippage + self.impact

    @property
    def costs(self) -> float:
        """Все издержки сделки: каждая учтена ровно один раз."""
        return self.slippage + self.impact + self.commissions + self.roll_costs


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
    # Без цели проверяется только стоп: цель «бесконечно далеко».
    target = signal.target
    if target is None:
        target = signal.side * math.inf
    if signal.side == LONG and not signal.stop < close_signal < target:
        raise ValueError("Для лонга требуется stop < close сигнала < target")
    if signal.side == SHORT and not target < close_signal < signal.stop:
        raise ValueError("Для шорта требуется target < close сигнала < stop")


def _tighter(side: int, stop: float, candidate: float) -> float:
    """Стоп только подтягивается: у лонга вверх, у шорта вниз."""
    return max(stop, candidate) if side == LONG else min(stop, candidate)


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


def _opens_beyond_level(
    side: int, open_adj: float, stop_adj: float, target_adj: float
) -> bool:
    """Бар открылся за стопом или за целью: выход по открытию, минутки не нужны."""
    if side == LONG:
        return open_adj <= stop_adj or open_adj >= target_adj
    return open_adj >= stop_adj or open_adj <= target_adj


def _level_by_minutes(
    minutes: RawMinutes,
    bar: dict[str, Any],
    side: int,
    stop_adj: float,
    target_adj: float,
    level: tuple[float, str],
) -> tuple[float, str, str]:
    """Бар задел один уровень: цена исполнения — по минутке, которая его задела.

    Минутка открылась за уровнем — выход по её открытию, иначе по уровню.
    Минутки не совпали с баром — выход по уровню, как по одному бару.
    """
    window = minutes.bar_minutes(bar["contract"], bar["timestamp"])
    if not _minutes_match_bar(window, bar):
        return level[0], level[1], "bar"
    return _walk_minutes(window, bar, side, stop_adj, target_adj)


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
    return _walk_minutes(window, bar, side, stop_adj, target_adj)


def _walk_minutes(
    window: pl.DataFrame,
    bar: dict[str, Any],
    side: int,
    stop_adj: float,
    target_adj: float,
) -> tuple[float, str, str]:
    """Первая минутка бара, задевшая уровень, и цена выхода по ней."""
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


def _impact(
    spec: ContractSpec, row: dict[str, Any], price: float, quantity: int
) -> float:
    """Влияние заявки на цену в пунктах по объёму и σ свечи исполнения."""
    if spec.impact == 0:
        return 0.0
    return impact_points(spec, price, quantity, row["volume"], row["bar_sigma"])


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
    такой бар закрывается по стопу. Для `signal.hold` "day" и "overnight"
    нужны отметки расписания `is_day_end` и `is_before_break`.
    """
    _validate_bars(bars)
    missing_impact = IMPACT_COLUMNS - set(bars.columns)
    if spec.impact > 0 and missing_impact:
        raise ValueError(
            "Для влияния на рынок в барах нужны столбцы: "
            + ", ".join(sorted(missing_impact))
        )
    if minutes is not None and "contract" not in bars.columns:
        raise ValueError("Для разбора по минуткам в барах нужен столбец contract")
    hold_column, hold_reason = HOLD_EXITS.get(signal.hold, (None, None))
    if hold_column is not None and hold_column not in bars.columns:
        raise ValueError(
            f"Для выхода по расписанию в барах нужен столбец {hold_column}"
        )
    if not 0 <= signal_bar < bars.height - 1:
        raise ValueError("Сигнал должен иметь следующий бар для входа")

    signal_row = bars.row(signal_bar, named=True)
    close_signal = float(signal_row["close"])
    _validate_levels(signal, close_signal)
    adjustment_signal = float(signal_row["price_adjustment"])
    side = signal.side
    quantity = signal.contracts
    stop_adj = signal.stop + adjustment_signal
    target_adj = (
        side * math.inf if signal.target is None else signal.target + adjustment_signal
    )
    # Расстояние до начального стопа — 1R в ценах, для порога безубытка.
    distance = side * (close_signal - signal.stop)
    stop_moved = False
    best_close_adj: float | None = None
    entry_open_adj = 0.0

    entry_adj: float | None = None
    entry_time: datetime | None = None
    entry_price: float | None = None
    mark = 0.0
    prev_close = 0.0
    bars_held = 0
    gross_pnl = 0.0
    commissions = 0.0
    roll_costs = 0.0
    slippage = 0.0
    impact = 0.0
    mae = 0.0
    mfe = 0.0
    path: list[tuple[datetime, float]] = []

    for i in range(signal_bar + 1, bars.height):
        row = bars.row(i, named=True)
        point_value = _point_value(row)
        adjustment = float(row["price_adjustment"])
        open_adj = float(row["open"]) + adjustment
        high_adj = float(row["high"]) + adjustment
        low_adj = float(row["low"]) + adjustment
        close_adj = float(row["close"]) + adjustment

        if entry_adj is None:
            entry_impact = _impact(spec, row, float(row["open"]), quantity)
            entry_price = fill_price(
                spec, float(row["open"]), side, quantity, entry_impact
            )
            entry_adj = entry_price + adjustment
            entry_open_adj = open_adj
            entry_time = row["timestamp"]
            commissions += commission(spec, entry_price, point_value, quantity)
            slippage += spec.slippage * point_value * quantity
            impact += entry_impact * point_value * quantity
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
                exit_adj is not None
                and reason is not None
                and minutes is not None
                and not _opens_beyond_level(side, open_adj, stop_adj, target_adj)
            ):
                exit_adj, reason, resolution = _level_by_minutes(
                    minutes, row, side, stop_adj, target_adj, (exit_adj, reason)
                )
        if (
            exit_adj is None
            and signal.max_bars is not None
            and bars_held >= signal.max_bars
        ):
            exit_adj, reason = close_adj, "time"
        if exit_adj is None and hold_column is not None and row[hold_column]:
            exit_adj, reason = close_adj, hold_reason
        if exit_adj is None and exit_at_end and i == bars.height - 1:
            exit_adj, reason = close_adj, "end"
        if reason == "stop" and stop_moved:
            reason = TRAIL if signal.trail is not None else BREAKEVEN

        # Цены бара, через которые позиция точно прошла. Выход по стопу или
        # цели внутри бара: high/low могли случиться уже после выхода.
        if exit_adj is not None and reason in LEVEL_EXITS:
            seen = (open_adj, exit_adj)
        else:
            seen = (low_adj, high_adj)
        for price in seen:
            excursion = gross_pnl + side * (price - mark) * point_value * quantity
            mae = min(mae, excursion)
            mfe = max(mfe, excursion)

        if exit_adj is not None:
            assert entry_time is not None
            assert entry_price is not None
            assert reason is not None
            exit_impact = _impact(spec, row, exit_adj - adjustment, quantity)
            exit_price = fill_price(
                spec, exit_adj - adjustment, -side, quantity, exit_impact
            )
            exit_fill_adj = exit_price + adjustment
            gross_pnl += side * (exit_fill_adj - mark) * point_value * quantity
            commissions += commission(spec, exit_price, point_value, quantity)
            slippage += spec.slippage * point_value * quantity
            impact += exit_impact * point_value * quantity
            path.append((row["timestamp"], gross_pnl - commissions - roll_costs))
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
                slippage=slippage,
                mae=mae,
                mfe=mfe,
                path=tuple(path),
                impact=impact,
            )

        gross_pnl += side * (close_adj - mark) * point_value * quantity
        path.append((row["timestamp"], gross_pnl - commissions - roll_costs))
        mark = close_adj
        prev_close = float(row["close"])

        # Стоп для следующего бара — по этому бару целиком.
        new_stop = stop_adj
        if signal.breakeven_r is not None:
            extreme = high_adj if side == LONG else low_adj
            if side * (extreme - entry_open_adj) >= signal.breakeven_r * distance:
                new_stop = _tighter(side, new_stop, entry_open_adj)
        if signal.trail is not None:
            best_close_adj = (
                close_adj
                if best_close_adj is None
                else _tighter(side, best_close_adj, close_adj)
            )
            new_stop = _tighter(side, new_stop, best_close_adj - side * signal.trail)
        if new_stop != stop_adj:
            stop_adj = new_stop
            stop_moved = True

    raise ValueError("Позиция осталась открытой к концу данных")
