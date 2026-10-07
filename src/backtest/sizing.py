"""Размер позиции в момент сигнала: число контрактов от риска и предел ГО.

Риск контракта — сколько рублей он потеряет, если сработает стоп:
расстояние от закрытия сигнальной свечи до стопа плюс проскальзывание
входа и выхода, × рубли за пункт, плюс комиссии входа и выхода.
Рубли за пункт — только известные на закрытии сигнальной свечи
(`point_value_known`): по какому курсу биржа рассчитает следующие свечи,
в момент сигнала ещё не известно.

Пример: Brent, закрытие 70, стоп 69, курс 85 → 850 ₽ за пункт.
Риск контракта = (1 + 2 × 0.01) × 850 + (70 + 69) × 850 × 0.0132% ≈ 883 ₽.
Бюджет риска 10 000 ₽ → 11 контрактов. ГО 11 × 70 × 850 × 25% ≈ 164 000 ₽ —
16% капитала, меньше предела 50%: сработал лимит риска.

В риск не входят: вход по открытию следующей свечи вместо закрытия
сигнальной, выход хуже стопа при гэпе и стоимость переноса контракта.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from backtest.costs import commission
from backtest.position import Signal
from backtest.specs import ContractSpec

RISK = "risk"
MARGIN = "margin"

REQUIRED_COLUMNS = {"timestamp", "close", "point_value_known", "margin_rate"}

# Погрешность float: 3 000 ₽ ÷ 300.0000000000007 ₽ должно дать 10 контрактов, а не 9.
FLOOR_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SizingRules:
    """Доли капитала: риск одной сделки и предел ГО позиции."""

    risk_fraction: float = 0.01
    max_margin_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.risk_fraction < 1.0:
            raise ValueError("Доля риска должна быть в (0, 1)")
        if not 0.0 < self.max_margin_fraction <= 1.0:
            raise ValueError("Доля ГО должна быть в (0, 1]")


@dataclass(frozen=True)
class PositionSize:
    """Число контрактов и почему именно столько.

    `limit` — какой лимит сработал: "risk" или "margin". Если контрактов 0,
    сигнал пропускается, и `limit` — причина пропуска.
    """

    contracts: int
    limit: str
    capital: float
    risk_per_contract: float
    margin_per_contract: float
    contracts_by_risk: int
    contracts_by_margin: int

    @property
    def skipped(self) -> bool:
        return self.contracts == 0

    @property
    def risk(self) -> float:
        """Плановый риск позиции до стопа в рублях — 1R сделки."""
        return self.contracts * self.risk_per_contract

    @property
    def margin(self) -> float:
        """ГО позиции в рублях на момент сигнала."""
        return self.contracts * self.margin_per_contract


def _positive(row: dict[str, Any], column: str) -> float:
    value = row[column]
    if value is None or not value > 0:
        raise ValueError(f"Нет {column} на свече сигнала {row['timestamp']}")
    return float(value)


def _whole_contracts(budget: float, per_contract: float) -> int:
    return math.floor(budget / per_contract + FLOOR_TOLERANCE)


def size_position(
    spec: ContractSpec,
    bars: pl.DataFrame,
    signal_bar: int,
    signal: Signal,
    capital: float,
    rules: SizingRules | None = None,
) -> PositionSize:
    """Число контрактов для сигнала по закрытию свечи `signal_bar`.

    `capital` — капитал на момент сигнала. Число контрактов в `signal`
    не используется; сигнал для симулятора:
    `dataclasses.replace(signal, contracts=size.contracts)`.
    """
    rules = rules or SizingRules()
    missing = REQUIRED_COLUMNS - set(bars.columns)
    if missing:
        raise ValueError(f"В барах нет столбцов: {', '.join(sorted(missing))}")
    if not 0 <= signal_bar < bars.height:
        raise ValueError("Нет свечи сигнала")
    if not capital > 0:
        raise ValueError("Капитал должен быть > 0")

    row = bars.row(signal_bar, named=True)
    close = float(row["close"])
    point_value = _positive(row, "point_value_known")
    margin_rate = _positive(row, "margin_rate")
    distance = signal.side * (close - signal.stop)
    if distance <= 0:
        raise ValueError("Стоп должен быть с убыточной стороны от закрытия сигнала")

    risk_per_contract = (
        (distance + 2 * spec.slippage) * point_value
        + commission(spec, close, point_value)
        + commission(spec, signal.stop, point_value)
    )
    margin_per_contract = close * point_value * margin_rate
    by_risk = _whole_contracts(capital * rules.risk_fraction, risk_per_contract)
    by_margin = _whole_contracts(
        capital * rules.max_margin_fraction, margin_per_contract
    )
    return PositionSize(
        contracts=min(by_risk, by_margin),
        limit=RISK if by_risk <= by_margin else MARGIN,
        capital=capital,
        risk_per_contract=risk_per_contract,
        margin_per_contract=margin_per_contract,
        contracts_by_risk=by_risk,
        contracts_by_margin=by_margin,
    )
