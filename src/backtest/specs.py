"""Спецификации контрактов для симулятора бэктеста.

Точка отсчёта — `point_value` (k): рублей за один пункт цены.
Ноционал контракта = цена × k, ГО = ноционал × margin_rate.
Для MOEX k = стоимость шага цены / шаг цены (снимок ISS или дневные данные).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """Постоянные параметры контракта, не зависящие от даты."""

    asset: str
    point_value: float
    tick_size: float
    margin_rate: float
    exchange_fee_rate: float
    broker_fee_rate: float = 0.0
    slippage_ticks: float = 1.0

    def __post_init__(self) -> None:
        if not self.asset:
            raise ValueError("У контракта должно быть имя актива")
        if self.point_value <= 0:
            raise ValueError(f"{self.asset}: point_value должен быть > 0")
        if self.tick_size <= 0:
            raise ValueError(f"{self.asset}: tick_size должен быть > 0")
        if not 0.0 < self.margin_rate < 1.0:
            raise ValueError(f"{self.asset}: margin_rate должен быть в (0, 1)")
        if self.exchange_fee_rate < 0 or self.broker_fee_rate < 0:
            raise ValueError(
                f"{self.asset}: ставки комиссий не могут быть отрицательными"
            )
        if self.slippage_ticks < 0:
            raise ValueError(
                f"{self.asset}: slippage_ticks не может быть отрицательным"
            )

    @property
    def fee_rate(self) -> float:
        """Суммарная доля ноционала: биржа + брокер."""
        return self.exchange_fee_rate + self.broker_fee_rate

    @property
    def slippage(self) -> float:
        """Проскальзывание одной стороны в пунктах цены."""
        return self.slippage_ticks * self.tick_size

    def notional(self, price: float, contracts: int = 1) -> float:
        """Стоимость позиции в рублях."""
        return price * self.point_value * contracts

    def margin(self, price: float, contracts: int = 1) -> float:
        """ГО, заблокированное под позицию, в рублях."""
        return self.notional(price, contracts) * self.margin_rate
