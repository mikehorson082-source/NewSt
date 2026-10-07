"""Постоянные правила торговли контрактом для симулятора бэктеста.

Здесь только то, что не меняется по датам: шаг цены, ставки комиссий
и проскальзывание. Рубли за пункт (k) и доля ГО меняются со временем
(у Brent и золота k идёт за курсом клиринга USD/RUB), поэтому они приходят
из баров — столбцы `point_value` и `margin_rate` (`markets.load_market`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """Постоянные параметры контракта, не зависящие от даты."""

    asset: str
    tick_size: float
    exchange_fee_rate: float
    broker_fee_rate: float = 0.0
    slippage_ticks: float = 1.0

    def __post_init__(self) -> None:
        if not self.asset:
            raise ValueError("У контракта должно быть имя актива")
        if self.tick_size <= 0:
            raise ValueError(f"{self.asset}: tick_size должен быть > 0")
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
