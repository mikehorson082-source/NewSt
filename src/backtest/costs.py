"""Издержки сделок: проскальзывание, комиссия и перенос контракта.

Проскальзывание моделируется в тиках и ухудшает цену любой заявки:
покупка по `цена + slippage`, продажа по `цена - slippage`.
Перенос (ролл) — это закрытие старого контракта и открытие нового:
две ноги проскальзывания плюс комиссии двух сделок.

Рубли за пункт `point_value` передаются явно — те, что действуют на баре
сделки: у долларовых контрактов они меняются вместе с курсом клиринга.

Влияние на рынок (`impact_points`, ML4T глава 18): чем большую долю объёма
свечи забирает заявка, тем хуже цена: η × σ × цена × √(контракты ÷ объём).
Пример: σ свечи Brent обычно 0.22% (медиана 2022–2026), цена $70, η = 1.
25 контрактов в свечу, где торговали 50: 0.0022 × 70 × √0.5 ≈ $0.11 —
11 шагов сверх обычного. Те же 25 контрактов в свечу на 4 000:
0.0022 × 70 × √(25 / 4 000) ≈ $0.012 — чуть больше 1 шага.
"""

from __future__ import annotations

import math

from backtest.specs import ContractSpec

BUY = 1
SELL = -1


def _check_point_value(point_value: float) -> None:
    if not point_value > 0:
        raise ValueError("Рубли за пункт должны быть > 0")


def impact_points(
    spec: ContractSpec,
    price: float,
    contracts: int,
    bar_volume: float | None,
    bar_sigma: float | None,
) -> float:
    """Влияние заявки на цену в пунктах: η × σ × цена × √(контракты ÷ объём).

    `bar_volume` — фактический объём свечи исполнения: на решение о входе он
    не влияет, а показывает, как рынок исполнил заявку. `bar_sigma` — обычное
    движение 15-минутной свечи в долях цены (`markets.add_bar_sigma`); в
    начале ряда, пока истории нет, его нет — влияние 0.
    """
    if spec.impact == 0 or bar_sigma is None:
        return 0.0
    if bar_volume is None or not bar_volume > 0:
        raise ValueError("Для влияния на рынок нужен объём свечи > 0")
    return spec.impact * bar_sigma * price * math.sqrt(contracts / bar_volume)


def fill_price(
    spec: ContractSpec,
    price: float,
    order_side: int,
    contracts: int = 1,
    impact: float = 0.0,
) -> float:
    """Цена исполнения заявки с проскальзыванием против покупателя/продавца.

    `impact` — влияние на рынок в пунктах (`impact_points`), сверх обычного
    проскальзывания.
    """
    if order_side not in (BUY, SELL):
        raise ValueError("Сторона заявки должна быть +1 или -1")
    if price <= 0:
        raise ValueError("Цена заявки должна быть > 0")
    if contracts < 1:
        raise ValueError("Число контрактов должно быть >= 1")
    return price + order_side * (spec.slippage + impact)


def commission(
    spec: ContractSpec,
    price: float,
    point_value: float,
    contracts: int = 1,
) -> float:
    """Комиссия одной сделки в рублях: доля стоимости контрактов."""
    if price <= 0:
        raise ValueError("Цена сделки должна быть > 0")
    _check_point_value(point_value)
    if contracts < 1:
        raise ValueError("Число контрактов должно быть >= 1")
    return price * point_value * contracts * spec.fee_rate


def roll_cost(
    spec: ContractSpec,
    old_price: float,
    new_price: float,
    point_value: float,
    contracts: int = 1,
) -> float:
    """Стоимость переноса позиции из старого контракта в новый.

    Старый контракт закрывается, новый открывается: проскальзывание двух ног
    плюс комиссия двух сделок. Базисный разрыв между контрактами сюда не входит,
    он компенсируется ценовой поправкой ряда.
    """
    if old_price <= 0 or new_price <= 0:
        raise ValueError("Цены переноса должны быть > 0")
    _check_point_value(point_value)
    if contracts < 1:
        raise ValueError("Число контрактов должно быть >= 1")
    slippage = 2 * spec.slippage * point_value * contracts
    fees = commission(spec, old_price, point_value, contracts) + commission(
        spec, new_price, point_value, contracts
    )
    return slippage + fees
