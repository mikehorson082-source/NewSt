"""Издержки сделок: проскальзывание, комиссия и перенос контракта.

Проскальзывание моделируется в тиках и ухудшает цену любой заявки:
покупка по `цена + slippage`, продажа по `цена - slippage`.
Перенос (ролл) — это закрытие старого контракта и открытие нового:
две ноги проскальзывания плюс комиссии двух сделок.

Рубли за пункт `point_value` передаются явно — те, что действуют на баре
сделки: у долларовых контрактов они меняются вместе с курсом клиринга.
"""

from __future__ import annotations

from backtest.specs import ContractSpec

BUY = 1
SELL = -1


def _check_point_value(point_value: float) -> None:
    if not point_value > 0:
        raise ValueError("Рубли за пункт должны быть > 0")


def fill_price(
    spec: ContractSpec,
    price: float,
    order_side: int,
    contracts: int = 1,
) -> float:
    """Цена исполнения заявки с проскальзыванием против покупателя/продавца."""
    if order_side not in (BUY, SELL):
        raise ValueError("Сторона заявки должна быть +1 или -1")
    if price <= 0:
        raise ValueError("Цена заявки должна быть > 0")
    if contracts < 1:
        raise ValueError("Число контрактов должно быть >= 1")
    return price + order_side * spec.slippage


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
