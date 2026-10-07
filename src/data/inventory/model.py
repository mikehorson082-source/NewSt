"""Формат отчёта инвентаря: раздел → инструмент → наборы данных.

Набор (DatasetStatus) хранится в отчёте один раз, а инструменты ссылаются
на него по id: курс клиринга USD/RUB нужен всем долларовым фьючерсам, но
проверяется и лежит в отчёте одной строкой.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

Status = Literal["ok", "lagging", "stale", "unchecked", "missing", "error"]

# Чем выше, тем хуже: так из двух статусов выбирается худший.
SEVERITY: dict[Status, int] = {
    "ok": 0,
    "unchecked": 0,
    "lagging": 1,
    "stale": 2,
    "missing": 3,
    "error": 3,
}


@dataclass(frozen=True)
class Fact:
    """Дополнительный факт о наборе: «Роллов: 53», «Контрактов: 77»."""

    label: str
    value: int | float | str


@dataclass(frozen=True)
class DatasetStatus:
    """Набор данных: папка контрактов, файл, срез справочника по инструменту."""

    id: str  # стабильный ключ: путь относительно data/, у среза — «путь#ключ»
    kind: str  # вид набора: futures_minutes, continuous, reference…
    title: str  # что это: «Минутные свечи», «margin_rates»
    description: str  # уточнение по-русски
    path: str  # файл или папка относительно data/
    status: Status
    files: int = 0
    rows: int | None = None
    first_date: date | None = None
    last_date: date | None = None
    size_bytes: int = 0  # у среза 0: размер есть только у файла целиком
    modified_at: datetime | None = None
    lag_days: int | None = None  # торговых дней после last_date до ожидаемой даты
    note: str | None = None  # почему такой статус
    # Правило проверки из rules.json по-русски: «Норма — отставание до 10 торговых дней. Бесплатный тариф…».
    rule: str | None = None
    facts: tuple[Fact, ...] = ()
    # Строк по календарным месяцам подряд: первое число — месяц first_date, последнее —
    # месяц last_date, месяц без данных — 0. Пусто — по месяцам не считается.
    # Пример: first_date 18.03.2019, monthly (9800, 21000, 0, …) — март, апрель, май без данных.
    monthly: tuple[int, ...] = ()


@dataclass(frozen=True)
class Instrument:
    """Инструмент (нефть, Сбербанк) или тематическая группа раздела «Общее»."""

    id: str
    title: str
    subtitle: str
    datasets: tuple[str, ...]  # id наборов из InventoryReport.datasets


@dataclass(frozen=True)
class Section:
    """Верхний уровень: Фьючерсы, Акции, Общее."""

    id: str
    title: str
    description: str
    instruments: tuple[Instrument, ...]


@dataclass(frozen=True)
class InventoryReport:
    """Весь инвентарь: разделы с инструментами, наборы и сводка по статусам."""

    generated_at: datetime
    expected_date: date  # последний завершённый торговый день: от него считается отставание
    data_root: str
    sections: tuple[Section, ...]
    datasets: tuple[DatasetStatus, ...]
    summary: dict[Status, int]  # число наборов по статусам
