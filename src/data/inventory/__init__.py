"""Инвентарь данных: что лежит в data/, за какие даты и насколько оно свежее.

Отчёт строится по инструментам: Фьючерсы и Акции → инструмент (нефть,
золото, Сбербанк) → всё, что по нему есть. Общие файлы — в разделе «Общее».

Функции возвращают данные, печатает только __main__. Этот же отчёт отдаёт
веб-пульт (app/api): движок про веб ничего не знает.

Модули: model — формат отчёта; scan — чтение файлов и свежесть; catalog —
что к какому инструменту относится; build — сборка отчёта.

Запуск:
    uv run python -m data.inventory              # по инструментам
    uv run python -m data.inventory --problems   # только то, что требует внимания
    uv run python -m data.inventory --json       # весь отчёт в JSON
"""

from .build import build_inventory
from .model import DatasetStatus, Fact, Instrument, InventoryReport, Section, Status

__all__ = ["DatasetStatus", "Fact", "Instrument", "InventoryReport", "Section", "Status", "build_inventory"]
