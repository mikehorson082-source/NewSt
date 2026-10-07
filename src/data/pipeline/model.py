"""Модель загрузки данных: шаги конвейера и итоги их запусков.

Шаг — один скрипт движка с фиксированными аргументами. Например, шаг
«Минутные свечи» — это `uv run python src/data/download_futures_min.py --update`.
Пульт показывает эту же команду: её можно запустить из терминала руками,
результат будет тот же.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# data — пишет в data/ (одновременно только один такой шаг), check — только читает.
StepKind = Literal["data", "check"]

# pending — ждёт очереди; skipped — не запускался, потому что не удался нужный ему шаг;
# stopped — не запускался, потому что попросили остановиться.
StepStatus = Literal["pending", "running", "ok", "error", "skipped", "stopped"]


@dataclass(frozen=True)
class Step:
    """Шаг конвейера: что запускается, куда пишет и от чего зависит."""

    id: str  # «futures_min»
    title: str  # «Минутные свечи»
    description: str  # что делает, по-русски
    kind: StepKind
    args: tuple[str, ...]  # аргументы Python: ("src/data/download_futures_min.py", "--update")
    command: str  # та же команда для терминала: «uv run python src/data/… --update»
    writes: tuple[str, ...] = ()  # папки относительно data/, куда шаг пишет
    # В одном запуске шаг пропускается, если не удался один из этих шагов:
    # склейка на недокачанных минутках дала бы ряд с дырой в конце.
    needs: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepResult:
    """Итог шага в одном запуске."""

    step: str  # Step.id
    status: StepStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None  # 0 — успех; отрицательный — процесс убит сигналом
    note: str | None = None  # почему не ok
