"""Конвейер данных: загрузка с биржи, склейка и проверки — по шагам.

Шаг — один скрипт движка с фиксированными аргументами (steps.py). «Обновить
всё» — справочники → дневки → минутки → FUTOI → склейка. Писать в data/
может только один процесс за раз (lock.py). Этот же конвейер запускает
веб-пульт (app/api): движок про веб ничего не знает.

Модули: model — шаги и итоги; steps — реестр шагов; lock — блокировка data/;
run — запуск процессов и очередь шагов.

Запуск:
    uv run python -m data.pipeline                         # список шагов
    uv run python -m data.pipeline all                     # обновить всё
    uv run python -m data.pipeline futures_min continuous  # выбранные шаги
"""

from .lock import DataBusy, DataLock, acquire, hold_data_lock, holder
from .model import Step, StepKind, StepResult, StepStatus
from .run import launch, run_steps
from .steps import STEP_BY_ID, STEPS, UPDATE_ALL

__all__ = [
    "STEPS",
    "STEP_BY_ID",
    "UPDATE_ALL",
    "DataBusy",
    "DataLock",
    "Step",
    "StepKind",
    "StepResult",
    "StepStatus",
    "acquire",
    "hold_data_lock",
    "holder",
    "launch",
    "run_steps",
]
