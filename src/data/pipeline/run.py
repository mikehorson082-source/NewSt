"""Запуск шагов: один скрипт — один процесс, шаги по очереди.

Скрипт запускается тем же Python, что и вызывающий (это и есть `uv run python`),
из корня проекта — ровно как команда из Step.command в терминале.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import BinaryIO
from zoneinfo import ZoneInfo

from core.config import PROJECT_ROOT

from .lock import HELD_ENV, DataLock
from .model import Step, StepResult

MSK = ZoneInfo("Europe/Moscow")


def now() -> datetime:
    return datetime.now(MSK).replace(microsecond=0)


def launch(step: Step, *, log: BinaryIO | None = None, lock: DataLock | None = None) -> subprocess.Popen[bytes]:
    """Запускает скрипт шага и сразу возвращает процесс.

    log — файл для вывода (stdout и stderr вместе); None — вывод в терминал.
    С файлом процесс живёт в своей сессии: Ctrl+C или перезапуск пульта не
    оборвёт его посреди записи parquet, и вывод не потеряется.
    lock — блокировка data/, которую держит вызывающий: процесс получает её
    копию и не берёт блокировку сам.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    pass_fds: tuple[int, ...] = ()
    if lock is not None:
        env[HELD_ENV] = "1"
        pass_fds = (lock.fd,)
    # Команда — из реестра шагов, а не от пользователя.
    return subprocess.Popen(
        [sys.executable, *step.args],
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT if log is not None else None,
        pass_fds=pass_fds,
        start_new_session=log is not None,
    )


def exit_note(code: int) -> str:
    if code < 0:
        return f"прерван сигналом {-code}"
    if code == 2:
        return "код выхода 2: неверные аргументы или данные уже пишет другой процесс"
    return f"код выхода {code}"


def run_steps(
    steps: Sequence[Step],
    run_one: Callable[[Step], int],
    *,
    should_stop: Callable[[], bool] = lambda: False,
    on_change: Callable[[list[StepResult]], None] = lambda _results: None,
) -> list[StepResult]:
    """Шаги по очереди; run_one запускает шаг и возвращает код выхода.

    Упал шаг — следующие идут дальше, кроме тех, кому он нужен (Step.needs):
    они пропускаются. should_stop — «остановить после текущего шага»:
    оставшиеся помечаются stopped. on_change вызывается при каждой смене
    статуса — пульт по нему обновляет экран.

    Пример: справочники ok, дневки ok, минутки упали → FUTOI запускается,
    склейка пропускается (ей нужны минутки).
    """
    titles = {step.id: step.title for step in steps}
    results = [StepResult(step=step.id, status="pending") for step in steps]
    failed: set[str] = set()
    for i, step in enumerate(steps):
        if should_stop():
            results[i:] = [replace(result, status="stopped", note="остановлено по запросу") for result in results[i:]]
            on_change(list(results))
            break
        broken = [titles[need] for need in step.needs if need in failed]
        if broken:
            names = ", ".join(f"«{title}»" for title in broken)
            results[i] = replace(results[i], status="skipped", note=f"не запускался: не удался шаг {names}")
            failed.add(step.id)
            on_change(list(results))
            continue
        results[i] = replace(results[i], status="running", started_at=now())
        on_change(list(results))
        try:
            code = run_one(step)
        except OSError as error:  # скрипт не запустился: нет файла, нет прав
            results[i] = replace(results[i], status="error", finished_at=now(), note=f"не запустился: {error}")
        else:
            ok = code == 0
            results[i] = replace(
                results[i],
                status="ok" if ok else "error",
                finished_at=now(),
                exit_code=code,
                note=None if ok else exit_note(code),
            )
        if results[i].status != "ok":
            failed.add(step.id)
        on_change(list(results))
    return results
