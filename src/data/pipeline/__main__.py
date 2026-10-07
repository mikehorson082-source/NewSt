"""Конвейер в консоли: uv run python -m data.pipeline [all | шаг …]"""

from __future__ import annotations

import argparse
import sys

from core.config import DATA_DIR

from .lock import DataBusy, acquire
from .model import StepResult
from .run import launch, run_steps
from .steps import STEP_BY_ID, STEPS, UPDATE_ALL

STATUS_TEXT = {
    "ok": "ok",
    "error": "ОШИБКА",
    "skipped": "пропущен",
    "stopped": "остановлен",
    "pending": "не дошёл",
    "running": "идёт",
}


def print_steps() -> None:
    print("Шаги конвейера (в «all» — шаги загрузки по порядку):\n")
    for step in STEPS:
        mark = "all" if step.id in UPDATE_ALL else "   "
        print(f"  {mark}  {step.id:<15} {step.title} — {step.description}")
        print(f"       {'':<15} {step.command}")
    print("\nЗапуск: uv run python -m data.pipeline all   или   … futures_min continuous")


def duration(result: StepResult) -> str:
    if result.started_at is None or result.finished_at is None:
        return ""
    seconds = int((result.finished_at - result.started_at).total_seconds())
    return f"{seconds // 60}:{seconds % 60:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвейер данных: загрузка, склейка, проверки")
    parser.add_argument("steps", nargs="*", help="all или id шагов; без аргументов — список шагов")
    args = parser.parse_args()
    if not args.steps:
        print_steps()
        return
    wanted = set(UPDATE_ALL) if args.steps == ["all"] else set(args.steps)
    unknown = sorted(wanted - STEP_BY_ID.keys())
    if unknown:
        parser.error(f"нет таких шагов: {', '.join(unknown)} (список: uv run python -m data.pipeline)")
    steps = [step for step in STEPS if step.id in wanted]  # всегда в порядке реестра

    lock = None
    if any(step.kind == "data" for step in steps):
        try:
            lock = acquire(DATA_DIR, "python -m data.pipeline (терминал)")
        except DataBusy as busy:
            sys.exit(f"{busy}. Дождитесь окончания и запустите снова.")

    def run_one(step):
        position = steps.index(step) + 1
        print(f"\n── {position}/{len(steps)} {step.title}: {step.command}", flush=True)
        return launch(step, lock=lock).wait()

    try:
        results = run_steps(steps, run_one)
    finally:
        if lock is not None:
            lock.release()

    print("\nИтог:")
    for step, result in zip(steps, results, strict=True):
        note = f"  {result.note}" if result.note else ""
        print(f"  {STATUS_TEXT[result.status]:<10} {step.title:<16} {duration(result):>6}{note}")
    if any(result.status != "ok" for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
