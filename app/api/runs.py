"""Фоновые запуски конвейера данных из пульта.

Пульт запускает шаги движка (data.pipeline) в фоновом потоке и помнит каждый
запуск: какие шаги, чем кончились, лог. Журнал лежит в app/runs/: history.json
(последние запуски) и лог каждого запуска — `<id>.log`. В data/ пульт сам ничего
не пишет: пишут только скрипты шагов.

Одновременно идут не больше двух запусков: один, который пишет данные (его
защищает блокировка data/.pipeline.lock — от второго запуска и из пульта, и из
терминала), и одна проверка (она только читает).

Пульт перезапустили посреди шага — скрипт доработает сам (он в своей сессии,
пишет лог в файл и держит блокировку), а запуск в журнале станет «прерван».
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from core.config import PROJECT_ROOT
from data.pipeline import (
    STEPS,
    UPDATE_ALL,
    DataBusy,
    DataLock,
    Step,
    StepKind,
    StepResult,
    acquire,
    holder,
    launch,
    run_steps,
)
from data.pipeline.run import now

HISTORY_FILE = "history.json"
KEEP_RUNS = 50  # старые запуски и их логи удаляются
LOG_CHUNK = 256 * 1024  # сколько лога отдавать за один запрос

# running — идёт; ok — все шаги ok; error — хоть один шаг упал или пропущен;
# stopped — остановили по запросу; interrupted — пульт перезапустили посреди запуска.
RunStatus = Literal["running", "ok", "error", "stopped", "interrupted"]


@dataclass(frozen=True)
class Run:
    """Запуск из пульта: один шаг или «Обновить всё»."""

    id: str  # «20261006-231500-futures_min»: время старта и что запущено
    title: str  # «Обновить всё», «Минутные свечи»
    kind: StepKind
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    stop_requested: bool  # попросили остановиться после текущего шага
    steps: tuple[StepResult, ...]


@dataclass(frozen=True)
class PipelineState:
    """Экран «Загрузка»: шаги, порядок «Обновить всё», журнал запусков и чужая загрузка."""

    steps: tuple[Step, ...]
    update_all: tuple[str, ...]  # id шагов «Обновить всё» по порядку
    runs: tuple[Run, ...]  # новые сверху
    busy: str | None  # данные пишет процесс вне пульта (терминал): кто; None — свободно или пишет сам пульт
    runs_dir: str  # где лежат логи: `<runs_dir>/<id запуска>.log`, от корня проекта, если внутри него


@dataclass(frozen=True)
class LogChunk:
    """Кусок лога запуска с позиции offset."""

    text: str
    offset: int  # откуда читать в следующий раз
    complete: bool  # запуск закончен и лог прочитан до конца


class RunRejected(Exception):
    """Запуск невозможен; status_code — код ответа API (400, 404, 409)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def display_path(path: Path) -> str:
    """Путь от корня проекта («app/runs»), если папка внутри него; иначе полный."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_status(results: Sequence[StepResult]) -> RunStatus:
    statuses = {result.status for result in results}
    if statuses & {"pending", "running"}:
        return "running"
    if "stopped" in statuses:
        return "stopped"
    return "ok" if statuses == {"ok"} else "error"


def interrupted(run: Run) -> Run:
    """Запуск, который пульт не довёл: процесс пульта завершился посреди него."""
    steps = tuple(
        replace(result, status="error", note="пульт перезапустили во время шага: скрипт мог доработать сам — смотрите лог")
        if result.status == "running"
        else replace(result, status="stopped", note="не запускался: пульт перезапустили")
        if result.status == "pending"
        else result
        for result in run.steps
    )
    return replace(run, status="interrupted", steps=steps)


def read_history(path: Path) -> list[Run]:
    """Журнал запусков; нет файла или он испорчен — пустой (журнал не настройка, терять его не страшно)."""
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text())
        return [
            Run(
                **{
                    **item,
                    "started_at": datetime.fromisoformat(item["started_at"]),
                    "finished_at": datetime.fromisoformat(item["finished_at"]) if item["finished_at"] else None,
                    "steps": tuple(
                        StepResult(
                            **{
                                **step,
                                "started_at": datetime.fromisoformat(step["started_at"]) if step["started_at"] else None,
                                "finished_at": datetime.fromisoformat(step["finished_at"]) if step["finished_at"] else None,
                            }
                        )
                        for step in item["steps"]
                    ),
                }
            )
            for item in raw
        ]
    except (ValueError, KeyError, TypeError):
        return []


def write_json(path: Path, value: object) -> None:
    """Сначала временный файл, потом подмена: оборванная запись не испортит прежний журнал."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as tmp:
        json.dump(value, tmp, ensure_ascii=False, indent=1, default=str)
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, path)


class RunManager:
    """Запуски конвейера: старт, остановка, журнал, логи. Один на процесс пульта."""

    def __init__(self, runs_dir: Path, data_root: Path, steps: Sequence[Step] = STEPS) -> None:
        self.runs_dir = runs_dir
        self.data_root = data_root
        self.steps = tuple(steps)
        self.by_id = {step.id: step for step in self.steps}
        self.mutex = threading.Lock()
        self.stop_requests: set[str] = set()
        # Запуск «идёт» в журнале, а пульт только что стартовал — значит, прежний процесс пульта
        # завершился посреди него. На диск это попадёт со следующим сохранением журнала.
        self.runs: list[Run] = [
            interrupted(run) if run.status == "running" else run for run in read_history(runs_dir / HISTORY_FILE)
        ]

    # ------------------------------------------------------------ чтение
    def state(self) -> PipelineState:
        with self.mutex:
            runs = tuple(self.runs)
        writing_here = any(run.status == "running" and run.kind == "data" for run in runs)
        return PipelineState(
            steps=self.steps,
            update_all=tuple(step for step in UPDATE_ALL if step in self.by_id),
            runs=runs,
            busy=None if writing_here else holder(self.data_root),
            runs_dir=display_path(self.runs_dir),
        )

    def log_file(self, run_id: str) -> Path:
        """Файл лога запуска целиком (для ссылки «Открыть файл»)."""
        self.find(run_id)  # неизвестный запуск — 404, а не чтение произвольного файла
        path = self.runs_dir / f"{run_id}.log"
        if not path.is_file():
            raise RunRejected(404, f"Лога запуска {run_id} нет: скрипт ещё ничего не вывел")
        return path

    def find(self, run_id: str) -> Run:
        with self.mutex:
            run = next((run for run in self.runs if run.id == run_id), None)
        if run is None:
            raise RunRejected(404, f"Нет запуска {run_id}")
        return run

    def read_log(self, run_id: str, offset: int) -> LogChunk:
        run = self.find(run_id)
        path = self.runs_dir / f"{run_id}.log"
        if not path.is_file():
            return LogChunk(text="", offset=0, complete=run.status != "running")
        with path.open("rb") as log:
            log.seek(max(0, offset))
            data = log.read(LOG_CHUNK)
            size = os.fstat(log.fileno()).st_size
        text, used = decode_whole_letters(data)
        end = max(0, offset) + used
        return LogChunk(text=text, offset=end, complete=run.status != "running" and end >= size)

    # ------------------------------------------------------------ запуск
    def start(self, step_ids: Sequence[str]) -> Run:
        unknown = [step_id for step_id in step_ids if step_id not in self.by_id]
        if not step_ids or unknown:
            raise RunRejected(400, f"Нет таких шагов: {', '.join(unknown)}" if unknown else "Не выбрано ни одного шага")
        wanted = set(step_ids)
        steps = [step for step in self.steps if step.id in wanted]  # всегда в порядке реестра
        kinds = {step.kind for step in steps}
        if len(kinds) > 1:
            raise RunRejected(400, "Загрузку и проверки запускайте отдельно")
        kind: StepKind = kinds.pop()
        title = "Обновить всё" if wanted == set(UPDATE_ALL) else ", ".join(step.title for step in steps)

        with self.mutex:
            current = next((run for run in self.runs if run.status == "running" and run.kind == kind), None)
            if current is not None:
                raise RunRejected(409, f"Уже идёт: {current.title}")
            lock: DataLock | None = None
            if kind == "data":
                try:
                    lock = acquire(self.data_root, f"пульт: {title}")
                except DataBusy as busy:
                    raise RunRejected(409, f"{busy}. Дождитесь окончания.") from None
            started = now()
            run = Run(
                id=self._new_id(started, "all" if title == "Обновить всё" else "-".join(step.id for step in steps)),
                title=title,
                kind=kind,
                status="running",
                started_at=started,
                finished_at=None,
                stop_requested=False,
                steps=tuple(StepResult(step=step.id, status="pending") for step in steps),
            )
            self.runs.insert(0, run)
            self._save()
        threading.Thread(target=self._work, args=(run.id, steps, lock), name=f"run-{run.id}", daemon=True).start()
        return run

    def stop(self, run_id: str) -> Run:
        """Остановить после текущего шага. Текущий шаг не обрывается: оборванная запись испортила бы parquet."""
        run = self.find(run_id)
        if run.status != "running":
            raise RunRejected(409, f"Запуск уже закончен: {run.title}")
        with self.mutex:
            self.stop_requests.add(run_id)
            return self._update(run_id, stop_requested=True)

    # ------------------------------------------------------------ фон
    def _work(self, run_id: str, steps: list[Step], lock: DataLock | None) -> None:
        log_path = self.runs_dir / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab") as log:

                def run_one(step: Step) -> int:
                    position = steps.index(step) + 1
                    log.write(f"── {position}/{len(steps)} {step.title}: {step.command}\n".encode())
                    log.flush()
                    process = launch(step, log=log, lock=lock)
                    if lock is not None:
                        lock.describe(f"пульт: {step.title}", process.pid)
                    code = process.wait()
                    log.write(f"── {step.title}: код выхода {code}\n\n".encode())
                    log.flush()
                    return code

                def on_change(results: list[StepResult]) -> None:
                    with self.mutex:
                        self._update(run_id, steps=tuple(results))

                results = run_steps(
                    steps, run_one, should_stop=lambda: run_id in self.stop_requests, on_change=on_change
                )
            status = run_status(results)
        except Exception as error:  # noqa: BLE001 — сбой пульта не должен оставить запуск «идущим» навсегда
            with log_path.open("ab") as log:
                log.write(f"\nСбой пульта: {error!r}\n".encode())
            status = "error"
        finally:
            if lock is not None:
                lock.release()
        with self.mutex:
            self.stop_requests.discard(run_id)
            self._update(run_id, status=status, finished_at=now())

    # ------------------------------------------------------------ под mutex
    def _new_id(self, started: datetime, what: str) -> str:
        base = f"{started:%Y%m%d-%H%M%S}-{what}"
        taken = {run.id for run in self.runs}
        run_id, n = base, 1
        while run_id in taken:
            n += 1
            run_id = f"{base}-{n}"
        return run_id

    def _update(self, run_id: str, **changes: object) -> Run:
        index = next(i for i, run in enumerate(self.runs) if run.id == run_id)
        self.runs[index] = replace(self.runs[index], **changes)  # type: ignore[arg-type]
        self._save()
        return self.runs[index]

    def _save(self) -> None:
        for old in self.runs[KEEP_RUNS:]:
            (self.runs_dir / f"{old.id}.log").unlink(missing_ok=True)
        del self.runs[KEEP_RUNS:]
        write_json(self.runs_dir / HISTORY_FILE, [asdict(run) for run in self.runs])


def decode_whole_letters(data: bytes) -> tuple[str, int]:
    """Текст и сколько байт он занял. Кусок лога может кончаться на середине
    русской буквы (2 байта в UTF-8): её начало не отдаём, дочитаем в следующий раз."""
    for cut in range(min(4, len(data) + 1)):
        try:
            return data[: len(data) - cut].decode(), len(data) - cut
        except UnicodeDecodeError:
            continue
    return data.decode(errors="replace"), len(data)  # не UTF-8 — показываем как есть
