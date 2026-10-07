"""Один писатель в data/ за раз.

Два процесса, пишущие один parquet, портят его. Поэтому всё, что пишет в
data/, берёт общую блокировку — файл data/.pipeline.lock (flock). Её берёт
тот, кто запустил загрузку: пульт, `python -m data.pipeline` или сам скрипт,
запущенный из терминала. Процесс завершился, даже аварийно, — система сама
снимает блокировку: «зависших» блокировок не бывает.

Пример: в пульте идёт «Обновить всё», а в терминале запустили
download_futures_min.py --update. Скрипт сразу выходит и пишет, кто сейчас
пишет данные: «пульт: Минутные свечи (PID 4182, с 23:10)».
"""

from __future__ import annotations

import fcntl
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from core.config import DATA_DIR

MSK = ZoneInfo("Europe/Moscow")
LOCK_NAME = ".pipeline.lock"

# Родитель уже держит блокировку и передал её дочернему процессу (шагу конвейера):
# скрипт шага не берёт её второй раз.
HELD_ENV = "NEWST_DATA_LOCK_HELD"


class DataBusy(RuntimeError):
    """Данные уже пишет другой процесс; holder — кто именно."""

    def __init__(self, holder: str) -> None:
        super().__init__(f"Данные сейчас пишет {holder}")
        self.holder = holder


class DataLock:
    """Взятая блокировка. fd передаётся дочерним процессам: пока жив хоть один, блокировка держится."""

    def __init__(self, fd: int) -> None:
        self.fd = fd

    def describe(self, what: str, pid: int) -> None:
        """Записывает, кто держит блокировку: это увидит тот, кому она не досталась."""
        text = f"{what} (PID {pid}, с {datetime.now(MSK):%H:%M})".encode()
        os.ftruncate(self.fd, 0)
        os.pwrite(self.fd, text, 0)

    def release(self) -> None:
        # Только закрыть свой дескриптор, без LOCK_UN: LOCK_UN снял бы блокировку
        # и у дочернего процесса, который ещё пишет (дескриптор у них общий).
        os.close(self.fd)


def lock_path(data_root: Path) -> Path:
    return data_root / LOCK_NAME


def _open(data_root: Path) -> int:
    data_root.mkdir(parents=True, exist_ok=True)
    return os.open(lock_path(data_root), os.O_RDWR | os.O_CREAT, 0o644)


def _holder_text(fd: int) -> str:
    return os.pread(fd, 4096, 0).decode(errors="replace").strip() or "другой процесс"


def acquire(data_root: Path, what: str) -> DataLock:
    """Берёт блокировку без ожидания; занято — DataBusy."""
    fd = _open(data_root)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = _holder_text(fd)
        os.close(fd)
        raise DataBusy(holder) from None
    lock = DataLock(fd)
    lock.describe(what, os.getpid())
    return lock


def holder(data_root: Path) -> str | None:
    """Кто сейчас пишет данные; None — никто. Сама блокировка не берётся."""
    if not lock_path(data_root).exists():
        return None
    fd = os.open(lock_path(data_root), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return _holder_text(fd)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


_held: DataLock | None = None


def hold_data_lock(what: str, data_root: Path = DATA_DIR) -> None:
    """Для main() скриптов загрузки: держать блокировку до конца процесса.

    Скрипт запущен шагом конвейера (родитель уже держит блокировку) — ничего
    не делает. Занято — печатает, кто пишет данные, и выходит с кодом 2.
    """
    global _held
    if _held is not None or os.environ.get(HELD_ENV) == "1":
        return
    try:
        _held = acquire(data_root, what)
    except DataBusy as busy:
        print(f"{busy}. Дождитесь окончания и запустите снова.", file=sys.stderr)
        raise SystemExit(2) from None
