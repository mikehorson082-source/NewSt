"""Проверки конвейера данных. Запуск: uv run pytest src/data/test_pipeline.py

Настоящие загрузчики не запускаются: шаги в тестах — `python -c "…"`.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from data.pipeline import (
    STEPS,
    UPDATE_ALL,
    DataBusy,
    Step,
    acquire,
    holder,
    launch,
    run_steps,
)
from data.pipeline.lock import HELD_ENV


def fake(id: str, code: str = "pass", needs: tuple[str, ...] = (), kind: str = "data") -> Step:
    return Step(
        id=id,
        title=id.capitalize(),
        description="",
        kind=kind,  # type: ignore[arg-type]
        args=("-c", code),
        command=f"python -c {code!r}",
        needs=needs,
    )


# ---------------------------------------------------------------- реестр
def test_update_all_order():
    assert UPDATE_ALL == ("reference", "futures_daily", "futures_min", "futoi", "continuous")


def test_tests_are_a_separate_check():
    tests = next(step for step in STEPS if step.id == "backtest_tests")
    assert (tests.kind, tests.command) == ("check", "uv run pytest src/backtest")
    assert "backtest_tests" not in UPDATE_ALL


def test_downloaders_only_update_never_overwrite():
    for step in STEPS:
        assert not {"--overwrite", "--force"} & set(step.args), step.id
        if step.args[0].startswith("src/data/download_"):
            assert "--update" in step.args, step.id
        assert Path(step.args[0]).exists() or step.args[0] == "-m", step.id


def test_continuous_needs_minutes():
    continuous = next(step for step in STEPS if step.id == "continuous")
    assert continuous.needs == ("futures_min",)


# ---------------------------------------------------------------- очередь
def test_failed_step_skips_only_its_dependents():
    steps = [fake("minutes"), fake("futoi"), fake("glue", needs=("minutes",))]
    codes = {"minutes": 1, "futoi": 0}
    results = run_steps(steps, lambda step: codes[step.id])
    assert [r.status for r in results] == ["error", "ok", "skipped"]
    assert results[0].note == "код выхода 1"
    assert results[2].note == "не запускался: не удался шаг «Minutes»"
    assert results[2].started_at is None


def test_stop_after_current_step():
    steps = [fake("a"), fake("b"), fake("c")]
    ran: list[str] = []

    def run_one(step):
        ran.append(step.id)
        return 0

    results = run_steps(steps, run_one, should_stop=lambda: len(ran) == 1)
    assert ran == ["a"]
    assert [r.status for r in results] == ["ok", "stopped", "stopped"]


def test_progress_is_reported_on_every_change():
    seen: list[list[str]] = []
    run_steps([fake("a"), fake("b")], lambda _step: 0, on_change=lambda rs: seen.append([r.status for r in rs]))
    assert seen == [["running", "pending"], ["ok", "pending"], ["ok", "running"], ["ok", "ok"]]


def test_step_that_cannot_start_is_an_error():
    def run_one(_step):
        raise FileNotFoundError("нет скрипта")

    (result,) = run_steps([fake("a")], run_one)
    assert (result.status, result.note) == ("error", "не запустился: нет скрипта")


# ---------------------------------------------------------------- блокировка
def test_second_writer_gets_busy_with_holder(tmp_path):
    lock = acquire(tmp_path, "пульт: Минутные свечи")
    with pytest.raises(DataBusy) as busy:
        acquire(tmp_path, "терминал")
    assert busy.value.holder.startswith("пульт: Минутные свечи (PID ")
    assert holder(tmp_path) == busy.value.holder
    lock.release()
    assert holder(tmp_path) is None
    acquire(tmp_path, "терминал").release()


def test_no_lock_file_means_free(tmp_path):
    assert holder(tmp_path / "data") is None


def hold_in_child(data_root: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    code = f"from data.pipeline.lock import hold_data_lock; from pathlib import Path; hold_data_lock('скрипт', Path({str(data_root)!r})); print('работаю')"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)


def test_script_exits_when_data_is_busy(tmp_path):
    lock = acquire(tmp_path, "пульт: Склейка")
    try:
        done = hold_in_child(tmp_path)
    finally:
        lock.release()
    assert done.returncode == 2
    assert "Данные сейчас пишет пульт: Склейка" in done.stderr
    assert "работаю" not in done.stdout


def test_script_of_a_step_does_not_lock_twice(tmp_path):
    import os

    lock = acquire(tmp_path, "пульт: Склейка")
    try:
        done = hold_in_child(tmp_path, env={**os.environ, HELD_ENV: "1"})
    finally:
        lock.release()
    assert (done.returncode, done.stdout.strip()) == (0, "работаю")


def test_lock_lives_while_child_lives(tmp_path):
    """Пульт перезапустили посреди шага — блокировка держится, пока скрипт пишет."""
    lock = acquire(tmp_path, "пульт: Минутные свечи")
    log_path = tmp_path / "run.log"
    with log_path.open("ab") as log:
        process = launch(fake("slow", "import os, time; print(os.environ['NEWST_DATA_LOCK_HELD']); time.sleep(1)"), log=log, lock=lock)
    lock.release()  # родитель «умер»: его дескриптор закрыт
    time.sleep(0.2)
    assert holder(tmp_path) is not None
    assert process.wait(timeout=10) == 0
    assert holder(tmp_path) is None
    assert log_path.read_text().strip() == "1"


def test_launch_writes_stdout_and_stderr_to_log(tmp_path):
    log_path = tmp_path / "run.log"
    with log_path.open("ab") as log:
        code = launch(fake("noisy", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"), log=log).wait()
    assert code == 3
    assert sorted(log_path.read_text().split()) == ["err", "out"]
