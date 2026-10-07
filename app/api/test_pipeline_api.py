"""Раздел «Загрузка» API. Запуск: uv run pytest app

Шаги — `python -c "…"` вместо настоящих загрузчиков: тесты не ходят в сеть
и не трогают data/.
"""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from data.pipeline import Step, acquire


def fake(id: str, code: str = "print('готово')", kind: str = "data", needs: tuple[str, ...] = ()) -> Step:
    return Step(
        id=id,
        title=id.capitalize(),
        description="",
        kind=kind,  # type: ignore[arg-type]
        args=("-c", code),
        command=f"python -c {code!r}",
        needs=needs,
    )


STEPS = (
    fake("reference"),
    fake("futures_min", "import sys; print('минутки упали'); sys.exit(1)"),
    fake("continuous", needs=("reference", "futures_min")),
    fake("slow", "import time; print('начал', flush=True); time.sleep(1); print('кончил')"),
    fake("after"),
    fake("backtest_tests", "print('тесты ok')", kind="check"),
)


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return {"data_root": tmp_path / "data", "runs_dir": tmp_path / "runs", "settings_dir": tmp_path / "settings"}


@pytest.fixture
def client(paths, tmp_path) -> TestClient:
    return TestClient(create_app(frontend_dist=tmp_path / "dist", steps=STEPS, **paths))


def wait_done(client: TestClient, run_id: str, timeout: float = 15) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = next(run for run in client.get("/api/pipeline").json()["runs"] if run["id"] == run_id)
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError(f"запуск {run_id} не закончился за {timeout} с")


def full_log(client: TestClient, run_id: str) -> str:
    text, offset = "", 0
    while True:
        chunk = client.get("/api/pipeline/log", params={"run_id": run_id, "offset": offset}).json()
        text, offset = text + chunk["text"], chunk["offset"]
        if chunk["complete"]:
            return text


def full_log_so_far(client: TestClient, run_id: str) -> str:
    return client.get("/api/pipeline/log", params={"run_id": run_id}).json()["text"]


def test_state_lists_steps_and_empty_journal(client, paths):
    state = client.get("/api/pipeline").json()
    assert [step["id"] for step in state["steps"]] == [step.id for step in STEPS]
    assert state["runs"] == []
    assert state["busy"] is None
    assert state["runs_dir"].endswith("/runs")  # tmp вне проекта — полный путь
    assert not paths["runs_dir"].exists()  # чтение ничего не создаёт


def test_run_one_step_and_read_its_log(client, paths):
    run = client.post("/api/pipeline/runs", json={"steps": ["reference"]}).json()
    assert (run["title"], run["status"], run["kind"]) == ("Reference", "running", "data")
    done = wait_done(client, run["id"])
    assert done["status"] == "ok"
    assert done["steps"][0]["exit_code"] == 0
    assert "готово" in full_log(client, run["id"])
    file = client.get("/api/pipeline/log/file", params={"run_id": run["id"]})
    assert file.headers["content-type"].startswith("text/plain")
    assert "готово" in file.text
    assert client.get("/api/pipeline/log/file", params={"run_id": "../history"}).status_code == 404
    assert json.loads((paths["runs_dir"] / "history.json").read_text())[0]["status"] == "ok"
    assert client.get("/api/pipeline").json()["busy"] is None  # блокировка снята


def test_failed_step_skips_its_dependents(client):
    run = client.post("/api/pipeline/runs", json={"steps": ["continuous", "futures_min", "reference"]}).json()
    done = wait_done(client, run["id"])
    assert [(step["step"], step["status"]) for step in done["steps"]] == [
        ("reference", "ok"),
        ("futures_min", "error"),
        ("continuous", "skipped"),
    ]  # порядок — как в реестре, а не как прислали
    assert done["status"] == "error"
    log = full_log(client, run["id"])
    assert "минутки упали" in log
    assert "── 2/3 Futures_min" in log


def test_second_data_run_is_rejected_but_check_runs_alongside(client):
    slow = client.post("/api/pipeline/runs", json={"steps": ["slow"]}).json()
    second = client.post("/api/pipeline/runs", json={"steps": ["reference"]})
    assert second.status_code == 409
    assert second.json()["detail"] == "Уже идёт: Slow"
    check = client.post("/api/pipeline/runs", json={"steps": ["backtest_tests"]})
    assert check.status_code == 200
    assert wait_done(client, check.json()["id"])["status"] == "ok"
    assert wait_done(client, slow["id"])["status"] == "ok"


def test_writer_outside_the_app_blocks_data_runs(client, paths):
    lock = acquire(paths["data_root"], "download_futures_min.py")
    try:
        assert client.get("/api/pipeline").json()["busy"].startswith("download_futures_min.py (PID ")
        response = client.post("/api/pipeline/runs", json={"steps": ["reference"]})
        assert response.status_code == 409
        assert response.json()["detail"].startswith("Данные сейчас пишет download_futures_min.py")
    finally:
        lock.release()


def test_stop_after_current_step(client):
    run = client.post("/api/pipeline/runs", json={"steps": ["slow", "after"]}).json()
    deadline = time.monotonic() + 10
    while "начал" not in full_log_so_far(client, run["id"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    stopped = client.post("/api/pipeline/stop", json={"run_id": run["id"]}).json()
    assert stopped["stop_requested"] is True
    done = wait_done(client, run["id"])
    assert done["status"] == "stopped"
    assert [step["status"] for step in done["steps"]] == ["ok", "stopped"]  # текущий шаг доработал
    assert "кончил" in full_log(client, run["id"])
    assert client.post("/api/pipeline/stop", json={"run_id": run["id"]}).status_code == 409


def test_bad_requests(client):
    assert client.post("/api/pipeline/runs", json={"steps": []}).status_code == 400
    assert client.post("/api/pipeline/runs", json={"steps": ["nope"]}).json()["detail"] == "Нет таких шагов: nope"
    mixed = client.post("/api/pipeline/runs", json={"steps": ["reference", "backtest_tests"]})
    assert (mixed.status_code, mixed.json()["detail"]) == (400, "Загрузку и проверки запускайте отдельно")
    assert client.get("/api/pipeline/log", params={"run_id": "nope"}).status_code == 404


def test_update_all_title(paths, tmp_path):
    from data.pipeline import UPDATE_ALL

    steps = tuple(fake(step_id) for step_id in UPDATE_ALL)
    client = TestClient(create_app(frontend_dist=tmp_path / "dist", steps=steps, **paths))
    run = client.post("/api/pipeline/runs", json={"steps": list(UPDATE_ALL)}).json()
    assert run["title"] == "Обновить всё"
    assert run["id"].endswith("-all")
    assert wait_done(client, run["id"])["status"] == "ok"


def test_run_left_running_by_previous_app_is_interrupted(paths, tmp_path):
    paths["runs_dir"].mkdir()
    started = "2026-10-06T23:00:00+03:00"
    history = [
        {
            "id": "20261006-230000-all",
            "title": "Обновить всё",
            "kind": "data",
            "status": "running",
            "started_at": started,
            "finished_at": None,
            "stop_requested": False,
            "steps": [
                {"step": "reference", "status": "ok", "started_at": started, "finished_at": started, "exit_code": 0, "note": None},
                {"step": "futures_min", "status": "running", "started_at": started, "finished_at": None, "exit_code": None, "note": None},
                {"step": "continuous", "status": "pending", "started_at": None, "finished_at": None, "exit_code": None, "note": None},
            ],
        }
    ]
    (paths["runs_dir"] / "history.json").write_text(json.dumps(history))
    client = TestClient(create_app(frontend_dist=tmp_path / "dist", steps=STEPS, **paths))
    (run,) = client.get("/api/pipeline").json()["runs"]
    assert run["status"] == "interrupted"
    assert [step["status"] for step in run["steps"]] == ["ok", "error", "stopped"]
    assert client.post("/api/pipeline/runs", json={"steps": ["reference"]}).status_code == 200  # не мешает новым


def test_log_never_splits_a_russian_letter(client, paths):
    from app.api.runs import decode_whole_letters

    data = "минутки".encode()
    text, used = decode_whole_letters(data[:3])  # «м» и половина «и»
    assert (text, used) == ("м", 2)
    assert decode_whole_letters(data) == ("минутки", len(data))
