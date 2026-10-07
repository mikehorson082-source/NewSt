"""Шаги конвейера данных и порядок «Обновить всё».

Порядок важен: минутки берут список контрактов из дневных свечей, склейка —
минутки и реестр контрактов из справочников. FUTOI ни от чего не зависит.
Все загрузчики запускаются только с --update (докачка): режимы перезаписи
(--overwrite, --force, справочники без --update) — только руками из терминала.
"""

from __future__ import annotations

from .model import Step, StepKind


def script_step(
    id: str,
    title: str,
    description: str,
    script: str,
    *flags: str,
    kind: StepKind = "data",
    writes: tuple[str, ...] = (),
    needs: tuple[str, ...] = (),
) -> Step:
    args = (script, *flags)
    return Step(
        id=id,
        title=title,
        description=description,
        kind=kind,
        args=args,
        command="uv run python " + " ".join(args),
        writes=writes,
        needs=needs,
    )


STEPS: tuple[Step, ...] = (
    script_step(
        "reference",
        "Справочники",
        "Ставки ЦБ, курсы, спот, индексы, календари, реестр контрактов, ГО, курс клиринга",
        "src/data/download_reference.py",
        "--update",
        writes=("reference",),
    ),
    script_step(
        "futures_daily",
        "Дневные свечи",
        "Итоги торгов FORTS по каждому контракту с MOEX ISS",
        "src/data/download_futures_daily.py",
        "--update",
        writes=("raw/futures_daily",),
    ),
    script_step(
        "futures_min",
        "Минутные свечи",
        "1-минутные свечи FORTS; список контрактов — из дневных свечей",
        "src/data/download_futures_min.py",
        "--update",
        writes=("raw/futures",),
    ),
    script_step(
        "futoi",
        "FUTOI",
        "Открытые позиции физлиц и юрлиц: 5 минут и дни (Algopack, задержка две недели)",
        "src/data/download_futoi.py",
        "--update",
        writes=("raw/futoi",),
    ),
    script_step(
        "continuous",
        "Склейка",
        "Непрерывные 15-минутные ряды из минуток и реестра контрактов; сети не нужно",
        "src/data/build_continuous_futures.py",
        writes=("continuous_15m",),
        # Склейка читает минутки и реестр контрактов (reference/futures_contracts.parquet).
        # Справочники в needs не входят: шаг падает, если не ответил любой из 13
        # источников (ставки Китая, бенчмарки…), а сбой реестра контрактов ошибкой
        # не считается — справочник остаётся прежним. Склейка из-за ставок Китая не пропускается.
        needs=("futures_min",),
    ),
    Step(
        id="backtest_tests",
        title="Тесты бэктеста",
        description="Симулятор, издержки, спецификации рынков; данные не меняет",
        kind="check",
        args=("-m", "pytest", "src/backtest"),
        command="uv run pytest src/backtest",
    ),
)

STEP_BY_ID: dict[str, Step] = {step.id: step for step in STEPS}

# «Обновить всё»: все шаги, которые пишут данные, в порядке STEPS.
UPDATE_ALL: tuple[str, ...] = tuple(step.id for step in STEPS if step.kind == "data")
