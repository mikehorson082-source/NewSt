"""Чтение data/ для инвентаря: метаданные файлов, свежесть, срезы справочников.

Только читает. У parquet-файлов открывается «оглавление» (метаданные): число
строк и min/max колонки даты из статистики, сами данные не читаются. Целиком
читаются только маленькие справочники, колонка даты склеек (для пропусков)
и колонки secid/tradedate Algopack (по акциям). Всё кэшируется по времени
изменения файла: файл не менялся — повторно не читается.

Свежесть. Ожидаемая дата — последний завершённый торговый день до сегодня
(пн–пт и рабочие субботы из reference/ru_working_saturdays.json). Праздники
не учитываются: сразу после длинных праздников возможен ложный «отстаёт».
Отставание — сколько торговых дней после последней даты в данных уже прошло.

Пропуски склеек. Календарь берётся из данных (правило core/calendar.py):
торговые дни — даты TRADEDATE дневных свечей всех рынков. Сессии выходных
и праздников биржа приписывает следующему торговому дню, поэтому праздник,
в который торговалась только часть рынков, пропуском не считается. Пропуск —
торговый день внутри ряда, в который у рынка нет ни одного бара.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq

from core.config import Paths

from .catalog import (
    ALGOPACK_DATASETS,
    ALGOPACK_MARKETS,
    CONTINUOUS_ALL,
    CONTINUOUS_SERVICE,
    FUTOI_ALLOWED_LAG,
    REBUILD_HINT,
    REFERENCE_FILES,
    ReferenceSpec,
    RefLink,
)
from .model import SEVERITY, DatasetStatus, Fact, Status

MSK = ZoneInfo("Europe/Moscow")

# Сколько торговых дней сверх допустимого отставания набор «отстаёт»;
# дальше он «устарел».
STALE_AFTER = 5

# Сколько дат пропусков перечислять в заметке.
GAPS_IN_NOTE = 5


# ----------------------------------------------------------------- calendar
def is_trading_day(day: date, working_saturdays: frozenset[date]) -> bool:
    """Будний день или рабочая суббота. Праздники не учитываются."""
    return day.weekday() < 5 or day in working_saturdays


def expected_last_date(today: date, working_saturdays: frozenset[date] = frozenset()) -> date:
    """Последний завершённый торговый день: вчера или раньше."""
    day = today - timedelta(days=1)
    while not is_trading_day(day, working_saturdays):
        day -= timedelta(days=1)
    return day


def trading_days_behind(last: date, expected: date, working_saturdays: frozenset[date] = frozenset()) -> int:
    """Сколько торговых дней из (last, expected] нет в данных."""
    behind = 0
    day = last + timedelta(days=1)
    while day <= expected:
        behind += is_trading_day(day, working_saturdays)
        day += timedelta(days=1)
    return behind


def freshness(
    last: date, expected: date, allowed_lag: int, working_saturdays: frozenset[date] = frozenset()
) -> tuple[Status, int]:
    """Статус свежести и отставание в торговых днях."""
    lag = trading_days_behind(last, expected, working_saturdays)
    if lag <= allowed_lag:
        return "ok", lag
    if lag <= allowed_lag + STALE_AFTER:
        return "lagging", lag
    return "stale", lag


def worse(a: Status, b: Status) -> Status:
    """Худший из двух статусов."""
    return a if SEVERITY[a] >= SEVERITY[b] else b


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """1 день, 2 дня, 5 дней."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return forms[0]
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return forms[1]
    return forms[2]


def trading_days_text(n: int) -> str:
    return f"{n} {plural(n, ('торговый день', 'торговых дня', 'торговых дней'))}"


# ------------------------------------------------------------ file metadata
@dataclass(frozen=True)
class FileMeta:
    rows: int
    first: date | None
    last: date | None


def as_date(value: object) -> date | None:
    """Дата из значения статистики parquet: datetime, date или строка ISO."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"не дата: {value!r}")


@lru_cache(maxsize=16384)
def parquet_meta(path: str, column: str | None, mtime_ns: int, size: int) -> FileMeta:
    """Число строк и min/max колонки даты по статистике parquet.

    mtime_ns и size — ключ кэша: файл изменился — оглавление читается заново.
    """
    del mtime_ns, size
    meta = pq.read_metadata(path)
    if column is None or meta.num_rows == 0:
        return FileMeta(meta.num_rows, None, None)
    lows: list[date] = []
    highs: list[date] = []
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        chunk = next(
            (group.column(j) for j in range(group.num_columns) if group.column(j).path_in_schema == column),
            None,
        )
        if chunk is None:
            raise KeyError(f"нет колонки {column}")
        stats = chunk.statistics
        if stats is None or not stats.has_min_max:
            # Статистики нет (или группа строк целиком пустая) — читаем колонку.
            return scan_dates(path, column, meta.num_rows)
        low, high = as_date(stats.min), as_date(stats.max)
        if low is not None and high is not None:
            lows.append(low)
            highs.append(high)
    return FileMeta(meta.num_rows, min(lows, default=None), max(highs, default=None))


def scan_dates(path: str, column: str, rows: int) -> FileMeta:
    """Запасной путь: min/max колонки даты чтением самой колонки."""
    frame = pl.scan_parquet(path).select(pl.col(column).min().alias("lo"), pl.col(column).max().alias("hi")).collect()
    return FileMeta(rows, as_date(frame["lo"][0]), as_date(frame["hi"][0]))


@dataclass(frozen=True)
class Scan:
    """Сводка по набору файлов: строки, даты, размер и ошибки чтения."""

    files: int = 0
    rows: int = 0
    first: date | None = None
    last: date | None = None
    size: int = 0
    modified: datetime | None = None
    errors: tuple[str, ...] = ()


def modified_at(timestamp: float | None) -> datetime | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=MSK).replace(microsecond=0)


def scan_parquet(files: Sequence[Path], column: str | None) -> Scan:
    """Сводка по parquet-файлам. Нечитаемый файл попадает в errors, а не роняет инвентарь."""
    rows = size = 0
    firsts: list[date] = []
    lasts: list[date] = []
    mtimes: list[float] = []
    errors: list[str] = []
    for path in files:
        stat = path.stat()
        size += stat.st_size
        mtimes.append(stat.st_mtime)
        try:
            meta = parquet_meta(str(path), column, stat.st_mtime_ns, stat.st_size)
        except Exception as error:  # noqa: BLE001 — любой сбой чтения показываем в строке
            errors.append(f"{path.name}: {error}")
            continue
        rows += meta.rows
        if meta.first is not None and meta.last is not None:
            firsts.append(meta.first)
            lasts.append(meta.last)
    return Scan(
        files=len(files),
        rows=rows,
        first=min(firsts, default=None),
        last=max(lasts, default=None),
        size=size,
        modified=modified_at(max(mtimes, default=None)),
        errors=tuple(errors),
    )


def plain_scan(files: Sequence[Path]) -> Scan:
    """Сводка по файлам без чтения содержимого: число, размер, время изменения."""
    stats = [path.stat() for path in files]
    return Scan(
        files=len(files),
        size=sum(stat.st_size for stat in stats),
        modified=modified_at(max((stat.st_mtime for stat in stats), default=None)),
    )


def csv_scan(path: Path, column: str) -> Scan:
    stat = path.stat()
    try:
        values = pl.read_csv(path, columns=[column], try_parse_dates=True)[column]
    except Exception as error:  # noqa: BLE001
        return Scan(files=1, size=stat.st_size, modified=modified_at(stat.st_mtime), errors=(f"{path.name}: {error}",))
    return Scan(
        files=1,
        rows=len(values),
        first=as_date(values.min()),
        last=as_date(values.max()),
        size=stat.st_size,
        modified=modified_at(stat.st_mtime),
    )


def scan_json_dates(path: Path) -> Scan:
    """JSON-список дат (рабочие субботы)."""
    stat = path.stat()
    days = sorted(date.fromisoformat(day) for day in json.loads(path.read_text()))
    return Scan(
        files=1,
        rows=len(days),
        first=days[0] if days else None,
        last=days[-1] if days else None,
        size=stat.st_size,
        modified=modified_at(stat.st_mtime),
    )


@lru_cache(maxsize=64)
def read_frame(path: str, mtime_ns: int, size: int) -> pl.DataFrame:
    """Файл справочника целиком; mtime_ns и size — ключ кэша."""
    del mtime_ns, size
    return pl.read_parquet(path)


def frame_of(path: Path) -> pl.DataFrame:
    stat = path.stat()
    return read_frame(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def file_dates(path: str, column: str, mtime_ns: int, size: int) -> frozenset[date]:
    """Все даты колонки файла; mtime_ns и size — ключ кэша."""
    del mtime_ns, size
    return frozenset(pl.read_parquet(path, columns=[column])[column].unique().drop_nulls().to_list())


def dates_of(path: Path, column: str) -> frozenset[date]:
    stat = path.stat()
    return file_dates(str(path), column, stat.st_mtime_ns, stat.st_size)


def visible_files(folder: Path, pattern: str = "*") -> list[Path]:
    """Файлы папки без скрытых (.DS_Store и т. п.)."""
    return sorted(p for p in folder.glob(pattern) if p.is_file() and not p.name.startswith("."))


def visible_dirs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


# ------------------------------------------------------------------ context
@dataclass(frozen=True)
class Context:
    """Общее для всех проверок: где данные и от какой даты считать отставание."""

    data_root: Path
    expected: date
    working_saturdays: frozenset[date]

    @property
    def paths(self) -> Paths:
        return Paths(root=self.data_root)

    def rel(self, path: Path) -> str:
        return path.relative_to(self.data_root).as_posix()


def load_working_saturdays(reference_dir: Path) -> frozenset[date]:
    """Рабочие субботы из справочника; нет файла — пустое множество."""
    path = reference_dir / "ru_working_saturdays.json"
    if not path.is_file():
        return frozenset()
    return frozenset(date.fromisoformat(day) for day in json.loads(path.read_text()))


def join_notes(*notes: str | None) -> str | None:
    parts = [note for note in notes if note]
    return "; ".join(parts) if parts else None


def make_status(
    ctx: Context,
    *,
    kind: str,
    title: str,
    description: str,
    path: Path,
    scan: Scan,
    allowed_lag: int | None,
    facts: tuple[Fact, ...] = (),
    note: str | None = None,
    key: str | None = None,
) -> DatasetStatus:
    """Набор по сводке файлов и правилу свежести (allowed_lag None — не проверять).

    key — ключ среза («BR»): id становится «путь#ключ».
    """
    lag: int | None = None
    if scan.errors:
        status: Status = "error"
        shown = "; ".join(scan.errors[:3]) + (" …" if len(scan.errors) > 3 else "")
        note = join_notes(f"не читается: {shown}", note)
    elif scan.files == 0:
        status = "missing"
        note = note or "нет файлов"
    elif allowed_lag is None:
        status = "unchecked"
    elif scan.last is None:
        status = "missing"
        note = join_notes("в файлах нет дат", note)
    else:
        status, lag = freshness(scan.last, ctx.expected, allowed_lag, ctx.working_saturdays)
    rel = ctx.rel(path)
    return DatasetStatus(
        id=f"{rel}#{key}" if key else rel,
        kind=kind,
        title=title,
        description=description,
        path=rel,
        status=status,
        files=scan.files,
        rows=scan.rows if scan.files else None,
        first_date=scan.first,
        last_date=scan.last,
        size_bytes=scan.size,
        modified_at=scan.modified,
        lag_days=lag,
        note=note,
        facts=facts,
    )


def missing_folder(ctx: Context, folder: Path) -> DatasetStatus:
    return make_status(
        ctx,
        kind="other",
        title=folder.name,
        description="Папка из раскладки data/",
        path=folder,
        scan=Scan(),
        allowed_lag=None,
        note="нет папки",
    )


# --------------------------------------------------------------- reference
def reference_facts(path: Path, spec: ReferenceSpec) -> tuple[Fact, ...]:
    if spec.count_column is None or spec.count_label is None:
        return ()
    values = frame_of(path)[spec.count_column]
    return (Fact(spec.count_label, values.n_unique()),)


def scan_reference(ctx: Context) -> dict[str, DatasetStatus]:
    """Файлы справочников целиком: {имя файла: набор}. Файлы вне каталога — без проверки."""
    folder = ctx.paths.reference
    if not folder.is_dir():
        return {}
    rows: dict[str, DatasetStatus] = {}
    for spec in REFERENCE_FILES:
        path = folder / spec.file
        facts: tuple[Fact, ...] = ()
        if not path.is_file():
            scan = Scan()
        elif path.suffix == ".json":
            scan = scan_json_dates(path)
        else:
            scan = scan_parquet([path], spec.date_column)
            if not scan.errors:
                facts = reference_facts(path, spec)
        rows[spec.file] = make_status(
            ctx,
            kind="reference",
            title=path.stem,
            description=spec.description,
            path=path,
            scan=scan,
            allowed_lag=spec.allowed_lag,
            facts=facts,
            note=None if path.is_file() else "нет файла",
        )
    for path in visible_files(folder):
        if path.name in rows:
            continue
        scan = scan_parquet([path], None) if path.suffix == ".parquet" else plain_scan([path])
        rows[path.name] = make_status(
            ctx,
            kind="reference",
            title=path.stem,
            description="Нет в каталоге инвентаря",
            path=path,
            scan=scan,
            allowed_lag=None,
        )
    return rows


def reference_slice(ctx: Context, spec: ReferenceSpec, link: RefLink) -> DatasetStatus | None:
    """Срез справочника по ключу (asset_code == "BR"). Пустой срез — None."""
    path = ctx.paths.reference / spec.file
    key = str(link.value)
    try:
        part = frame_of(path).filter(pl.col(str(link.column)) == link.value)
    except Exception as error:  # noqa: BLE001 — сбой чтения показываем в строке инструмента
        scan = Scan(files=1, errors=(f"{path.name}: {error}",))
        return make_status(
            ctx,
            kind="reference",
            title=path.stem,
            description=spec.description,
            path=path,
            scan=scan,
            allowed_lag=spec.allowed_lag,
            key=key,
        )
    if part.is_empty():
        return None
    first = last = None
    if spec.date_column is not None:
        first, last = as_date(part[spec.date_column].min()), as_date(part[spec.date_column].max())
    facts: tuple[Fact, ...] = ()
    if link.fact_column is not None and link.fact_label is not None:
        facts = (Fact(link.fact_label, part[link.fact_column].n_unique()),)
    scan = Scan(files=1, rows=part.height, first=first, last=last, modified=modified_at(path.stat().st_mtime))
    return make_status(
        ctx,
        kind="reference",
        title=path.stem,
        description=f"{spec.description} · {key}",
        path=path,
        scan=scan,
        allowed_lag=spec.allowed_lag,
        facts=facts,
        key=key,
    )


def column_values(path: Path, column: str) -> set[str]:
    """Уникальные значения колонки справочника; нет файла — пустое множество."""
    if not path.is_file():
        return set()
    with contextlib.suppress(Exception):
        return set(frame_of(path)[column].drop_nulls().unique().to_list())
    return set()


# ----------------------------------------------------------------- futures
def scan_contract_dirs(ctx: Context, kind: str, folder: Path, column: str, title: str) -> dict[str, DatasetStatus]:
    """Папка на рынок, файл на контракт: {рынок: набор}. Минутные и дневные свечи."""
    if not folder.is_dir():
        return {}
    rows = {}
    for market in visible_dirs(folder):
        scan = scan_parquet(visible_files(market, "*.parquet"), column)
        rows[market.name] = make_status(
            ctx,
            kind=kind,
            title=title,
            description="файл на контракт",
            path=market,
            scan=scan,
            allowed_lag=0,
            facts=(Fact("Контрактов", scan.files),),
        )
    return rows


def scan_futoi(ctx: Context, kind: str, folder: Path, title: str) -> dict[str, DatasetStatus]:
    """Файл на тикер: {тикер: набор}. Задержка бесплатного API — норма."""
    if not folder.is_dir():
        return {}
    return {
        path.stem: make_status(
            ctx,
            kind=kind,
            title=title,
            description=f"тикер {path.stem} · задержка бесплатного API — 2 недели",
            path=path,
            scan=scan_parquet([path], "tradedate"),
            allowed_lag=FUTOI_ALLOWED_LAG,
        )
        for path in visible_files(folder, "*.parquet")
    }


# -------------------------------------------------------------- continuous
@dataclass(frozen=True)
class RollStats:
    rolls: int
    stale_adjustments: int


def roll_stats(path: Path) -> dict[str, RollStats]:
    """Число переходов и поправок по устаревшей общей минуте по рынкам."""
    if not path.is_file():
        return {}
    frame = (
        pl.read_csv(path, columns=["asset", "adjustment_is_stale"])
        .group_by("asset")
        .agg(pl.len().alias("rolls"), pl.col("adjustment_is_stale").sum().alias("stale"))
    )
    return {row["asset"]: RollStats(row["rolls"], row["stale"]) for row in frame.iter_rows(named=True)}


def official_trading_days(daily_root: Path) -> frozenset[date]:
    """Торговые дни биржи: даты TRADEDATE дневных свечей всех рынков.

    Нечитаемый файл пропускается: его ошибку покажет набор дневных свечей.
    """
    days: set[date] = set()
    if daily_root.is_dir():
        for path in sorted(daily_root.glob("*/*.parquet")):
            with contextlib.suppress(Exception):
                days |= dates_of(path, "TRADEDATE")
    return frozenset(days)


def scan_continuous(
    ctx: Context, raw_last: dict[str, date | None], trading_days: frozenset[date]
) -> tuple[dict[str, DatasetStatus], list[DatasetStatus]]:
    """Склейки: ({рынок: набор}, общие файлы). Свежесть, пропуски, роллы, отставание от минуток."""
    folder = ctx.data_root / "continuous_15m"
    if not folder.is_dir():
        return {}, []
    rolls = roll_stats(folder / "roll_schedule.csv")
    calendar = sorted(trading_days)

    markets: dict[str, DatasetStatus] = {}
    for path in visible_files(folder, "*_15m.parquet"):
        if path.name == CONTINUOUS_ALL:
            continue
        market = path.name.removesuffix("_15m.parquet")
        scan = scan_parquet([path], "date")
        try:
            days = dates_of(path, "date")
        except Exception:  # noqa: BLE001 — ошибку уже показывает scan
            days = frozenset()
        facts: list[Fact] = [Fact("Дней с данными", len(days))]
        notes: list[str] = []
        if scan.first is not None and scan.last is not None:
            gaps = [day for day in calendar if scan.first <= day <= scan.last and day not in days]
            facts.append(Fact("Пропусков", len(gaps)))
            if gaps:
                listed = ", ".join(day.isoformat() for day in gaps[:GAPS_IN_NOTE])
                notes.append(f"пропуски: {listed}" + (" …" if len(gaps) > GAPS_IN_NOTE else ""))
        if market in rolls:
            facts.append(Fact("Роллов", rolls[market].rolls))
            facts.append(Fact("Поправок по устаревшей минуте", rolls[market].stale_adjustments))
        row = make_status(
            ctx,
            kind="continuous",
            title="Склейка, 15 минут",
            description="по обороту, с NAV",
            path=path,
            scan=scan,
            allowed_lag=0,
            facts=tuple(facts),
            note=join_notes(*notes),
        )
        # Минутки свежее склейки — склейку пора пересобрать.
        source_last = raw_last.get(market)
        if row.status != "error" and scan.last is not None and source_last is not None:
            behind = trading_days_behind(scan.last, source_last, ctx.working_saturdays)
            if behind > 0:
                row = replace(
                    row,
                    status=worse(row.status, "lagging"),
                    note=join_notes(
                        f"отстаёт от минуток на {trading_days_text(behind)}: {REBUILD_HINT}",
                        row.note,
                    ),
                )
        markets[market] = row

    common: list[DatasetStatus] = []
    for name, (description, column) in CONTINUOUS_SERVICE.items():
        path = folder / name
        common.append(
            make_status(
                ctx,
                kind="continuous",
                title=path.stem,
                description=description,
                path=path,
                scan=scan_parquet([path], column) if path.is_file() else Scan(),
                allowed_lag=0,
                note=None if path.is_file() else "нет файла",
            )
        )
    schedule = folder / "roll_schedule.csv"
    common.append(
        make_status(
            ctx,
            kind="continuous",
            title=schedule.stem,
            description="Расписание переходов между контрактами",
            path=schedule,
            scan=csv_scan(schedule, "roll_date") if schedule.is_file() else Scan(),
            allowed_lag=None,
            facts=(Fact("Рынков", len(rolls)),) if rolls else (),
            note=None if schedule.is_file() else "нет файла",
        )
    )
    return markets, common


# ---------------------------------------------------------------- algopack
def scan_algopack(ctx: Context) -> list[DatasetStatus]:
    """Algopack целиком: папка на рынок и набор, файл на год. Архив — свежесть не проверяется."""
    folder = ctx.paths.raw / "algopack"
    if not folder.is_dir():
        return []
    rows = []
    for market in visible_dirs(folder):
        for dataset in visible_dirs(market):
            label = ALGOPACK_DATASETS.get(dataset.name, dataset.name)
            scan = scan_parquet(visible_files(dataset, "*.parquet"), "tradedate")
            rows.append(
                make_status(
                    ctx,
                    kind="algopack",
                    title=f"{market.name}/{dataset.name}",
                    description=f"{ALGOPACK_MARKETS.get(market.name, market.name)}: {label}",
                    path=dataset,
                    scan=scan,
                    allowed_lag=None,
                    facts=(Fact("Файлов", scan.files),),
                )
            )
    return rows


@lru_cache(maxsize=256)
def secid_ranges(path: str, mtime_ns: int, size: int) -> dict[str, tuple[date, date, int]]:
    """Файл Algopack по бумагам: {secid: (первая дата, последняя, строк)}; mtime_ns и size — ключ кэша."""
    del mtime_ns, size
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("secid").is_not_null())
        .group_by("secid")
        .agg(pl.col("tradedate").min().alias("lo"), pl.col("tradedate").max().alias("hi"), pl.len().alias("n"))
        .collect()
    )
    return {row["secid"]: (row["lo"], row["hi"], row["n"]) for row in frame.iter_rows(named=True)}


def file_secid_ranges(path: Path) -> dict[str, tuple[date, date, int]]:
    """Нечитаемый файл даёт пустой результат: его ошибку покажет набор Algopack целиком."""
    stat = path.stat()
    with contextlib.suppress(Exception):
        return secid_ranges(str(path), stat.st_mtime_ns, stat.st_size)
    return {}


def algopack_by_secid(dataset: Path) -> dict[str, Scan]:
    """Набор Algopack (папка с файлами по годам) по бумагам: {secid: сводка}."""
    merged: dict[str, Scan] = {}
    files = visible_files(dataset, "*.parquet")
    modified = modified_at(max((path.stat().st_mtime for path in files), default=None))
    for path in files:
        for secid, (lo, hi, n) in file_secid_ranges(path).items():
            old = merged.get(secid)
            merged[secid] = Scan(
                files=(old.files if old else 0) + 1,
                rows=(old.rows if old else 0) + n,
                first=min(lo, old.first) if old and old.first else lo,
                last=max(hi, old.last) if old and old.last else hi,
                modified=modified,
            )
    return merged


def algopack_slice(ctx: Context, dataset: Path, secid: str, scan: Scan) -> DatasetStatus:
    label = ALGOPACK_DATASETS.get(dataset.name, dataset.name)
    return make_status(
        ctx,
        kind="algopack",
        title=f"Algopack: {label}",
        description=f"{dataset.name} · {secid} · архив",
        path=dataset,
        scan=scan,
        allowed_lag=None,
        key=secid,
    )


# ------------------------------------------------------------------- other
KNOWN_TOP = {"raw", "continuous_15m", "reference"}
KNOWN_RAW = {"futures", "futures_daily", "futoi", "algopack"}


def scan_other(ctx: Context) -> list[DatasetStatus]:
    """Непустые папки, которых нет в каталоге: чтобы новые данные не оставались невидимыми."""
    candidates = [p for p in visible_dirs(ctx.data_root) if p.name not in KNOWN_TOP]
    if ctx.paths.raw.is_dir():
        candidates += [p for p in visible_dirs(ctx.paths.raw) if p.name not in KNOWN_RAW]
    rows = []
    for folder in candidates:
        files = [p for p in folder.rglob("*") if p.is_file() and not p.name.startswith(".")]
        if not files:
            continue
        rows.append(
            make_status(
                ctx,
                kind="other",
                title=folder.name,
                description="Нет в каталоге инвентаря",
                path=folder,
                scan=plain_scan(files),
                allowed_lag=None,
                facts=(Fact("Файлов", len(files)),),
            )
        )
    return rows
