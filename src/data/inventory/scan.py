"""Чтение data/ для инвентаря: метаданные файлов, свежесть, срезы справочников.

Только читает. У parquet-файлов открывается «оглавление» (метаданные): число
строк и min/max колонки даты из статистики. Колонка даты читается ещё и для
счёта строк по месяцам (график на странице). Целиком читаются только маленькие
справочники и колонки secid/tradedate Algopack (по акциям). Всё кэшируется по
времени изменения файла: файл не менялся — повторно не читается.

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
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq

from core.config import Paths
from core.instruments import ASSETS

from .catalog import (
    ALGOPACK_DATASETS,
    ALGOPACK_MARKETS,
    CONTINUOUS_ALL,
    CONTINUOUS_SERVICE,
    CONTRACTS_HINT,
    REBUILD_HINT,
    REFERENCE_FILES,
    ReferenceSpec,
    RefLink,
    futoi_market,
)
from .model import SEVERITY, DatasetStatus, Fact, Status
from .rules import CONTRACTS_FILE, Rule, Rules

MSK = ZoneInfo("Europe/Moscow")

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
    last: date, expected: date, allowed_lag: int, stale_after: int, working_saturdays: frozenset[date] = frozenset()
) -> tuple[Status, int]:
    """Статус свежести и отставание в торговых днях.

    stale_after — сколько торговых дней сверх допустимого набор «отстаёт»; дальше «устарел».
    """
    lag = trading_days_behind(last, expected, working_saturdays)
    if lag <= allowed_lag:
        return "ok", lag
    if lag <= allowed_lag + stale_after:
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


def rule_text(rule: Rule) -> str:
    """Правило по-русски для кабинета: норма и почему."""
    if rule.min_active_contracts is not None:
        n = rule.min_active_contracts
        norm = f"Норма — у рынка не меньше {n} {plural(n, ('торгуемого контракта', 'торгуемых контрактов', 'торгуемых контрактов'))}."
    elif rule.allowed_lag == 0:
        norm = "Норма — данные по последний завершённый торговый день."
    elif rule.allowed_lag is not None:
        n = rule.allowed_lag
        norm = f"Норма — отставание до {n} {plural(n, ('торгового дня', 'торговых дней', 'торговых дней'))}."
    else:
        norm = "Свежесть не проверяется."
    return f"{norm} {rule.why}"


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


# ------------------------------------------------------------------- months
# Строки по месяцам: ((первое число месяца, строк), …) по возрастанию месяцев.
Months = tuple[tuple[date, int], ...]


def months_of(values: pl.Series) -> Months:
    """Строки по месяцам из колонки дат (date, datetime или ISO-строки)."""
    frame = (
        values.drop_nulls()
        .cast(pl.Date)
        .dt.truncate("1mo")
        .to_frame("month")
        .group_by("month")
        .len()
        .sort("month")
    )
    return tuple(zip(frame["month"].to_list(), frame["len"].to_list(), strict=True))


def merge_months(parts: Iterable[Months]) -> Months:
    total: Counter[date] = Counter()
    for part in parts:
        for month, rows in part:
            total[month] += rows
    return tuple(sorted(total.items()))


@lru_cache(maxsize=16384)
def file_months(path: str, column: str, mtime_ns: int, size: int) -> Months:
    """Строки файла по месяцам колонки даты; mtime_ns и size — ключ кэша."""
    del mtime_ns, size
    return months_of(pl.read_parquet(path, columns=[column])[column])


def next_month(month: date) -> date:
    return date(month.year + month.month // 12, month.month % 12 + 1, 1)


def monthly_rows(months: Months, first: date | None, last: date | None) -> tuple[int, ...]:
    """Строк по месяцам подряд от месяца first до месяца last; месяц без данных — 0."""
    if not months or first is None or last is None:
        return ()
    counts = dict(months)
    out: list[int] = []
    month = first.replace(day=1)
    while month <= last:
        out.append(counts.get(month, 0))
        month = next_month(month)
    return tuple(out)


# ------------------------------------------------------------ parquet meta
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
    months: Months = ()  # пусто — по месяцам не считали


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
    months: list[Months] | None = [] if column is not None else None
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
        if months is not None and column is not None:
            try:
                months.append(file_months(str(path), column, stat.st_mtime_ns, stat.st_size))
            except Exception:  # noqa: BLE001 — неполный график хуже, чем никакого
                months = None
    return Scan(
        files=len(files),
        rows=rows,
        first=min(firsts, default=None),
        last=max(lasts, default=None),
        size=size,
        modified=modified_at(max(mtimes, default=None)),
        errors=tuple(errors),
        months=merge_months(months) if months is not None else (),
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
        months=months_of(values),
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
        months=months_of(pl.Series(days, dtype=pl.Date)),
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
    rules: Rules

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
    rule: Rule | None,
    facts: tuple[Fact, ...] = (),
    note: str | None = None,
    key: str | None = None,
) -> DatasetStatus:
    """Набор по сводке файлов и правилу свежести (нет правила или allowed_lag None — не проверять).

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
    elif rule is None or rule.allowed_lag is None:
        status = "unchecked"
    elif scan.last is None:
        status = "missing"
        note = join_notes("в файлах нет дат", note)
    else:
        status, lag = freshness(scan.last, ctx.expected, rule.allowed_lag, ctx.rules.stale_after, ctx.working_saturdays)
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
        rule=rule_text(rule) if rule is not None else None,
        facts=facts,
        monthly=monthly_rows(scan.months, scan.first, scan.last),
    )


def missing_folder(ctx: Context, folder: Path) -> DatasetStatus:
    return make_status(
        ctx,
        kind="other",
        title=folder.name,
        description="Папка из раскладки data/",
        path=folder,
        scan=Scan(),
        rule=None,
        note="нет папки",
    )


# --------------------------------------------------------------- reference
def reference_facts(path: Path, spec: ReferenceSpec) -> tuple[Fact, ...]:
    if spec.count_column is None or spec.count_label is None:
        return ()
    values = frame_of(path)[spec.count_column]
    return (Fact(spec.count_label, values.n_unique()),)


def active_contracts(frame: pl.DataFrame, expected: date) -> dict[str, int]:
    """Торгуемых контрактов по asset_code: последний день торгов позже ожидаемой даты."""
    active = frame.filter(pl.col("last_trade_date") > expected).group_by("asset_code").len()
    return dict(active.iter_rows())


def check_contracts(ctx: Context, row: DatasetStatus, frame: pl.DataFrame, market: str | None = None) -> DatasetStatus:
    """Хватает ли торгуемых контрактов по правилу futures_contracts.parquet.

    market None — все рынки реестра (файл целиком), иначе один рынок (срез карточки).
    Пример: в справочнике у Si остался один SIZ6 при норме 2 — «отстаёт»:
    биржа уже открыла SIH7, а справочник о нём не знает. Ни одного — «устарел».
    """
    if row.status != "unchecked":  # нет файла, не читается
        return row
    try:
        counts = active_contracts(frame, ctx.expected)
    except Exception as error:  # noqa: BLE001 — нет колонки last_trade_date: показываем в строке
        return replace(row, status="error", note=join_notes(f"не посчитать торгуемые контракты: {error}", row.note))
    status: Status = "ok"
    short = []
    for name in ASSETS if market is None else [market]:
        rule = ctx.rules.rule(CONTRACTS_FILE, name)
        if rule is None or rule.min_active_contracts is None:
            continue
        count = counts.get(ASSETS[name].asset_code, 0)
        if count < rule.min_active_contracts:
            status = worse(status, "lagging" if count else "stale")
            short.append(f"{name} — {count} из {rule.min_active_contracts}")
    note = f"мало торгуемых контрактов: {', '.join(short)}; обновите: {CONTRACTS_HINT}" if short else None
    facts = row.facts
    if market is not None:
        facts += (Fact("Торгуемых", counts.get(ASSETS[market].asset_code, 0)),)
    return replace(row, status=status, note=join_notes(note, row.note), facts=facts)


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
            rule=ctx.rules.rule(spec.file),
            facts=facts,
            note=None if path.is_file() else "нет файла",
        )
        if spec.file == CONTRACTS_FILE and rows[spec.file].status == "unchecked":
            rows[spec.file] = check_contracts(ctx, rows[spec.file], frame_of(path))
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
            rule=None,
        )
    return rows


def reference_slice(
    ctx: Context, spec: ReferenceSpec, link: RefLink, market: str | None = None
) -> DatasetStatus | None:
    """Срез справочника по ключу (asset_code == "BR"). Пустой срез — None.

    market — рынок карточки: по нему берётся исключение из rules.json.
    """
    path = ctx.paths.reference / spec.file
    key = str(link.value)
    rule = ctx.rules.rule(spec.file, market)
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
            rule=rule,
            key=key,
        )
    if part.is_empty():
        return None
    first = last = None
    months: Months = ()
    if spec.date_column is not None:
        first, last = as_date(part[spec.date_column].min()), as_date(part[spec.date_column].max())
        months = months_of(part[spec.date_column])
    facts: tuple[Fact, ...] = ()
    if link.fact_column is not None and link.fact_label is not None:
        facts = (Fact(link.fact_label, part[link.fact_column].n_unique()),)
    scan = Scan(
        files=1, rows=part.height, first=first, last=last, modified=modified_at(path.stat().st_mtime), months=months
    )
    row = make_status(
        ctx,
        kind="reference",
        title=path.stem,
        description=f"{spec.description} · {key}",
        path=path,
        scan=scan,
        rule=rule,
        facts=facts,
        key=key,
    )
    if spec.file == CONTRACTS_FILE and market in ASSETS:
        row = check_contracts(ctx, row, part, market)
    return row


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
            rule=ctx.rules.rule(kind, market.name),
            facts=(Fact("Контрактов", scan.files),),
        )
    return rows


def scan_futoi(ctx: Context, kind: str, folder: Path, title: str) -> dict[str, DatasetStatus]:
    """Файл на тикер: {тикер: набор}. Правило — по рынку тикера (Si → si)."""
    if not folder.is_dir():
        return {}
    return {
        path.stem: make_status(
            ctx,
            kind=kind,
            title=title,
            description=f"тикер {path.stem}",
            path=path,
            scan=scan_parquet([path], "tradedate"),
            rule=ctx.rules.rule(kind, futoi_market(path.stem)),
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
        rule = ctx.rules.rule("continuous", market)
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
            rule=rule,
            facts=tuple(facts),
            note=join_notes(*notes),
        )
        # Минутки свежее склейки больше чем на день — склейку пора пересобрать.
        # На один день склейка позади всегда: последний день минуток она не берёт.
        source_last = raw_last.get(market)
        allowed = rule.allowed_lag if rule is not None else None
        if row.status != "error" and allowed is not None and scan.last is not None and source_last is not None:
            behind = trading_days_behind(scan.last, source_last, ctx.working_saturdays)
            if behind > allowed:
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
                rule=ctx.rules.rule("continuous"),
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
            rule=None,
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
                    rule=ctx.rules.rule("algopack"),
                    facts=(Fact("Файлов", scan.files),),
                )
            )
    return rows


@dataclass(frozen=True)
class SecidRange:
    first: date
    last: date
    rows: int
    months: Months


@lru_cache(maxsize=256)
def secid_ranges(path: str, mtime_ns: int, size: int) -> dict[str, SecidRange]:
    """Файл Algopack по бумагам: {secid: даты, строки, строки по месяцам}; mtime_ns и size — ключ кэша."""
    del mtime_ns, size
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("secid").is_not_null())
        .group_by("secid", pl.col("tradedate").cast(pl.Date).dt.truncate("1mo").alias("month"))
        .agg(pl.col("tradedate").min().alias("lo"), pl.col("tradedate").max().alias("hi"), pl.len().alias("n"))
        .sort("secid", "month")
        .collect()
    )
    out: dict[str, SecidRange] = {}
    for part in frame.partition_by("secid", maintain_order=True):
        out[part["secid"][0]] = SecidRange(
            first=as_date(part["lo"].min()) or date.min,
            last=as_date(part["hi"].max()) or date.min,
            rows=int(part["n"].sum()),
            months=tuple(zip(part["month"].to_list(), part["n"].to_list(), strict=True)),
        )
    return out


def file_secid_ranges(path: Path) -> dict[str, SecidRange]:
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
        for secid, part in file_secid_ranges(path).items():
            old = merged.get(secid)
            merged[secid] = Scan(
                files=(old.files if old else 0) + 1,
                rows=(old.rows if old else 0) + part.rows,
                first=min(part.first, old.first) if old and old.first else part.first,
                last=max(part.last, old.last) if old and old.last else part.last,
                modified=modified,
                months=merge_months([old.months, part.months]) if old else part.months,
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
        rule=ctx.rules.rule("algopack"),
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
                rule=None,
                facts=(Fact("Файлов", len(files)),),
            )
        )
    return rows
