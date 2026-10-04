"""Инвентарь данных: что лежит в data/, за какие даты и насколько оно свежее.

Только читает. У parquet-файлов открывается «оглавление» (метаданные): число
строк и min/max колонки даты из статистики, сами данные не читаются. Целиком
читаются только маленькие справочники и колонка даты склеек (для пропусков).

Функции возвращают данные, печатает только main(). Этот же отчёт отдаёт
веб-пульт (app/api): движок про веб ничего не знает.

Свежесть. Ожидаемая дата — последний завершённый торговый день до сегодня
(пн–пт и рабочие субботы из reference/ru_working_saturdays.json). Праздники
не учитываются: сразу после длинных праздников возможен ложный «отстаёт».
Отставание — сколько торговых дней после последней даты в данных уже прошло.

Пропуски склеек. Календарь берётся из данных (правило core/calendar.py):
торговые дни — даты TRADEDATE дневных свечей всех рынков. Сессии выходных
и праздников биржа приписывает следующему торговому дню, поэтому праздник,
в который торговалась только часть рынков, пропуском не считается. Пропуск —
торговый день внутри ряда, в который у рынка нет ни одного бара.

Запуск:
    uv run python -m data.inventory              # таблица по группам
    uv run python -m data.inventory --problems   # только то, что требует внимания
    uv run python -m data.inventory --json       # весь отчёт в JSON
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq

from core.config import DATA_DIR, Paths
from core.instruments import ASSETS, Cadence

MSK = ZoneInfo("Europe/Moscow")

# Сколько торговых дней сверх допустимого отставания набор «отстаёт»;
# дальше он «устарел».
STALE_AFTER = 5

# Сколько дат пропусков перечислять в заметке.
GAPS_IN_NOTE = 5

Status = Literal["ok", "lagging", "stale", "unchecked", "missing", "error"]

# Чем выше, тем хуже: так из двух статусов выбирается худший.
SEVERITY: dict[Status, int] = {
    "ok": 0,
    "unchecked": 0,
    "lagging": 1,
    "stale": 2,
    "missing": 3,
    "error": 3,
}


@dataclass(frozen=True)
class Fact:
    """Дополнительный факт о наборе: «Роллов: 53», «Рынков: 24»."""

    label: str
    value: int | float | str


@dataclass(frozen=True)
class DatasetStatus:
    """Одна строка инвентаря: файл справочника, папка рынка или ряд."""

    id: str  # стабильный ключ: путь относительно data/
    group: str
    title: str  # короткое техническое имя: файл, рынок, тикер
    description: str  # по-русски, что это
    path: str  # относительно data/
    status: Status
    files: int = 0
    rows: int | None = None
    first_date: date | None = None
    last_date: date | None = None
    size_bytes: int = 0
    modified_at: datetime | None = None
    lag_days: int | None = None  # торговых дней после last_date до ожидаемой даты
    note: str | None = None  # почему такой статус
    facts: tuple[Fact, ...] = ()


@dataclass(frozen=True)
class Group:
    """Группа строк инвентаря: справочники, минутки, склейки…"""

    id: str
    title: str
    description: str
    files_label: str | None = None  # как назвать число файлов в строке; None — не показывать


@dataclass(frozen=True)
class InventoryReport:
    """Весь инвентарь: группы, строки и сводка по статусам."""

    generated_at: datetime
    expected_date: date  # последний завершённый торговый день: от него считается отставание
    data_root: str
    groups: tuple[Group, ...]
    datasets: tuple[DatasetStatus, ...]
    summary: dict[Status, int]  # число строк по статусам


GROUPS: tuple[Group, ...] = (
    Group("reference", "Справочники и макро", "data/reference: ставки ГО, курсы, ставки, календари"),
    Group("futures_minutes", "Минутные свечи", "data/raw/futures: файл на контракт", "Контрактов"),
    Group("futures_daily", "Дневные свечи", "data/raw/futures_daily: файл на контракт", "Контрактов"),
    Group("continuous", "Непрерывные ряды, 15 минут", "data/continuous_15m: склейка по обороту и NAV"),
    Group("futoi_5min", "Открытый интерес FUTOI, 5 минут", "data/raw/futoi/5min: файл на тикер"),
    Group("futoi_daily", "Открытый интерес FUTOI, дни", "data/raw/futoi/daily: файл на тикер"),
    Group("algopack", "Algopack", "data/raw/algopack: архив, свежесть не проверяется", "Файлов"),
    Group("other", "Прочее", "Папки, которых нет в каталоге инвентаря", "Файлов"),
)


@dataclass(frozen=True)
class ReferenceSpec:
    """Как проверять файл справочника."""

    file: str
    description: str
    date_column: str | None  # колонка даты; None — без дат
    allowed_lag: int | None  # допустимое отставание в торговых днях; None — не проверять
    count_column: str | None = None  # что посчитать уникальным: рынки, валюты…
    count_label: str | None = None


REFERENCE_FILES: tuple[ReferenceSpec, ...] = (
    ReferenceSpec("margin_rates.parquet", "Минимальные ставки ГО mr1–mr3", "date", 0, "asset_code", "Рынков"),
    ReferenceSpec(
        "margin_snapshot.parquet",
        "Снимки фактического ГО (копятся при обновлении)",
        "snapshot_date",
        0,
        "snapshot_date",
        "Снимков",
    ),
    ReferenceSpec("rates_moex_clearing.parquet", "Курс клиринга MOEX USD/RUB", "date", 0),
    ReferenceSpec("rates_cbr_fx.parquet", "Официальные курсы ЦБ", "date", 0, "currency", "Валют"),
    ReferenceSpec("cbr_key_rate.parquet", "Ключевая ставка ЦБ", "date", 0),
    # SOFR публикуется утром следующего рабочего дня США — день отставания норма.
    ReferenceSpec("rates_usd.parquet", "Ставка SOFR", "date", 1),
    # Праздники Китая (Чуньцзе, Золотая неделя) — до 5 торговых дней без публикаций.
    ReferenceSpec("rates_china.parquet", "Ставки Китая", "date", 5, "series", "Рядов"),
    ReferenceSpec("spot_currency.parquet", "Спот валют MOEX", "date", 0, "secid", "Инструментов"),
    ReferenceSpec("indices_daily.parquet", "Индексы MOEX", "date", 0, "secid", "Индексов"),
    ReferenceSpec("benchmarks_world.parquet", "Мировые бенчмарки", "date", 0, "symbol", "Инструментов"),
    # Отчёт EIA выходит раз в неделю: неделя отставания норма.
    ReferenceSpec("eia_ng_storage.parquet", "Запасы газа в США (EIA)", "release_date", 5),
    ReferenceSpec(
        "futures_contracts.parquet", "Справочник контрактов", "first_trade_date", None, "asset_code", "Активов"
    ),
    ReferenceSpec("calendar_events.parquet", "Календарь событий", "date", None),
    ReferenceSpec("corp_dividends.parquet", "Дивиденды", "declared_date", None, "ticker", "Тикеров"),
    ReferenceSpec("corp_listing.parquet", "Листинг акций", None, None, "ticker", "Тикеров"),
    ReferenceSpec("corp_splits.parquet", "Сплиты акций", "date", None),
    ReferenceSpec("ru_working_saturdays.json", "Рабочие субботы (торги идут)", None, None),
)

ALGOPACK_DATASETS = {
    "tradestats": "сделки",
    "orderstats": "заявки",
    "obstats": "стакан",
}
ALGOPACK_MARKETS = {"eq": "Акции", "fo": "Фьючерсы", "fx": "Валюта"}

CONTINUOUS_ALL = "continuous_futures_15m.parquet"
CONTINUOUS_SERVICE = {
    CONTINUOUS_ALL: ("Все рынки одним файлом", "date"),
    "daily_contract_selection.parquet": ("Выбранный контракт по дням", "date"),
}
REBUILD_HINT = "uv run python src/data/build_continuous_futures.py"


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
    group: str,
    title: str,
    description: str,
    path: Path,
    scan: Scan,
    allowed_lag: int | None,
    facts: tuple[Fact, ...] = (),
    note: str | None = None,
) -> DatasetStatus:
    """Строка инвентаря по сводке файлов и правилу свежести (allowed_lag None — не проверять)."""
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
    return DatasetStatus(
        id=ctx.rel(path),
        group=group,
        title=title,
        description=description,
        path=ctx.rel(path),
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


def missing_folder(ctx: Context, group: str, folder: Path, description: str) -> DatasetStatus:
    return make_status(
        ctx,
        group=group,
        title=folder.name,
        description=description,
        path=folder,
        scan=Scan(),
        allowed_lag=None,
        note="нет папки",
    )


# --------------------------------------------------------------- reference
def reference_facts(path: Path, spec: ReferenceSpec) -> tuple[Fact, ...]:
    if spec.count_column is None or spec.count_label is None:
        return ()
    values = pl.read_parquet(path, columns=[spec.count_column])[spec.count_column]
    return (Fact(spec.count_label, values.n_unique()),)


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


def scan_reference(ctx: Context) -> list[DatasetStatus]:
    """Справочники по каталогу REFERENCE_FILES; файлы вне каталога — без проверки."""
    group = "reference"
    folder = ctx.paths.reference
    if not folder.is_dir():
        return [missing_folder(ctx, group, folder, "Справочники")]
    rows: list[DatasetStatus] = []
    known = {spec.file for spec in REFERENCE_FILES}
    for spec in REFERENCE_FILES:
        path = folder / spec.file
        title = Path(spec.file).stem
        if not path.is_file():
            rows.append(
                make_status(
                    ctx,
                    group=group,
                    title=title,
                    description=spec.description,
                    path=path,
                    scan=Scan(),
                    allowed_lag=spec.allowed_lag,
                    note="нет файла",
                )
            )
            continue
        facts: tuple[Fact, ...] = ()
        if path.suffix == ".json":
            scan = scan_json_dates(path)
        else:
            scan = scan_parquet([path], spec.date_column)
            if not scan.errors:
                facts = reference_facts(path, spec)
        rows.append(
            make_status(
                ctx,
                group=group,
                title=title,
                description=spec.description,
                path=path,
                scan=scan,
                allowed_lag=spec.allowed_lag,
                facts=facts,
            )
        )
    for path in visible_files(folder):
        if path.name in known:
            continue
        scan = scan_parquet([path], None) if path.suffix == ".parquet" else plain_scan([path])
        rows.append(
            make_status(
                ctx,
                group=group,
                title=path.stem,
                description="Нет в каталоге инвентаря",
                path=path,
                scan=scan,
                allowed_lag=None,
            )
        )
    return rows


# ----------------------------------------------------------------- futures
def market_name(name: str) -> str:
    """Название рынка из реестра core/instruments.py."""
    spec = ASSETS.get(name)
    return spec.short_name if spec else name


def scan_contract_dirs(ctx: Context, group: str, folder: Path, column: str) -> list[DatasetStatus]:
    """Папка на рынок, файл на контракт: минутные и дневные свечи."""
    if not folder.is_dir():
        return [missing_folder(ctx, group, folder, "Папка рынков")]
    return [
        make_status(
            ctx,
            group=group,
            title=market.name,
            description=market_name(market.name),
            path=market,
            scan=scan_parquet(visible_files(market, "*.parquet"), column),
            allowed_lag=0,
        )
        for market in visible_dirs(folder)
    ]


def futoi_names() -> dict[str, str]:
    """Тикер FUTOI → название рынка: по префиксу контрактов и тикерам вечных фьючерсов."""
    names: dict[str, str] = {}
    for spec in ASSETS.values():
        if spec.cadence == Cadence.PERPETUAL:
            names.update({ticker: f"Вечный фьючерс {ticker}" for ticker in spec.fixed_tickers})
            continue
        for prefix in (spec.ticker_prefix, *spec.extra_prefixes):
            names.setdefault(prefix, spec.short_name)
    return names


def scan_futoi(ctx: Context, group: str, folder: Path) -> list[DatasetStatus]:
    """Файл на тикер: FUTOI 5 минут и по дням."""
    if not folder.is_dir():
        return [missing_folder(ctx, group, folder, "FUTOI")]
    names = futoi_names()
    return [
        make_status(
            ctx,
            group=group,
            title=path.stem,
            description=names.get(path.stem, path.stem),
            path=path,
            scan=scan_parquet([path], "tradedate"),
            allowed_lag=0,
        )
        for path in visible_files(folder, "*.parquet")
    ]


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


@lru_cache(maxsize=4096)
def file_dates(path: str, column: str, mtime_ns: int, size: int) -> frozenset[date]:
    """Все даты колонки файла; mtime_ns и size — ключ кэша."""
    del mtime_ns, size
    return frozenset(pl.read_parquet(path, columns=[column])[column].unique().drop_nulls().to_list())


def dates_of(path: Path, column: str) -> frozenset[date]:
    stat = path.stat()
    return file_dates(str(path), column, stat.st_mtime_ns, stat.st_size)


def official_trading_days(daily_root: Path) -> frozenset[date]:
    """Торговые дни биржи: даты TRADEDATE дневных свечей всех рынков.

    Нечитаемый файл пропускается: его ошибку покажет строка дневных свечей.
    """
    days: set[date] = set()
    if daily_root.is_dir():
        for path in sorted(daily_root.glob("*/*.parquet")):
            with contextlib.suppress(Exception):
                days |= dates_of(path, "TRADEDATE")
    return frozenset(days)


def scan_continuous(
    ctx: Context, raw_last: dict[str, date | None], trading_days: frozenset[date]
) -> list[DatasetStatus]:
    """Склейки по рынкам: свежесть, пропуски, роллы и отставание от минуток в raw."""
    group = "continuous"
    folder = ctx.data_root / "continuous_15m"
    if not folder.is_dir():
        return [missing_folder(ctx, group, folder, "Склейки")]
    rolls = roll_stats(folder / "roll_schedule.csv")
    market_files = [p for p in visible_files(folder, "*_15m.parquet") if p.name != CONTINUOUS_ALL]

    days_by_market: dict[str, frozenset[date]] = {}
    for path in market_files:
        try:
            days_by_market[path.name] = dates_of(path, "date")
        except Exception:  # noqa: BLE001 — ошибку покажет scan_parquet в строке рынка
            days_by_market[path.name] = frozenset()
    calendar = sorted(trading_days)

    rows: list[DatasetStatus] = []
    for path in market_files:
        market = path.name.removesuffix("_15m.parquet")
        scan = scan_parquet([path], "date")
        days = days_by_market[path.name]
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
            group=group,
            title=market,
            description=market_name(market),
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
        rows.append(row)

    for name, (description, column) in CONTINUOUS_SERVICE.items():
        path = folder / name
        scan = scan_parquet([path], column) if path.is_file() else Scan()
        rows.append(
            make_status(
                ctx,
                group=group,
                title=path.stem,
                description=description,
                path=path,
                scan=scan,
                allowed_lag=0,
                note=None if path.is_file() else "нет файла",
            )
        )
    schedule = folder / "roll_schedule.csv"
    rows.append(
        make_status(
            ctx,
            group=group,
            title=schedule.stem,
            description="Расписание переходов между контрактами",
            path=schedule,
            scan=csv_scan(schedule, "roll_date") if schedule.is_file() else Scan(),
            allowed_lag=None,
            facts=(Fact("Рынков", len(rolls)),) if rolls else (),
            note=None if schedule.is_file() else "нет файла",
        )
    )
    return rows


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


# ---------------------------------------------------------------- algopack
def scan_algopack(ctx: Context) -> list[DatasetStatus]:
    """Algopack: папка на рынок и набор, файл на год. Архив — свежесть не проверяется."""
    group = "algopack"
    folder = ctx.paths.raw / "algopack"
    if not folder.is_dir():
        return [missing_folder(ctx, group, folder, "Algopack")]
    rows = []
    for market in visible_dirs(folder):
        for dataset in visible_dirs(market):
            label = ALGOPACK_DATASETS.get(dataset.name, dataset.name)
            rows.append(
                make_status(
                    ctx,
                    group=group,
                    title=f"{market.name}/{dataset.name}",
                    description=f"{ALGOPACK_MARKETS.get(market.name, market.name)}: {label}",
                    path=dataset,
                    scan=scan_parquet(visible_files(dataset, "*.parquet"), "tradedate"),
                    allowed_lag=None,
                )
            )
    return rows


# ------------------------------------------------------------------- other
KNOWN_TOP = {"raw", "continuous_15m", "reference"}
KNOWN_RAW = {"futures", "futures_daily", "futoi", "algopack"}


def plain_scan(files: Sequence[Path]) -> Scan:
    stats = [path.stat() for path in files]
    return Scan(
        files=len(files),
        size=sum(stat.st_size for stat in stats),
        modified=modified_at(max((stat.st_mtime for stat in stats), default=None)),
    )


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
                group="other",
                title=folder.name,
                description="Нет в каталоге инвентаря",
                path=folder,
                scan=plain_scan(files),
                allowed_lag=None,
            )
        )
    return rows


# ------------------------------------------------------------------- report
def build_inventory(data_root: Path = DATA_DIR, today: date | None = None) -> InventoryReport:
    """Инвентарь data/: группы, строки и сводка по статусам. Ничего не пишет."""
    data_root = data_root.resolve()
    now = datetime.now(MSK).replace(microsecond=0)
    working_saturdays = load_working_saturdays(Paths(root=data_root).reference)
    ctx = Context(
        data_root=data_root,
        expected=expected_last_date(today or now.date(), working_saturdays),
        working_saturdays=working_saturdays,
    )
    paths = ctx.paths
    minutes = scan_contract_dirs(ctx, "futures_minutes", paths.raw_futures_min, "timestamp")
    datasets = [
        *scan_reference(ctx),
        *minutes,
        *scan_contract_dirs(ctx, "futures_daily", paths.raw_futures_daily, "TRADEDATE"),
        *scan_continuous(
            ctx,
            {row.title: row.last_date for row in minutes},
            official_trading_days(paths.raw_futures_daily),
        ),
        *scan_futoi(ctx, "futoi_5min", paths.raw_futoi_5min),
        *scan_futoi(ctx, "futoi_daily", paths.raw_futoi_daily),
        *scan_algopack(ctx),
        *scan_other(ctx),
    ]
    used = {row.group for row in datasets}
    summary = {status: 0 for status in SEVERITY}
    for row in datasets:
        summary[row.status] += 1
    return InventoryReport(
        generated_at=now,
        expected_date=ctx.expected,
        data_root=str(data_root),
        groups=tuple(group for group in GROUPS if group.id in used),
        datasets=tuple(datasets),
        summary=summary,
    )


# ---------------------------------------------------------------------- CLI
STATUS_TEXT = {
    "ok": "ok",
    "lagging": "ОТСТАЁТ",
    "stale": "УСТАРЕЛ",
    "unchecked": "·",
    "missing": "НЕТ",
    "error": "ОШИБКА",
}


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def number(value: int | None) -> str:
    return "" if value is None else f"{value:,}".replace(",", " ")


def format_table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows]


WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


def facts_text(row: DatasetStatus, files_label: str | None) -> str:
    facts = [(files_label, row.files)] if files_label else []
    facts += [(fact.label, fact.value) for fact in row.facts]
    return " · ".join(
        f"{label.lower()} {number(value) if isinstance(value, int) else value}" for label, value in facts
    )


def print_report(report: InventoryReport, problems_only: bool = False) -> None:
    expected = report.expected_date
    print(f"Инвентарь {report.data_root}")
    print(
        f"Ожидаемая дата: {WEEKDAYS[expected.weekday()]} {expected:%d.%m.%Y}"
        f" · проверено {report.generated_at:%d.%m.%Y %H:%M}"
    )
    for group in report.groups:
        rows = [row for row in report.datasets if row.group == group.id]
        if problems_only:
            rows = [row for row in rows if SEVERITY[row.status] > 0]
        if not rows:
            continue
        print(f"\n{group.title} — {group.description}")
        table = [["статус", "набор", "с", "по", "отстаёт", "строк", "размер", "факты", "заметка"]]
        for row in rows:
            table.append(
                [
                    STATUS_TEXT[row.status],
                    row.title,
                    row.first_date.isoformat() if row.first_date else "",
                    row.last_date.isoformat() if row.last_date else "",
                    "" if row.lag_days is None else str(row.lag_days),
                    number(row.rows),
                    human_size(row.size_bytes),
                    facts_text(row, group.files_label),
                    row.note or "",
                ]
            )
        print("\n".join(format_table(table)))
    counts = ", ".join(f"{STATUS_TEXT[status]}: {count}" for status, count in report.summary.items() if count)
    print(f"\nИтого строк: {len(report.datasets)} ({counts})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Инвентарь data/: что лежит, за какие даты и насколько свежее")
    parser.add_argument("--problems", action="store_true", help="показать только то, что требует внимания")
    parser.add_argument("--json", action="store_true", help="вывести весь отчёт в JSON")
    args = parser.parse_args()
    report = build_inventory()
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str))
        return
    print_report(report, problems_only=args.problems)


if __name__ == "__main__":
    main()
