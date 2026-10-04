"""Сборка отчёта: Фьючерсы и Акции по инструментам, плюс «Общее».

Карточка фьючерса: минутки, дневки, склейка, FUTOI и справочники рынка
(срезы ставок ГО, снимков ГО и контрактов по asset_code, курс клиринга у
долларовых, спот и ставки у валют — см. catalog.market_links).
Карточка акции: Algopack (архив) и корпоративные события по тикеру.
«Общее»: справочники целиком, общие файлы склейки, Algopack целиком,
FUTOI без инструмента и папки вне каталога.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

from core.config import DATA_DIR, Paths
from core.instruments import ASSETS

from .catalog import (
    REFERENCE_BY_FILE,
    SECTOR_TITLES,
    UNDERLYING_STOCKS,
    RefLink,
    futoi_market,
    market_links,
    stock_links,
)
from .model import SEVERITY, DatasetStatus, Instrument, InventoryReport, Section
from .scan import (
    MSK,
    Context,
    algopack_by_secid,
    algopack_slice,
    column_values,
    expected_last_date,
    frame_of,
    load_working_saturdays,
    missing_folder,
    official_trading_days,
    reference_slice,
    scan_algopack,
    scan_continuous,
    scan_contract_dirs,
    scan_futoi,
    scan_other,
    scan_reference,
)


class Collector:
    """Собирает наборы без повторов: инструменты ссылаются на них по id."""

    def __init__(self) -> None:
        self.datasets: dict[str, DatasetStatus] = {}

    def instrument(
        self, id: str, title: str, subtitle: str, rows: Iterable[DatasetStatus | None]
    ) -> Instrument | None:
        present = [row for row in rows if row is not None]
        if not present:
            return None
        for row in present:
            self.datasets.setdefault(row.id, row)
        return Instrument(id=id, title=title, subtitle=subtitle, datasets=tuple(dict.fromkeys(r.id for r in present)))


def linked(ctx: Context, reference: dict[str, DatasetStatus], link: RefLink) -> DatasetStatus | None:
    """Справочник для карточки: файл целиком или срез.

    Нет файла — None: отсутствие показывается один раз в «Общее», а не в
    каждой из сотен карточек.
    """
    spec = REFERENCE_BY_FILE.get(link.file)
    path = ctx.paths.reference / link.file
    if not path.is_file():
        return None
    if link.column is None or spec is None:
        return reference.get(link.file)
    return reference_slice(ctx, spec, link)


def section(id: str, title: str, description: str, instruments: Iterable[Instrument | None]) -> Section:
    return Section(id, title, description, tuple(item for item in instruments if item is not None))


def build_inventory(data_root: Path = DATA_DIR, today: date | None = None) -> InventoryReport:
    """Инвентарь data/ по инструментам. Ничего не пишет."""
    data_root = data_root.resolve()
    now = datetime.now(MSK).replace(microsecond=0)
    working_saturdays = load_working_saturdays(Paths(root=data_root).reference)
    ctx = Context(
        data_root=data_root,
        expected=expected_last_date(today or now.date(), working_saturdays),
        working_saturdays=working_saturdays,
    )
    paths = ctx.paths
    reference = scan_reference(ctx)
    minutes = scan_contract_dirs(ctx, "futures_minutes", paths.raw_futures_min, "timestamp", "Минутные свечи")
    daily = scan_contract_dirs(ctx, "futures_daily", paths.raw_futures_daily, "TRADEDATE", "Дневные свечи")
    continuous, continuous_common = scan_continuous(
        ctx,
        {market: row.last_date for market, row in minutes.items()},
        official_trading_days(paths.raw_futures_daily),
    )
    futoi_5min = scan_futoi(ctx, "futoi_5min", paths.raw_futoi_5min, "FUTOI, 5 минут")
    futoi_daily = scan_futoi(ctx, "futoi_daily", paths.raw_futoi_daily, "FUTOI, дни")

    # Тикеры FUTOI по рынкам (Si → si, USDRUBF → perpetual); без рынка — в «Общее».
    futoi_by_market: dict[str | None, list[str]] = {}
    for ticker in sorted(set(futoi_5min) | set(futoi_daily)):
        futoi_by_market.setdefault(futoi_market(ticker), []).append(ticker)

    # Algopack по акциям: {папка набора: {secid: сводка}}.
    algopack_eq = paths.raw / "algopack" / "eq"
    by_dataset = {
        dataset: algopack_by_secid(dataset)
        for dataset in (sorted(p for p in algopack_eq.iterdir() if p.is_dir()) if algopack_eq.is_dir() else [])
    }

    def stock_rows(ticker: str) -> list[DatasetStatus | None]:
        rows: list[DatasetStatus | None] = [
            algopack_slice(ctx, dataset, ticker, secids[ticker])
            for dataset, secids in by_dataset.items()
            if ticker in secids
        ]
        return rows + [linked(ctx, reference, link) for link in stock_links(ticker)]

    collector = Collector()

    # ------------------------------------------------------------- фьючерсы
    with_data = set(minutes) | set(daily) | set(continuous) | {m for m in futoi_by_market if m}
    markets = [name for name in ASSETS if name in with_data] + sorted(with_data - set(ASSETS))
    futures = []
    for name in markets:
        rows: list[DatasetStatus | None] = [minutes.get(name), daily.get(name), continuous.get(name)]
        for ticker in futoi_by_market.get(name, []):
            rows += [futoi_5min.get(ticker), futoi_daily.get(ticker)]
        rows += [linked(ctx, reference, link) for link in market_links(name)]
        for ticker in UNDERLYING_STOCKS.get(name, ()):
            rows += stock_rows(ticker)
        spec = ASSETS.get(name)
        title = spec.short_name if spec else name
        subtitle = f"{name} · {spec.asset_code} · {SECTOR_TITLES[spec.sector]}" if spec else name
        futures.append(collector.instrument(f"futures/{name}", title, subtitle, rows))

    # ---------------------------------------------------------------- акции
    tickers = set().union(*by_dataset.values()) if by_dataset else set()
    tickers |= column_values(paths.reference / "corp_dividends.parquet", "ticker")
    tickers |= column_values(paths.reference / "corp_splits.parquet", "ticker")
    listing = paths.reference / "corp_listing.parquet"
    names: dict[str, str] = {}
    if listing.is_file() and reference["corp_listing.parquet"].status != "error":
        names = dict(frame_of(listing).select("ticker", "shortname").drop_nulls().iter_rows())
    stocks = []
    for ticker in sorted(tickers):
        stocks.append(collector.instrument(f"stocks/{ticker}", names.get(ticker, ticker), ticker, stock_rows(ticker)))

    # ---------------------------------------------------------------- общее
    expected_folders = [
        paths.reference,
        paths.raw_futures_min,
        paths.raw_futures_daily,
        data_root / "continuous_15m",
        paths.raw_futoi_5min,
        paths.raw_futoi_daily,
        paths.raw / "algopack",
    ]
    missing = [missing_folder(ctx, folder) for folder in expected_folders if not folder.is_dir()]
    unmatched = futoi_by_market.get(None, [])
    common = [
        collector.instrument(
            "common/reference", "Справочники и макро", "data/reference — файлы целиком", reference.values()
        ),
        collector.instrument("common/continuous", "Склейка: общие файлы", "data/continuous_15m", continuous_common),
        collector.instrument(
            "common/futoi",
            "FUTOI без инструмента",
            "тикеры, которых нет в core/instruments.py",
            [row for ticker in unmatched for row in (futoi_5min.get(ticker), futoi_daily.get(ticker))],
        ),
        collector.instrument(
            "common/algopack", "Algopack целиком", "data/raw/algopack — архив", scan_algopack(ctx)
        ),
        collector.instrument(
            "common/other", "Прочее", "папки вне каталога и отсутствующие папки", [*scan_other(ctx), *missing]
        ),
    ]

    sections = [
        section("futures", "Фьючерсы", "Минутки, дневки, склейка, FUTOI и справочники по каждому рынку", futures),
        section("stocks", "Акции", "Algopack (архив) и корпоративные события по каждой бумаге", stocks),
        section("common", "Общее", "Справочники целиком и файлы, общие для всех рынков", common),
    ]
    datasets = tuple(collector.datasets.values())
    summary = {status: 0 for status in SEVERITY}
    for row in datasets:
        summary[row.status] += 1
    return InventoryReport(
        generated_at=now,
        expected_date=ctx.expected,
        data_root=str(data_root),
        sections=tuple(item for item in sections if item.instruments),
        datasets=datasets,
        summary=summary,
    )
