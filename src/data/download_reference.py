"""Загрузка и инкрементальная докачка макро-ставок, спота, бенчмарков и календарей.

Сохраняет референсные данные в формате Parquet в папку data/reference/:
  1. cbr_key_rate.parquet       — ключевая ставка Банка России (с 2014 года)
  2. rates_cbr_fx.parquet       — официальные курсы валют ЦБ РФ (USD, CNY, EUR)
  3. rates_china.parquet        — ставки Китая: SHIBOR (1Y, 3M) и LPR 1Y (через akshare)
  4. rates_usd.parquet          — ставка США SOFR (New York Fed / FRED)
  5. spot_currency.parquet      — биржевой спот MOEX: CNYRUB_TOM, USD000UTSTOM, EUR_RUB__TOM
  6. indices_daily.parquet      — фондовые индексы: Индекс МосБиржи (IMOEX) и РТС (RTSI)
  7. benchmarks_world.parquet   — мировые бенчмарки (Brent, Henry Hub Gas, Gold, Silver, Copper, S&P 500, Nasdaq)
  8. eia_ng_storage.parquet     — еженедельные запасы природного газа в ПХГ США (EIA Lower 48)
  9. calendar_events.parquet    — календарь событий: заседания ЦБ, налоговые даты (ЕНП 28 числа), экспирации
 10. Корпоративные реестры      — corp_dividends, corp_splits, corp_listing, ru_working_saturdays
 11. futures_contracts.parquet  — реестр спецификаций фьючерсов FORTS (активные + архивные)
 12. margin_rates.parquet       — дневная история минимальных ставок ГО (mr1–mr3) по базовым активам FORTS
     margin_snapshot.parquet    — снимки текущего ГО всех фьючерсов и коэффициент ГО к mr1 (копятся при каждом запуске)
 13. rates_moex_clearing.parquet — курсы клиринга MOEX (USD/RUB): по ним биржа считает стоимость шага долларовых контрактов

Поддержка режимов:
  - Первичная загрузка с глубокой историей (с 2020 года):
      uv run python src/data/download_reference.py
  - Быстрая инкрементальная докачка новых дней:
      uv run python src/data/download_reference.py --update

Надёжность:
  - Любая ошибка источника — файл справочника не меняется: частичный
    результат не записывается (без --update он стирал бы целый ряд).
  - При --update старт берётся по самому отставшему ряду файла (валюта,
    тикер, актив), а не по последней дате всего файла: ряд, не скачавшийся
    в прошлый раз, догоняется, а не остаётся с дырой.
  - В futures_contracts шаг цены истёкших контрактов не угадывается: только
    значение с биржи, пока контракт торговался, иначе пусто. Контракты из
    прежнего справочника не выпадают при пересборке.
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Гарантия импорта src независимо от способа вызова
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import polars as pl
import requests
from bs4 import BeautifulSoup

from src.core.config import (
    ASSETS,
    ISS_BASE_URL,
    MAX_RETRIES,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    Cadence,
    Paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REFERENCE_DIR = Paths().reference
# Только для разовой копии корпоративных реестров (sync_corporate_and_calendar_files).
ML_REFERENCE_DIR = Path("/Users/anton/Documents/app/ml/data/reference")

DEFAULT_FROM_DATE = "2020-01-01"


# -----------------------------------------------------------------------------
# Хелпер для докачки (incremental update)
# -----------------------------------------------------------------------------
def get_update_start_date(
    file_path: Path,
    default_from: str,
    date_col: str = "date",
    key_col: str | None = None,
    keys: Sequence[str] = (),
) -> str:
    """Определяет дату, с которой нужно докачивать данные, если файл уже существует.

    В файле с несколькими рядами (key_col: валюта, тикер, актив) старт берётся
    по ряду, который отстал сильнее всех. Иначе ряд, не скачавшийся в прошлый
    раз, остался бы с дырой: остальные ряды уже ушли вперёд. Если какого-то
    ряда из keys в файле нет совсем, качаем с default_from.
    """
    if not file_path.exists():
        return default_from
    try:
        df = pl.read_parquet(file_path)
        if len(df) > 0 and date_col in df.columns:
            if key_col is None:
                max_d = df[date_col].max()
            else:
                last_by_key = dict(
                    df.group_by(key_col).agg(pl.col(date_col).max()).iter_rows()
                )
                if any(key not in last_by_key for key in keys):
                    return default_from
                max_d = min(last_by_key[key] for key in keys) if keys else min(last_by_key.values())
            if max_d is not None:
                # Начинаем со следующего дня после максимальной даты
                next_d = max_d + timedelta(days=1)
                return next_d.strftime("%Y-%m-%d")
    except Exception as exc:
        log.warning("Не удалось прочитать существующий файл %s (%s), качаем с %s", file_path.name, exc, default_from)
    return default_from


def refresh_reference_file(
    label: str,
    path: Path,
    fetch: Callable[[str, str], pl.DataFrame],
    *,
    from_date: str,
    till_date: str,
    update: bool,
    key_col: str | None = None,
    keys: Sequence[str] = (),
    finalize: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> None:
    """Скачивает файл справочника или докачивает его, не портя при ошибке.

    - Любая ошибка источника — файл не меняется. Раньше частичный результат
      записывался: без --update пропадал целый ряд, при --update оставалась дыра.
    - При докачке новые строки заменяют старые за те же даты.
    - finalize пересчитывает производные столбцы по всему файлу после слияния.
    """
    start = (
        get_update_start_date(path, from_date, key_col=key_col, keys=keys)
        if update
        else from_date
    )
    if start > till_date:
        log.info("%s: актуален, пропускаем.", label)
        return
    log.info("%s (с %s по %s)...", label, start, till_date)
    t0 = time.time()
    try:
        new_df = fetch(start, till_date)
    except Exception as exc:  # noqa: BLE001 — граница: любая ошибка источника
        log.error("  ✗ %s: ошибка загрузки (%s), файл не изменён", path.name, exc)
        return
    if new_df.is_empty():
        log.info("  %s: новых данных нет, файл не изменён.", path.name)
        return

    full_df = new_df
    if update and path.exists():
        full_df = pl.concat([pl.read_parquet(path), new_df], how="diagonal_relaxed")
    subset = [key_col, "date"] if key_col else ["date"]
    full_df = full_df.unique(subset=subset, keep="last", maintain_order=True).sort(subset)
    if finalize is not None:
        full_df = finalize(full_df)
    full_df.write_parquet(path, compression="zstd")
    log.info("  ✓ %s: %d записей за %.1fс", path.name, len(full_df), time.time() - t0)


# -----------------------------------------------------------------------------
# 1. Ключевая ставка Банка России
# -----------------------------------------------------------------------------
def fetch_cbr_key_rate(from_date: str, till_date: str) -> pl.DataFrame:
    """Выгружает историю ключевой ставки с официального сайта Банка России."""
    d_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d.%m.%Y")
    d_till = datetime.strptime(till_date, "%Y-%m-%d").strftime("%d.%m.%Y")
    url = f"https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From={d_from}&UniDbQuery.To={d_till}"

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "data"})
    if not table:
        return pl.DataFrame()

    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) >= 2:
            dt_str = tds[0].get_text(strip=True)
            val_str = tds[1].get_text(strip=True).replace(",", ".")
            try:
                dt = datetime.strptime(dt_str, "%d.%m.%Y").date()
                rate = float(val_str)
                rows.append({"date": dt, "rate": rate})
            except ValueError:
                continue

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(rows)
        .unique(subset=["date"])
        .sort("date")
    )


# -----------------------------------------------------------------------------
# 2. Официальные курсы валют ЦБ РФ (USD, CNY, EUR)
# -----------------------------------------------------------------------------
CBR_CURRENCIES = {
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",
}


def fetch_cbr_fx_rates(from_date: str, till_date: str) -> pl.DataFrame:
    """Загружает официальные курсы USD, CNY, EUR через XML-сервис ЦБ РФ."""
    d1 = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    d2 = datetime.strptime(till_date, "%Y-%m-%d").strftime("%d/%m/%Y")

    all_rows = []
    for curr, val_code in CBR_CURRENCIES.items():
        url = f"https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={d1}&date_req2={d2}&VAL_NM_RQ={val_code}"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for rec in root.findall("Record"):
            dt_str = rec.attrib.get("Date")
            val_node = rec.find("Value")
            nom_node = rec.find("Nominal")
            if dt_str and val_node is not None and val_node.text:
                dt = datetime.strptime(dt_str, "%d.%m.%Y").date()
                nominal = float(nom_node.text) if nom_node is not None and nom_node.text else 1.0
                rate = float(val_node.text.replace(",", ".")) / nominal
                all_rows.append({"date": dt, "currency": curr, "rate": rate})

    if not all_rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(all_rows)
        .unique(subset=["date", "currency"])
        .sort(["currency", "date"])
    )


# -----------------------------------------------------------------------------
# 3. Ставки Китая: SHIBOR (1Y, 3M) и LPR 1Y
# -----------------------------------------------------------------------------
CHINA_SERIES = ["shibor_1y", "shibor_3m", "lpr_1y"]


def _is_number(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def fetch_china_rates(from_date: str, till_date: str) -> pl.DataFrame:
    """Загружает ставки Китая через библиотеку akshare."""
    import akshare as ak

    all_rows = []

    # 1. SHIBOR
    shibor_df = ak.macro_china_shibor_all()
    shibor_df = shibor_df[(shibor_df["日期"] >= from_date) & (shibor_df["日期"] <= till_date)]
    for _, row in shibor_df.iterrows():
        d = date.fromisoformat(str(row["日期"])[:10])
        r_1y = row["1Y-定价"]
        r_3m = row["3M-定价"]
        if _is_number(r_1y):
            all_rows.append({"date": d, "series": "shibor_1y", "rate": float(r_1y)})
        if _is_number(r_3m):
            all_rows.append({"date": d, "series": "shibor_3m", "rate": float(r_3m)})

    # 2. LPR
    lpr_df = ak.macro_china_lpr()
    for _, row in lpr_df.iterrows():
        d_str = str(row["TRADE_DATE"])[:10]
        if d_str < from_date or d_str > till_date:
            continue
        r_lpr = row["LPR1Y"]
        if _is_number(r_lpr):
            all_rows.append({"date": date.fromisoformat(d_str), "series": "lpr_1y", "rate": float(r_lpr)})

    if not all_rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(all_rows)
        .unique(subset=["date", "series"])
        .sort(["series", "date"])
    )


# -----------------------------------------------------------------------------
# 4. Ставка США SOFR (New York Fed / FRED)
# -----------------------------------------------------------------------------
def fetch_usd_sofr(from_date: str, till_date: str) -> pl.DataFrame:
    """Загружает ставку SOFR с New York Fed (с резервом на FRED)."""
    rows = []
    # 1. New York Fed
    try:
        url = f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate={from_date}&endDate={till_date}"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            payload = resp.json()
            for r in payload.get("refRates", []):
                dt_str = r.get("effectiveDate")
                rate_val = r.get("percentRate")
                if dt_str and rate_val is not None:
                    rows.append({"date": date.fromisoformat(dt_str), "rate": float(rate_val)})
    except (requests.RequestException, ValueError) as exc:
        log.warning("New York Fed SOFR недоступен (%s), пробуем FRED...", exc)

    # 2. FRED — второй официальный источник того же ряда; его ошибка уже не глотается.
    if not rows:
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"
        resp = requests.get(fred_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw_df = pl.read_csv(io.BytesIO(resp.content), null_values=["."])
        d_col, v_col = raw_df.columns[0], raw_df.columns[1]
        df_fred = (
            raw_df.filter(pl.col(v_col).is_not_null())
            .select([
                pl.col(d_col).str.to_date().alias("date"),
                pl.col(v_col).cast(pl.Float64).alias("rate"),
            ])
            .filter((pl.col("date") >= date.fromisoformat(from_date)) & (pl.col("date") <= date.fromisoformat(till_date)))
        )
        return df_fred.unique(subset=["date"]).sort("date")

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(rows)
        .unique(subset=["date"])
        .sort("date")
    )


# -----------------------------------------------------------------------------
# 5. Биржевой спот валют (MOEX ISS CETS)
# -----------------------------------------------------------------------------
SPOT_TICKERS = ["CNYRUB_TOM", "USD000UTSTOM", "EUR_RUB__TOM"]


def fetch_spot_currencies(from_date: str, till_date: str) -> pl.DataFrame:
    """Загружает дневную историю CNYRUB_TOM, USD000UTSTOM и EUR_RUB__TOM с MOEX ISS."""
    all_rows = []

    for secid in SPOT_TICKERS:
        cursor = 0
        while True:
            url = (
                f"https://iss.moex.com/iss/history/engines/currency/markets/selt"
                f"/boards/CETS/securities/{secid}.json"
                f"?from={from_date}&till={till_date}&start={cursor}&iss.meta=off&iss.only=history"
            )
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            history = payload.get("history", {})
            cols = history.get("columns", [])
            data = history.get("data", [])
            if not data:
                break

            i_date = cols.index("TRADEDATE")
            i_open = cols.index("OPEN")
            i_high = cols.index("HIGH")
            i_low = cols.index("LOW")
            i_close = cols.index("CLOSE")
            i_wap = cols.index("WAPRICE") if "WAPRICE" in cols else -1
            i_vol = cols.index("VOLUME") if "VOLUME" in cols else -1

            for r in data:
                if r[i_close] is None:
                    continue
                all_rows.append({
                    "date": date.fromisoformat(r[i_date]),
                    "secid": secid,
                    "open": float(r[i_open]) if r[i_open] is not None else None,
                    "high": float(r[i_high]) if r[i_high] is not None else None,
                    "low": float(r[i_low]) if r[i_low] is not None else None,
                    "close": float(r[i_close]),
                    "waprice": float(r[i_wap]) if i_wap >= 0 and r[i_wap] is not None else None,
                    "volume": float(r[i_vol]) if i_vol >= 0 and r[i_vol] is not None else 0.0,
                })

            cursor += len(data)
            if len(data) < 100:
                break
            time.sleep(REQUEST_DELAY)

    if not all_rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(all_rows)
        .unique(subset=["date", "secid"])
        .sort(["secid", "date"])
    )


# -----------------------------------------------------------------------------
# 6. Фондовые индексы (IMOEX, RTSI)
# -----------------------------------------------------------------------------
INDEX_BOARDS = {"IMOEX": "SNDX", "RTSI": "RTSI"}


def fetch_indices(from_date: str, till_date: str) -> pl.DataFrame:
    """Загружает дневную историю IMOEX (SNDX) и RTSI (RTSI) с MOEX ISS."""
    all_rows = []

    for secid, board in INDEX_BOARDS.items():
        cursor = 0
        while True:
            url = (
                f"https://iss.moex.com/iss/history/engines/stock/markets/index"
                f"/boards/{board}/securities/{secid}.json"
                f"?from={from_date}&till={till_date}&start={cursor}&iss.meta=off&iss.only=history"
            )
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            history = payload.get("history", {})
            cols = history.get("columns", [])
            data = history.get("data", [])
            if not data:
                break

            i_date = cols.index("TRADEDATE")
            i_open = cols.index("OPEN")
            i_high = cols.index("HIGH")
            i_low = cols.index("LOW")
            i_close = cols.index("CLOSE")
            i_vol = cols.index("VOLUME") if "VOLUME" in cols else -1
            i_val = cols.index("VALUE") if "VALUE" in cols else -1

            for r in data:
                if r[i_close] is None:
                    continue
                all_rows.append({
                    "date": date.fromisoformat(r[i_date]),
                    "secid": secid,
                    "open": float(r[i_open]) if r[i_open] is not None else None,
                    "high": float(r[i_high]) if r[i_high] is not None else None,
                    "low": float(r[i_low]) if r[i_low] is not None else None,
                    "close": float(r[i_close]),
                    "volume": float(r[i_vol]) if i_vol >= 0 and r[i_vol] is not None else 0.0,
                    "value": float(r[i_val]) if i_val >= 0 and r[i_val] is not None else 0.0,
                })

            cursor += len(data)
            if len(data) < 100:
                break
            time.sleep(REQUEST_DELAY)

    if not all_rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(all_rows)
        .unique(subset=["date", "secid"])
        .sort(["secid", "date"])
    )


# -----------------------------------------------------------------------------
# 7. Мировые товарные и индексные бенчмарки через akshare
# -----------------------------------------------------------------------------
WORLD_BENCHMARKS = {
    "brent": "OIL",     # Brent Crude Oil (ICE)
    "ng": "NG",         # Henry Hub Natural Gas (NYMEX)
    "gold": "GC",       # Gold (COMEX)
    "silver": "SI",     # Silver (COMEX)
    "copper": "HG",     # Copper (COMEX)
    "spy": "ES",        # S&P 500 E-mini (CME)
    "nasdaq": "NQ",     # Nasdaq 100 E-mini (CME)
}


def fetch_world_benchmarks(from_date: str, till_date: str) -> pl.DataFrame:
    """Загружает дневные котировки мировых бенчмарков через akshare.

    Если хоть один бенчмарк не скачался, бросает исключение: файл не меняется.
    Дневная доходность ret_1d здесь не считается — её пересчитывает
    add_benchmark_returns по всему файлу после слияния со старыми данными.
    """
    import akshare as ak

    def fetch_single(asset: str, symbol: str) -> pl.DataFrame:
        df_pd = ak.futures_foreign_hist(symbol=symbol)
        if df_pd is None or df_pd.empty:
            raise ValueError(f"akshare вернул пустую историю {symbol} ({asset})")

        df = pl.from_pandas(df_pd)
        if df.schema["date"] in (pl.Utf8, pl.String):
            df = df.with_columns(pl.col("date").str.to_date().alias("date"))
        else:
            df = df.with_columns(pl.col("date").cast(pl.Date).alias("date"))

        return (
            df.filter((pl.col("date") >= date.fromisoformat(from_date)) & (pl.col("date") <= date.fromisoformat(till_date)))
            .select([
                pl.col("date"),
                pl.lit(symbol).alias("symbol"),
                pl.lit(asset).alias("asset"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ])
            .unique(subset=["date"], keep="last", maintain_order=True)
        )

    all_frames = []
    # Пул потоков: бенчмарки качаются параллельно, result(timeout) не даёт
    # бесконечно ждать один источник (сам поток при этом не прерывается).
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fetch_single, asset, sym): (asset, sym) for asset, sym in WORLD_BENCHMARKS.items()}
        for fut, (asset, sym) in future_map.items():
            try:
                res = fut.result(timeout=12)
            except Exception as exc:
                raise RuntimeError(f"бенчмарк {sym} ({asset}) не скачался: {exc}") from exc
            all_frames.append(res)
            if len(res) > 0:
                log.info("  ✓ Мировой бенчмарк %s (%s): %d дней (последний Close: %.2f)", sym, asset, len(res), res["close"][-1])

    return pl.concat(all_frames).sort(["asset", "date"])


def add_benchmark_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Дневная доходность каждого бенчмарка по всему файлу, после сортировки.

    Считается после слияния, чтобы на стыке старых и новых данных не было
    пропуска, и после сортировки по дате: удаление дублей порядок не хранит.
    """
    return df.sort(["asset", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("asset") - 1.0).alias("ret_1d")
    )


# -----------------------------------------------------------------------------
# 8. Запасы природного газа в ПХГ США (EIA Lower 48 Natural Gas Storage)
# -----------------------------------------------------------------------------
def fetch_eia_ng_storage() -> pl.DataFrame:
    """Загружает еженедельный отчет EIA о запасах природного газа в США."""
    import pandas as pd

    url = "https://www.eia.gov/dnav/ng/xls/NG_STOR_WKLY_S1_W.xls"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    df_raw = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data 1", skiprows=3, header=None)
    df_raw.columns = [
        "date",
        "storage_total_bcf",
        "storage_east_bcf",
        "storage_midwest_bcf",
        "storage_mountain_bcf",
        "storage_pacific_bcf",
        "storage_south_central_bcf",
        "storage_salt_bcf",
        "storage_nonsalt_bcf",
    ]
    df_raw = df_raw.dropna(subset=["date", "storage_total_bcf"])
    df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.date

    df = pl.from_dict({col: df_raw[col].tolist() for col in df_raw.columns})
    df = (
        df.with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("storage_total_bcf").cast(pl.Float64),
            (pl.col("date") + pl.duration(days=6)).alias("release_date"),
        )
        .sort("date")
        .with_columns(
            (pl.col("storage_total_bcf") - pl.col("storage_total_bcf").shift(1)).alias("net_change_bcf")
        )
    )
    return df


# -----------------------------------------------------------------------------
# 9. Календарь макро-событий (Заседания ЦБ, Налоги, Экспирации)
# -----------------------------------------------------------------------------
def build_calendar_events(from_date: str, till_date: str) -> pl.DataFrame:
    """Формирует сводный календарь регулярных макроэкономических и биржевых событий."""
    events = []
    d_start = datetime.strptime(from_date, "%Y-%m-%d").date()
    d_end = datetime.strptime(till_date, "%Y-%m-%d").date()

    # 1. Налоговые даты (Единый налоговый платеж ЕНП — 28-е число каждого месяца)
    cur = d_start.replace(day=1)
    while cur <= d_end:
        try:
            tax_d = cur.replace(day=28)
            if tax_d.weekday() == 5:
                tax_d += timedelta(days=2)
            elif tax_d.weekday() == 6:
                tax_d += timedelta(days=1)

            if d_start <= tax_d <= d_end:
                events.append({
                    "date": tax_d,
                    "event_type": "tax_payment_enp",
                    "description": "Единый налоговый платеж (ЕНП, уплата налогов экспортерами)",
                })
        except ValueError:
            pass

        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    # 2. Квартальные экспирации FORTS (3-й четверг марта, июня, сентября, декабря)
    cur = d_start.replace(day=1)
    while cur <= d_end:
        if cur.month in (3, 6, 9, 12):
            first_day = cur.replace(day=1)
            offset = (3 - first_day.weekday()) % 7
            third_thursday = first_day + timedelta(days=offset + 14)
            if d_start <= third_thursday <= d_end:
                events.append({
                    "date": third_thursday,
                    "event_type": "quarterly_expiry",
                    "description": f"Квартальная экспирация фьючерсов FORTS ({cur.strftime('%b %Y')})",
                })

        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    # 3. Заседания Банка России по ставке
    try:
        kr_path = REFERENCE_DIR / "cbr_key_rate.parquet"
        if kr_path.exists():
            kr_df = pl.read_parquet(kr_path)
            diffs = kr_df.filter(pl.col("rate") != pl.col("rate").shift(1))
            for row in diffs.iter_rows(named=True):
                if d_start <= row["date"] <= d_end:
                    events.append({
                        "date": row["date"],
                        "event_type": "cbr_rate_change",
                        "description": f"Изменение ключевой ставки ЦБ РФ: {row['rate']:.2f}%",
                    })
    except Exception as exc:
        log.warning("Не удалось включить решения ЦБ в календарь: %s", exc)

    if not events:
        return pl.DataFrame()

    return (
        pl.DataFrame(events)
        .unique(subset=["date", "event_type"])
        .sort("date")
    )


# -----------------------------------------------------------------------------
# 10. Корпоративные реестры и рабочие субботы
# -----------------------------------------------------------------------------
def _parse_iso_date(val: Any) -> date | None:
    if val is None or val == "" or str(val).lower() == "none":
        return None
    try:
        return date.fromisoformat(str(val)[:10])
    except (ValueError, TypeError):
        return None


def fetch_futures_contracts(
    output_path: Path,
    update: bool = False,
    max_workers: int = 8,
) -> pl.DataFrame:
    """Формирует и обновляет централизованный справочник спецификаций фьючерсов FORTS.

    Использует скиллы moexalgo-market-data и moex-iss:
      1. moexalgo.Market("FO").tickers("*") — снимок всех активных контрактов FORTS
         (шаг цены, стоимость шага, лотность, последнее ГО, дата экспирации).
      2. data/raw/futures/ — обнаружение всех 650+ исторических серий проекта.
      3. src.core.instruments.ASSETS — реестр активов и префиксов кодов.
      4. MOEX ISS (securities/{ticker}.json) — паспортные даты FRSTTRADE/LSTTRADE и лотность
         для завершенных исторических контрактов.

    Режим update:
      Уже завершенные контракты с известными датами не запрашиваются повторно по сети.
    """
    from moexalgo import Market

    # 1. Построение маппинга тикеров и базовых активов
    ticker_to_asset: dict[str, str] = {}
    prefix_to_asset: dict[str, str] = {}
    assetcode_to_asset: dict[str, str] = {}

    for asset_name, spec in ASSETS.items():
        if spec.ticker_prefix:
            prefix_to_asset[spec.ticker_prefix.upper()] = asset_name
        for p in getattr(spec, "extra_prefixes", []):
            prefix_to_asset[p.upper()] = asset_name
        if spec.asset_code:
            assetcode_to_asset[spec.asset_code.upper()] = asset_name
        for fixed in spec.fixed_tickers:
            ticker_to_asset[fixed.upper()] = asset_name
        for ticker in spec.generate_tickers():
            ticker_to_asset.setdefault(ticker.upper(), asset_name)

    # Исторические алиасы базовых активов (например, TCSI -> tbank после переименования в T)
    assetcode_to_asset["TCSI"] = "tbank"
    assetcode_to_asset["T"] = "tbank"

    raw_futures_dir = Paths().raw_futures_min
    if raw_futures_dir.exists():
        for asset_dir in raw_futures_dir.iterdir():
            if asset_dir.is_dir() and not asset_dir.name.startswith("."):
                asset_name = asset_dir.name
                for f in asset_dir.glob("*.parquet"):
                    ticker_to_asset[f.stem.upper()] = asset_name

    # 2. Выгрузка активных контрактов через moexalgo
    active_map: dict[str, dict[str, Any]] = {}
    try:
        fo = Market("FO")
        active_df = fo.tickers("*")
        for rec in active_df.to_dict(orient="records"):
            secid = str(rec.get("ticker", "")).strip().upper()
            if secid:
                active_map[secid] = rec
                if secid not in ticker_to_asset:
                    pref = secid[:2]
                    acode = str(rec.get("assetcode", "")).upper()
                    matched_asset = prefix_to_asset.get(pref) or assetcode_to_asset.get(acode) or acode.lower()
                    ticker_to_asset[secid] = matched_asset
        log.info("  ✓ moexalgo: получено %d активных контрактов FORTS", len(active_map))
    except Exception as exc:
        log.warning("Не удалось получить срез активных контрактов через moexalgo: %s", exc)

    # 3. Загрузка существующего кэша при наличии
    cached_records: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        try:
            old_df = pl.read_parquet(output_path)
            for row in old_df.iter_rows(named=True):
                cached_records[row["ticker"].upper()] = row
            log.info("  Загружено %d существующих записей из %s", len(cached_records), output_path.name)
        except Exception as exc:
            log.warning("Ошибка чтения существующего справочника %s: %s", output_path, exc)

    # Контракты из старого справочника остаются всегда. Иначе истёкшие контракты
    # рынков вне реестра (их нет ни в генераторе тикеров, ни среди активных)
    # пропадали бы из справочника при следующем запуске.
    all_tickers = sorted(set(ticker_to_asset) | set(active_map) | set(cached_records))
    log.info("  Всего контрактов для учета в справочнике: %d", len(all_tickers))

    today = date.today()
    tickers_to_query: list[str] = []

    for t in all_tickers:
        if update and t in cached_records:
            rec = cached_records[t]
            lst = rec.get("last_trade_date")
            if lst is not None and lst < today and rec.get("first_trade_date") is not None and rec.get("lot") is not None:
                continue
        tickers_to_query.append(t)

    log.info("  Требуется запрос спецификаций MOEX ISS для %d контрактов...", len(tickers_to_query))

    # 4. Запрос спецификаций через MOEX ISS
    def fetch_desc(secid: str) -> tuple[str, dict[str, Any]]:
        url = f"{ISS_BASE_URL}/securities/{secid}.json"
        params = {"iss.only": "description", "iss.meta": "off"}
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    payload = resp.json()
                    desc_data = payload.get("description", {}).get("data", [])
                    return secid, {row[0]: row[2] for row in desc_data if len(row) >= 3}
                elif resp.status_code in (429, 502, 503, 504):
                    time.sleep(REQUEST_DELAY * (2**attempt))
                else:
                    break
            except (requests.RequestException, ValueError):
                time.sleep(REQUEST_DELAY * (2**attempt))
        return secid, {}

    queried_meta: dict[str, dict[str, Any]] = {}
    if tickers_to_query:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for secid, desc in pool.map(fetch_desc, tickers_to_query):
                queried_meta[secid] = desc

    # 5. Сборка итоговых записей
    final_rows: list[dict[str, Any]] = []
    for t in all_tickers:
        act = active_map.get(t, {})
        desc = queried_meta.get(t, {})
        cached = cached_records.get(t, {})

        name = desc.get("NAME") or desc.get("CONTRACTNAME") or act.get("secname") or cached.get("name") or t
        short_name = desc.get("SHORTNAME") or act.get("shortname") or cached.get("short_name") or t
        asset_code = desc.get("ASSETCODE") or act.get("assetcode") or cached.get("asset_code")
        asset = ticker_to_asset.get(t) or cached.get("asset") or (asset_code.lower() if asset_code else "")

        first_trade = _parse_iso_date(desc.get("FRSTTRADE")) or cached.get("first_trade_date")
        last_trade = (
            _parse_iso_date(desc.get("LSTTRADE"))
            or _parse_iso_date(act.get("lasttradedate"))
            or cached.get("last_trade_date")
        )
        expiration = (
            _parse_iso_date(desc.get("LSTDELDATE"))
            or _parse_iso_date(act.get("lastdeldate"))
            or cached.get("expiration_date")
        )

        raw_lot = act.get("lotvolume") or desc.get("LOTSIZE") or cached.get("lot") or 1
        try:
            lot = int(raw_lot)
        except (ValueError, TypeError):
            lot = 1

        # Шаг цены и его стоимость биржа отдаёт только по действующим контрактам.
        # Для истёкших шаг раньше угадывался по числу знаков после запятой
        # (MIX 1 вместо 25, MXI 0.01 вместо 0.05). Теперь из кэша берётся только
        # шаг, записанный вместе со стоимостью шага — то есть пришедший с биржи,
        # пока контракт торговался. Иначе — пусто, а не догадка.
        min_step = act.get("minstep")
        step_price = act.get("stepprice")
        if (
            min_step is None
            and step_price is None
            and cached.get("min_price_increment_amount") is not None
        ):
            min_step = cached.get("min_price_increment")
            step_price = cached.get("min_price_increment_amount")

        final_rows.append({
            "ticker": t,
            "name": str(name),
            "short_name": str(short_name),
            "asset": str(asset),
            "asset_code": str(asset_code) if asset_code else None,
            "first_trade_date": first_trade,
            "last_trade_date": last_trade,
            "expiration_date": expiration,
            "lot": lot,
            "min_price_increment": float(min_step) if min_step is not None else None,
            "min_price_increment_amount": float(step_price) if step_price is not None else None,
        })

    result_df = (
        pl.DataFrame(final_rows)
        .sort(["asset", "last_trade_date", "ticker"])
    )
    return result_df


# -----------------------------------------------------------------------------
# 12. Ставки обеспечения (ГО) срочного рынка MOEX
# -----------------------------------------------------------------------------
def margin_asset_codes() -> list[str]:
    """Коды базовых активов реестра, для которых нужны ставки ГО (без вечных)."""
    return sorted({
        spec.asset_code
        for spec in ASSETS.values()
        if spec.cadence != Cadence.PERPETUAL
    })


def fetch_margin_rates(starts: dict[str, str], till_date: str) -> pl.DataFrame:
    """История минимальных ставок обеспечения (ГО) по базовым активам FORTS.

    ISS rms/limits: на каждый торговый день биржа задаёт три уровня ставки
    mr1 < mr2 < mr3 — доли от стоимости контракта — и границы объёма позиции
    lk1, lk2, после которых действует следующий уровень. Для небольшой позиции
    действует mr1. Фактическое ГО контракта не ниже mr1 × стоимость контракта;
    бесплатной истории фактического ГО в ISS нет.

    starts — код актива -> дата начала загрузки (у каждого актива своя).
    При ошибке запроса бросает исключение: частично скачанная история
    при докачке оставила бы дыры, которые потом не заполнятся.
    """
    rows = []
    for code, from_date in sorted(starts.items()):
        if from_date > till_date:
            continue
        cursor = 0
        while True:
            url = (
                f"{ISS_BASE_URL}/rms/engines/futures/objects/limits/{code}.json"
                f"?from={from_date}&till={till_date}&start={cursor}&iss.meta=off"
            )
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            block = payload.get("limits", {})
            cols = block.get("columns", [])
            data = block.get("data", [])
            for values in data:
                r = dict(zip(cols, values))
                rows.append({
                    "date": date.fromisoformat(r["tradedate"]),
                    "asset_code": r["assetcode"],
                    "mr1": float(r["mr1"]) if r["mr1"] is not None else None,
                    "mr2": float(r["mr2"]) if r["mr2"] is not None else None,
                    "mr3": float(r["mr3"]) if r["mr3"] is not None else None,
                    "lk1": float(r["lk1"]) if r["lk1"] is not None else None,
                    "lk2": float(r["lk2"]) if r["lk2"] is not None else None,
                })
            cursor += len(data)
            total = payload.get("limits.cursor", {}).get("data", [[0, 0, 0]])[0][1]
            if not data or cursor >= total:
                break
            time.sleep(REQUEST_DELAY)
        # Пусто бывает и в норме: в выходные новых торговых дней нет.
        # Об активах, которых нет в файле вовсе, предупреждает main().
        time.sleep(REQUEST_DELAY)

    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .unique(subset=["date", "asset_code"])
        .sort(["asset_code", "date"])
    )


def _iss_block(url: str, block: str) -> list[dict[str, Any]]:
    """Строки одного блока ответа ISS как словари; ошибка запроса — исключение."""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json().get(block, {})
    cols = payload.get("columns", [])
    return [dict(zip(cols, values)) for values in payload.get("data", [])]


def fetch_margin_snapshot(snapshot_date: date) -> pl.DataFrame:
    """Снимок фактического ГО всех фьючерсов FORTS и его коэффициент к mr1.

    Истории фактического ГО в бесплатном ISS нет, есть только текущее значение.
    Коэффициент связывает его с историей минимальных ставок:
      стоимость контракта = расчётная цена × стоимость шага / шаг цены
      доля ГО             = ГО / стоимость контракта
      коэффициент         = доля ГО / mr1
    Пример BRX6: ГО 21.7% от стоимости при mr1 = 18% — коэффициент 1.21.
    ГО за прошлую дату оценивается как mr1 на ту дату × коэффициент.
    mr1 берётся из последних опубликованных ставок (rates_date): биржа
    публикует их заранее, на ближайший торговый день.
    """
    securities = _iss_block(
        f"{ISS_BASE_URL}/engines/futures/markets/forts/securities.json"
        "?iss.meta=off&iss.only=securities",
        "securities",
    )
    limits = _iss_block(
        f"{ISS_BASE_URL}/rms/engines/futures/objects/limits.json?iss.meta=off",
        "limits",
    )
    if not securities or not limits:
        raise ValueError("ISS вернул пустой список контрактов или ставок ГО")

    rates = pl.DataFrame(
        [
            {
                "asset_code": r["assetcode"],
                "rates_date": date.fromisoformat(r["tradedate"]),
                "mr1": float(r["mr1"]) if r["mr1"] is not None else None,
            }
            for r in limits
        ]
    ).unique(subset=["asset_code"], keep="last")

    def number(value: Any) -> float | None:
        return float(value) if value not in (None, "", 0) else None

    contracts = pl.DataFrame(
        [
            {
                "snapshot_date": snapshot_date,
                "im_time": datetime.fromisoformat(r["IMTIME"]) if r["IMTIME"] else None,
                "secid": r["SECID"],
                "asset_code": r["ASSETCODE"],
                "last_trade_date": (
                    date.fromisoformat(r["LASTTRADEDATE"]) if r["LASTTRADEDATE"] else None
                ),
                "open_interest": number(r["PREVOPENPOSITION"]),
                "prev_settle_price": number(r["PREVSETTLEPRICE"]),
                "min_step": number(r["MINSTEP"]),
                "step_price": number(r["STEPPRICE"]),
                "initial_margin": number(r["INITIALMARGIN"]),
            }
            for r in securities
        ],
        schema_overrides={"im_time": pl.Datetime("us")},
    )
    return (
        contracts.join(rates, on="asset_code", how="left")
        .with_columns(
            (
                pl.col("prev_settle_price") * pl.col("step_price") / pl.col("min_step")
            ).alias("contract_value")
        )
        .with_columns(
            (pl.col("initial_margin") / pl.col("contract_value")).alias("margin_share")
        )
        .with_columns((pl.col("margin_share") / pl.col("mr1")).alias("margin_factor"))
        .sort(["asset_code", "last_trade_date", "secid"])
    )


# -----------------------------------------------------------------------------
# 13. Курсы клиринга срочного рынка MOEX (индикативные курсы)
# -----------------------------------------------------------------------------
# Пары, по которым биржа пересчитывает стоимость шага цены долларовых контрактов.
CLEARING_FX_PAIRS = ["USD/RUB"]


def fetch_clearing_fx_rates(from_date: str, till_date: str) -> pl.DataFrame:
    """Курсы, по которым биржа считает стоимость шага валютных контрактов.

    У долларовых контрактов (Brent, золото) рублёвая стоимость пункта
    = размер контракта × курс клиринга. Курсов два в день: tc — дневной
    клиринг (14:00), mc — вечерний. fixing_time — когда курс стал известен.
    Курс ЦБ совпадает с биржевым только с середины 2024 года, в 2022 году
    расходился в среднем на 1.4%, в отдельные дни до 11.5%.
    """
    rows = []
    for pair in CLEARING_FX_PAIRS:
        cursor = 0
        while True:
            url = (
                f"{ISS_BASE_URL}/statistics/engines/futures/markets/indicativerates"
                f"/securities/{pair}.json"
                f"?from={from_date}&till={till_date}&start={cursor}&iss.meta=off"
            )
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            block = payload.get("securities", {})
            cols = block.get("columns", [])
            data = block.get("data", [])
            for values in data:
                r = dict(zip(cols, values))
                rows.append({
                    "date": date.fromisoformat(r["tradedate"]),
                    "clearing": r["clearing"],
                    "fixing_time": datetime.fromisoformat(f"{r['tradedate']}T{r['tradetime']}"),
                    "pair": pair,
                    "rate": float(r["rate"]),
                })
            cursor += len(data)
            total = payload.get("securities.cursor", {}).get("data", [[0, 0, 0]])[0][1]
            if not data or cursor >= total:
                break
            time.sleep(REQUEST_DELAY)

    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .unique(subset=["date", "clearing", "pair"])
        .sort(["pair", "fixing_time"])
    )


def sync_corporate_and_calendar_files() -> None:
    """Копирует корпоративные действия и рабочие субботы из ml/data/reference при наличии."""
    files = [
        "corp_dividends.parquet",
        "corp_splits.parquet",
        "corp_listing.parquet",
        "ru_working_saturdays.json",
    ]
    if not ML_REFERENCE_DIR.exists():
        return

    for fname in files:
        src = ML_REFERENCE_DIR / fname
        dst = REFERENCE_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            log.info("  ✓ Скопирован реестр: %s", fname)


# -----------------------------------------------------------------------------
# Главная функция запуска
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка и инкрементальная докачка макро-ставок, спота, бенчмарков и календарей")
    parser.add_argument(
        "--from-date",
        type=str,
        default=DEFAULT_FROM_DATE,
        help=f"Начальная дата (по умолчанию {DEFAULT_FROM_DATE})",
    )
    parser.add_argument(
        "--till-date",
        type=str,
        default=None,
        help="Конечная дата (по умолчанию сегодня)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Инкрементальный режим: докачивать только новые дни для уже существующих файлов",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Фильтр загрузки: all или через запятую (cbr,fx,china,usd,spot,indices,benchmarks,eia,calendar,contracts,margins,clearing_fx)",
    )
    args = parser.parse_args()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    till_date = args.till_date or date.today().strftime("%Y-%m-%d")
    only_set = {x.strip().lower() for x in args.only.split(",")}
    run_all = "all" in only_set

    log.info("Параметры запуска справочников:")
    log.info("  Режим: %s", "UPDATE (докачка новых дней)" if args.update else "FULL (загрузка с нуля)")
    log.info("  Конечная дата: %s", till_date)

    common = {"from_date": args.from_date, "till_date": till_date, "update": args.update}

    # 1. Ключевая ставка ЦБ
    if run_all or "cbr" in only_set:
        refresh_reference_file(
            "1/9. Ключевая ставка ЦБ РФ",
            REFERENCE_DIR / "cbr_key_rate.parquet",
            fetch_cbr_key_rate,
            **common,
        )

    # 2. Курсы валют ЦБ
    if run_all or "fx" in only_set:
        refresh_reference_file(
            "2/9. Официальные курсы валют ЦБ РФ (USD, CNY, EUR)",
            REFERENCE_DIR / "rates_cbr_fx.parquet",
            fetch_cbr_fx_rates,
            key_col="currency",
            keys=list(CBR_CURRENCIES),
            **common,
        )

    # 3. Ставки Китая
    if run_all or "china" in only_set:
        refresh_reference_file(
            "3/9. Ставки Китая (SHIBOR, LPR)",
            REFERENCE_DIR / "rates_china.parquet",
            fetch_china_rates,
            key_col="series",
            keys=CHINA_SERIES,
            **common,
        )

    # 4. Ставка США SOFR
    if run_all or "usd" in only_set:
        refresh_reference_file(
            "4/9. Ставка США SOFR",
            REFERENCE_DIR / "rates_usd.parquet",
            fetch_usd_sofr,
            **common,
        )

    # 5. Спот валют MOEX
    if run_all or "spot" in only_set:
        refresh_reference_file(
            "5/9. Дневной спот валют (CNYRUB_TOM, USD, EUR)",
            REFERENCE_DIR / "spot_currency.parquet",
            fetch_spot_currencies,
            key_col="secid",
            keys=SPOT_TICKERS,
            **common,
        )

    # 6. Фондовые индексы
    if run_all or "indices" in only_set:
        refresh_reference_file(
            "6/9. Фондовые индексы IMOEX и RTSI",
            REFERENCE_DIR / "indices_daily.parquet",
            fetch_indices,
            key_col="secid",
            keys=list(INDEX_BOARDS),
            **common,
        )

    # 7. Мировые бенчмарки
    if run_all or "benchmarks" in only_set:
        refresh_reference_file(
            "7/9. Мировые бенчмарки через akshare",
            REFERENCE_DIR / "benchmarks_world.parquet",
            fetch_world_benchmarks,
            key_col="asset",
            keys=list(WORLD_BENCHMARKS),
            finalize=add_benchmark_returns,
            **common,
        )

    # 8. Запасы газа EIA
    if run_all or "eia" in only_set:
        p = REFERENCE_DIR / "eia_ng_storage.parquet"
        need_eia = True
        if p.exists() and args.update:
            try:
                old_eia = pl.read_parquet(p)
                if len(old_eia) > 0:
                    max_d = old_eia["date"].max()
                    if max_d and (date.today() - max_d).days < 7:
                        need_eia = False
                        log.info("8/9. eia_ng_storage: актуален (последний релиз %s), пропускаем.", max_d)
            except Exception as exc:  # noqa: BLE001 — не прочитали: просто качаем заново
                log.warning("eia_ng_storage: не удалось прочитать файл (%s), качаем заново", exc)

        if need_eia:
            log.info("8/9. Загрузка еженедельного отчета EIA о запасах газа...")
            t0 = time.time()
            try:
                df_eia = fetch_eia_ng_storage()
                df_eia.write_parquet(p, compression="zstd")
                log.info("  ✓ eia_ng_storage: %d недель за %.1fс", len(df_eia), time.time() - t0)
            except Exception as exc:
                log.warning("Не удалось загрузить отчет EIA: %s", exc)

    # 9. Календарь макро-событий
    if run_all or "calendar" in only_set:
        log.info("9/9. Формирование календаря событий (заседания ЦБ, налоги, экспирации)...")
        t0 = time.time()
        p = REFERENCE_DIR / "calendar_events.parquet"
        df_cal = build_calendar_events(args.from_date, till_date)
        if len(df_cal) > 0:
            df_cal.write_parquet(p, compression="zstd")
            log.info("  ✓ calendar_events: %d событий за %.1fс", len(df_cal), time.time() - t0)

    # 10. Корпоративные реестры
    sync_corporate_and_calendar_files()

    # 11. Справочник фьючерсных контрактов FORTS
    if run_all or "contracts" in only_set or "futures" in only_set:
        p = REFERENCE_DIR / "futures_contracts.parquet"
        log.info("11/11. Справочник спецификаций фьючерсов FORTS (futures_contracts.parquet)...")
        t0 = time.time()
        df_contracts = fetch_futures_contracts(p, update=args.update)
        if len(df_contracts) > 0:
            df_contracts.write_parquet(p, compression="zstd")
            log.info("  ✓ futures_contracts: %d контрактов сохранено в %s за %.1fс", len(df_contracts), p.name, time.time() - t0)

    # 12. Ставки обеспечения (ГО) срочного рынка
    if run_all or "margins" in only_set:
        p = REFERENCE_DIR / "margin_rates.parquet"
        codes = margin_asset_codes()
        starts = {code: args.from_date for code in codes}
        old_df = pl.read_parquet(p) if p.exists() and args.update else None
        if old_df is not None:
            # Докачка у каждого актива со своей даты: новый рынок реестра
            # получит всю историю, а не только последние дни.
            last = old_df.group_by("asset_code").agg(pl.col("date").max())
            for code, last_date in last.iter_rows():
                if code in starts:
                    starts[code] = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log.info("12/12. Ставки ГО срочного рынка по %d активам (по %s)...", len(codes), till_date)
        t0 = time.time()
        try:
            new_df = fetch_margin_rates(starts, till_date)
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.error("  ✗ margin_rates: ошибка загрузки (%s), файл не изменён", exc)
        else:
            if len(new_df) == 0:
                full_df = old_df
                log.info("  margin_rates: новых дней нет, файл не изменён.")
            else:
                full_df = new_df
                if old_df is not None:
                    full_df = (
                        pl.concat([old_df, new_df])
                        .unique(subset=["date", "asset_code"], keep="last")
                        .sort(["asset_code", "date"])
                    )
                full_df.write_parquet(p, compression="zstd")
                log.info("  ✓ margin_rates: %d записей за %.1fс", len(full_df), time.time() - t0)
            present = set(full_df["asset_code"].unique()) if full_df is not None else set()
            missing = sorted(set(codes) - present)
            if missing:
                log.warning("  ставок ГО нет совсем для: %s", ", ".join(missing))

        # Снимок текущего ГО всегда «на сейчас», --till-date на него не влияет.
        # Снимки копятся: по ним видно, стабилен ли коэффициент ГО к mr1.
        p = REFERENCE_DIR / "margin_snapshot.parquet"
        log.info("12/12. Снимок текущего ГО по всем фьючерсам и коэффициент к mr1...")
        t0 = time.time()
        try:
            snap_df = fetch_margin_snapshot(datetime.now(ZoneInfo("Europe/Moscow")).date())
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.error("  ✗ margin_snapshot: ошибка загрузки (%s), файл не изменён", exc)
        else:
            full_df = snap_df
            if p.exists():
                full_df = (
                    pl.concat([pl.read_parquet(p), snap_df], how="vertical_relaxed")
                    .unique(subset=["snapshot_date", "secid"], keep="last")
                    .sort(["snapshot_date", "asset_code", "last_trade_date", "secid"])
                )
            full_df.write_parquet(p, compression="zstd")
            with_factor = snap_df.filter(pl.col("margin_factor").is_not_null()).height
            log.info(
                "  ✓ margin_snapshot: %d контрактов (с коэффициентом %d), всего снимков %d, за %.1fс",
                snap_df.height,
                with_factor,
                full_df["snapshot_date"].n_unique(),
                time.time() - t0,
            )

    # 13. Курсы клиринга MOEX для стоимости шага долларовых контрактов
    if run_all or "clearing_fx" in only_set:
        p = REFERENCE_DIR / "rates_moex_clearing.parquet"
        from_d = get_update_start_date(p, args.from_date) if args.update else args.from_date
        if from_d > till_date:
            log.info("13/13. rates_moex_clearing: актуален, пропускаем.")
        else:
            log.info("13/13. Курсы клиринга MOEX %s (с %s по %s)...", ", ".join(CLEARING_FX_PAIRS), from_d, till_date)
            t0 = time.time()
            try:
                new_df = fetch_clearing_fx_rates(from_d, till_date)
            except (requests.RequestException, ValueError, KeyError) as exc:
                log.error("  ✗ rates_moex_clearing: ошибка загрузки (%s), файл не изменён", exc)
            else:
                if len(new_df) == 0:
                    log.info("  rates_moex_clearing: новых дней нет, файл не изменён.")
                else:
                    full_df = new_df
                    if p.exists() and args.update:
                        full_df = (
                            pl.concat([pl.read_parquet(p), new_df])
                            .unique(subset=["date", "clearing", "pair"], keep="last")
                            .sort(["pair", "fixing_time"])
                        )
                    full_df.write_parquet(p, compression="zstd")
                    log.info("  ✓ rates_moex_clearing: %d записей за %.1fс", len(full_df), time.time() - t0)

    log.info("✅ Обработка справочных данных завершена!")


if __name__ == "__main__":
    main()
