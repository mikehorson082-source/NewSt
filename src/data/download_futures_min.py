"""Загрузка 1-минутных исторических свечей по фьючерсам с MOEX ISS.

Выгружает полный минутный архив (OHLCV, begin, end) для контрактов срочного рынка
и сохраняет их в формате Parquet (data/raw/futures/{asset}/{secid}.parquet).

Особенности:
  - Максимальная глубина архива MOEX ISS (доступна с декабря 2011 года).
  - Пагинация по 500 свечей за запрос до полного исчерпания истории контракта.
  - Поддержка пропуска уже скачанных контрактов (resume).
  - Поддержка инкрементальной докачки (--update) для активных контрактов.
  - Фильтрация по конкретным активам (--assets) или отдельным контрактам (--contracts).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Гарантия импорта src независимо от способа вызова
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import polars as pl
import requests

from src.core.config import (
    ASSETS,
    DEFAULT_BOARD,
    DEFAULT_ENGINE,
    DEFAULT_MARKET,
    ISS_BASE_URL,
    MAX_RETRIES,
    PROJECT_ROOT,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    Paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Размер страницы выдачи свечей в MOEX ISS
CANDLES_PAGE_SIZE = 500


def fetch_minute_candles(
    secid: str,
    from_time: str | datetime | None = None,
    session: requests.Session | None = None,
    engine: str = DEFAULT_ENGINE,
    market: str = DEFAULT_MARKET,
    board: str = DEFAULT_BOARD,
) -> list[dict[str, Any]]:
    """Постранично выгружает все 1-минутные свечи инструмента с MOEX ISS."""
    s = session or requests.Session()
    url = f"{ISS_BASE_URL}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"

    start = 0
    all_rows: list[dict[str, Any]] = []

    while True:
        params: dict[str, Any] = {
            "interval": 1,
            "start": start,
            "iss.meta": "off",
        }
        if from_time is not None:
            params["from"] = str(from_time)

        data = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = s.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    payload = resp.json()
                    data = payload.get("candles", {})
                    break
                elif resp.status_code in (429, 502, 503, 504):
                    time.sleep(REQUEST_DELAY * (2**attempt))
                else:
                    log.warning("SECID %s: HTTP %s на start=%s", secid, resp.status_code, start)
                    break
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    log.error("SECID %s: исчерпаны попытки (%s)", secid, exc)
                    raise
                time.sleep(REQUEST_DELAY * (2**attempt))

        if not data:
            break

        columns = data.get("columns", [])
        rows = data.get("data", [])
        if not rows:
            break

        for row in rows:
            all_rows.append(dict(zip(columns, row)))

        # Если пришло меньше полного размера страницы — данные исчерпаны
        if len(rows) < CANDLES_PAGE_SIZE:
            break

        start += len(rows)

        if REQUEST_DELAY > 0:
            time.sleep(REQUEST_DELAY)

    return all_rows


def _clean_minute_df(raw_rows: list[dict[str, Any]], secid: str) -> pl.DataFrame:
    """Нормализует схему минутных свечей и парсит временные метки."""
    df = pl.DataFrame(raw_rows, infer_schema_length=None)

    # Приведение базовых числовых полей
    num_cols = {
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "value": pl.Float64,
    }
    for col_name, dtype in num_cols.items():
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).cast(dtype, strict=False))

    # Парсинг временной метки begin (время по Москве)
    if "begin" in df.columns:
        df = df.with_columns(
            pl.col("begin").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("timestamp")
        )

    # Добавляем идентификатор инструмента
    df = df.with_columns(pl.lit(secid).alias("secid"))

    if "timestamp" in df.columns:
        df = df.sort("timestamp")

    return df


def process_contract_minutes(
    asset: str,
    secid: str,
    out_dir: Path,
    overwrite: bool = False,
    update: bool = False,
) -> dict[str, Any]:
    """Скачивает или обновляет 1-минутные свечи для одного контракта."""
    asset_dir = out_dir / asset
    asset_dir.mkdir(parents=True, exist_ok=True)
    target_file = asset_dir / f"{secid}.parquet"

    # Режим инкрементального обновления
    if target_file.exists() and not overwrite:
        if update:
            try:
                df_existing = pl.read_parquet(target_file)
                if len(df_existing) > 0 and "timestamp" in df_existing.columns:
                    max_ts = df_existing["timestamp"].max()
                    if max_ts is not None:
                        # Форматируем для запроса ISS
                        from_str = max_ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(max_ts, datetime) else str(max_ts)
                        new_rows = fetch_minute_candles(secid, from_time=from_str)
                        if new_rows:
                            df_new = _clean_minute_df(new_rows, secid)
                            df_combined = (
                                pl.concat([df_existing, df_new], how="diagonal")
                                .unique(subset=["timestamp"], keep="last")
                                .sort("timestamp")
                            )
                            df_combined.write_parquet(target_file, compression="zstd")
                            added = len(df_combined) - len(df_existing)
                            return {
                                "secid": secid,
                                "asset": asset,
                                "status": "updated",
                                "rows": len(df_combined),
                                "added": added,
                            }
                        else:
                            return {"secid": secid, "asset": asset, "status": "up_to_date", "rows": len(df_existing)}
            except Exception as exc:
                log.warning("Ошибка при обновлении %s (%s), качаю заново: %s", secid, asset, exc)

        # Обычный пропуск уже скачанного
        try:
            count = pl.scan_parquet(target_file).select(pl.len()).collect().item()
            return {"secid": secid, "asset": asset, "status": "skipped", "rows": count}
        except Exception:
            pass

    # Первичная загрузка
    try:
        raw_rows = fetch_minute_candles(secid)
        if not raw_rows:
            return {"secid": secid, "asset": asset, "status": "empty", "rows": 0}

        df = _clean_minute_df(raw_rows, secid)
        df.write_parquet(target_file, compression="zstd")
        return {"secid": secid, "asset": asset, "status": "downloaded", "rows": len(df)}

    except Exception as exc:
        log.error("Ошибка при выгрузке минуток %s (%s): %s", secid, asset, exc)
        return {"secid": secid, "asset": asset, "status": "error", "error": str(exc), "rows": 0}


def get_contracts_catalog(daily_dir: Path) -> dict[str, list[str]]:
    """Собирает список контрактов для загрузки на основе выгруженных дневных данных или ASSETS."""
    catalog: dict[str, set[str]] = {}

    # 1. Если уже есть дневная база — берем точно подтвержденные биржей тикеры
    if daily_dir.exists():
        for asset_dir in sorted(daily_dir.iterdir()):
            if asset_dir.is_dir() and not asset_dir.name.startswith("."):
                tickers = {f.stem for f in asset_dir.glob("*.parquet")}
                if tickers:
                    catalog.setdefault(asset_dir.name, set()).update(tickers)

    # 2. Дополняем из реестра ASSETS
    for asset_name, spec in ASSETS.items():
        catalog.setdefault(asset_name, set()).update(spec.generate_tickers())

    return {k: sorted(v) for k, v in sorted(catalog.items())}


def download_futures_minutes(
    max_workers: int = 6,
    overwrite: bool = False,
    update: bool = False,
    asset_filter: list[str] | None = None,
    contract_filter: list[str] | None = None,
) -> None:
    """Главный пайплайн загрузки минутных данных."""
    paths = Paths(root=PROJECT_ROOT / "data")
    dest_dir = paths.raw_futures_min
    daily_dir = paths.raw_futures_daily

    catalog = get_contracts_catalog(daily_dir)
    if not catalog:
        log.error("Каталог инструментов пуст")
        return

    # Фильтры
    if asset_filter:
        catalog = {k: v for k, v in catalog.items() if k in asset_filter}

    tasks = []
    for asset, tickers in catalog.items():
        for secid in tickers:
            if contract_filter and secid not in contract_filter:
                continue
            tasks.append((asset, secid))

    total_tasks = len(tasks)
    mode_str = "ПЕРЕЗАПИСЬ" if overwrite else ("ДОКАЧКА/ОБНОВЛЕНИЕ" if update else "ПРОПУСК СУЩЕСТВУЮЩИХ")
    log.info("Режим: %s | Инструментов: %s | Потоков: %s", mode_str, total_tasks, max_workers)

    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_contract_minutes, asset, secid, dest_dir, overwrite, update): (asset, secid)
            for asset, secid in tasks
        }

        done_count = 0
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            done_count += 1
            if done_count % 10 == 0 or done_count == total_tasks:
                elapsed = time.time() - t0
                log.info(
                    "Минутки: %s/%s контрактов (%.1f%%) за %.1f сек...",
                    done_count,
                    total_tasks,
                    (done_count / total_tasks) * 100,
                    elapsed,
                )

    total_time = time.time() - t0
    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    updated = sum(1 for r in results if r["status"] == "updated")
    up_to_date = sum(1 for r in results if r["status"] == "up_to_date")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    empty = sum(1 for r in results if r["status"] == "empty")
    errors = sum(1 for r in results if r["status"] == "error")
    total_bars = sum(r.get("rows", 0) for r in results)
    added_bars = sum(r.get("added", 0) for r in results)

    print("\n" + "=" * 70)
    print("ИТОГИ ВЫГРУЗКИ 1-МИНУТНЫХ СВЕЧЕЙ MOEX FORTS:")
    print(f"  Всего инструментов:            {total_tasks}")
    print(f"  Вновь скачано:                 {downloaded}")
    print(f"  Обновлено свежими минутами:    {updated} (+{added_bars:,} баров)")
    print(f"  Уже актуально / пропущено:     {up_to_date + skipped}")
    print(f"  Пустых ответов (не торговались): {empty}")
    print(f"  Ошибок:                        {errors}")
    print(f"  Всего 1-минутных баров в базе: {total_bars:,}")
    print(f"  Затраченное время:             {total_time:.1f} сек ({total_time / 60:.2f} мин)")
    print(f"  Папка сохранения:              {dest_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Выгрузка 1-минутных свечей по фьючерсам с MOEX ISS")
    parser.add_argument("--workers", type=int, default=6, help="Параллельные потоки (по умолчанию: 6)")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписать существующие файлы заново")
    parser.add_argument("--update", action="store_true", help="Инкрементально докачать новые бары для активных контрактов")
    parser.add_argument("--assets", nargs="+", help="Фильтр по активам (например: si brent gold)")
    parser.add_argument("--contracts", nargs="+", help="Фильтр по конкретным тикерам (например: SiU4 USDRUBF)")
    args = parser.parse_args()

    download_futures_minutes(
        max_workers=args.workers,
        overwrite=args.overwrite,
        update=args.update,
        asset_filter=args.assets,
        contract_filter=args.contracts,
    )
