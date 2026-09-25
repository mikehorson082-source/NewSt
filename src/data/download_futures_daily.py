"""Загрузка и инкрементальное обновление дневных итогов торгов по фьючерсам с MOEX ISS.

Собирает полные исторические данные (OHLC, SETTLEPRICE, OPENPOSITION, VOLUME, VALUE, NUMTRADES)
для всех контрактов из data/raw/futures/ и расширенного каталога ликвидных активов
(Какао, Евро, EUR/USD, Мини-МИКС, Платина, Палладий, Озон, Яндекс, Т-Банк).

Поддерживает:
  - Пропуск уже скачанных контрактов (resume).
  - Инкрементальную докачку новых дней (--update) для активных контрактов.
  - Полную перезапись при необходимости (--overwrite).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
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
    DEFAULT_ENGINE,
    DEFAULT_MARKET,
    ISS_BASE_URL,
    MAX_RETRIES,
    PAGE_SIZE,
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

# Схема типов для Polars
SCHEMA_OVERRIDES = {
    "BOARDID": pl.Utf8,
    "TRADEDATE": pl.Date,
    "SECID": pl.Utf8,
    "OPEN": pl.Float64,
    "LOW": pl.Float64,
    "HIGH": pl.Float64,
    "CLOSE": pl.Float64,
    "OPENPOSITIONVALUE": pl.Float64,
    "VALUE": pl.Float64,
    "VOLUME": pl.Float64,
    "OPENPOSITION": pl.Float64,
    "SETTLEPRICE": pl.Float64,
    "SWAPRATE": pl.Float64,
    "WAPRICE": pl.Float64,
    "CHANGE": pl.Float64,
    "QTY": pl.Float64,
    "NUMTRADES": pl.Float64,
    "SHORTNAME": pl.Utf8,
    "ASSETCODE": pl.Utf8,
}

# Каталог дополнительных ликвидных инструментов Мосбиржи
MONTHS_QUARTERLY = ["H", "M", "U", "Z"]
YEARS_STD = [1, 2, 3, 4, 5, 6, 7]  # 2021-2027

ADDITIONAL_LIQUID_ASSETS: dict[str, list[str]] = {
    "cocoa": [
        "CCZ4", "CCH5", "CCK5", "CCN5", "CCU5", "CCZ5",
        "CCH6", "CCJ6", "CCM6", "CCQ6", "CCU6", "CCV6", "CCX6", "CCZ6",
        "CCG7", "CCH7",
    ],
    "euro": [f"Eu{m}{y}" for y in YEARS_STD for m in MONTHS_QUARTERLY],
    "eurusd": [f"ED{m}{y}" for y in YEARS_STD for m in MONTHS_QUARTERLY],
    "perpetual": ["EURRUBF"],
    "mix_mini": [f"MM{m}{y}" for y in YEARS_STD for m in MONTHS_QUARTERLY],
    "platinum": [f"PT{m}{y}" for y in YEARS_STD for m in MONTHS_QUARTERLY],
    "palladium": [f"PD{m}{y}" for y in YEARS_STD for m in MONTHS_QUARTERLY],
    "ozon": [f"ON{m}{y}" for y in [4, 5, 6, 7] for m in MONTHS_QUARTERLY],
    "yandex": [f"YD{m}{y}" for y in [4, 5, 6, 7] for m in MONTHS_QUARTERLY],
    "tbank": [f"TI{m}{y}" for y in [3, 4, 5, 6, 7] for m in MONTHS_QUARTERLY],
}


def fetch_contract_history(
    secid: str,
    from_date: str | date | None = None,
    session: requests.Session | None = None,
    engine: str = DEFAULT_ENGINE,
    market: str = DEFAULT_MARKET,
) -> list[dict[str, Any]]:
    """Постранично выгружает историю инструмента с MOEX ISS (опционально с даты from_date)."""
    s = session or requests.Session()
    url = f"{ISS_BASE_URL}/history/engines/{engine}/markets/{market}/securities/{secid}.json"

    start = 0
    all_rows: list[dict[str, Any]] = []

    while True:
        params: dict[str, Any] = {"start": start, "iss.meta": "off"}
        if from_date:
            params["from"] = str(from_date)

        data = None
        payload: dict[str, Any] = {}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = s.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    payload = resp.json()
                    data = payload.get("history", {})
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

        # Проверка курсора пагинации
        cursor = payload.get("history.cursor", {}).get("data", [])
        if cursor and len(cursor[0]) >= 3:
            total = cursor[0][1]
            start += len(rows)
            if start >= total:
                break
        else:
            if len(rows) < PAGE_SIZE:
                break
            start += len(rows)

        if REQUEST_DELAY > 0:
            time.sleep(REQUEST_DELAY)

    return all_rows


def _cast_and_clean_df(raw_rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Приводит типы строк к стандартной схеме."""
    df = pl.DataFrame(raw_rows, infer_schema_length=None)
    for col_name, target_type in SCHEMA_OVERRIDES.items():
        if col_name in df.columns:
            if target_type == pl.Date:
                df = df.with_columns(pl.col(col_name).str.to_date("%Y-%m-%d", strict=False))
            else:
                df = df.with_columns(pl.col(col_name).cast(target_type, strict=False))
    if "TRADEDATE" in df.columns:
        df = df.sort("TRADEDATE")
    return df


def process_single_contract(
    asset: str,
    secid: str,
    out_dir: Path,
    overwrite: bool = False,
    update: bool = False,
) -> dict[str, Any]:
    """Скачивает или инкрементально обновляет историю одного контракта."""
    asset_dir = out_dir / asset
    asset_dir.mkdir(parents=True, exist_ok=True)
    target_file = asset_dir / f"{secid}.parquet"

    # Режим инкрементальной докачки
    if target_file.exists() and not overwrite:
        if update:
            try:
                df_existing = pl.read_parquet(target_file)
                if len(df_existing) > 0 and "TRADEDATE" in df_existing.columns:
                    max_date = df_existing["TRADEDATE"].max()
                    # Если контракт завершился с обнулением ОИ более 30 дней назад — он экспирирован
                    last_oi = df_existing.filter(pl.col("OPENPOSITION").is_not_null())["OPENPOSITION"].tail(1)
                    if len(last_oi) > 0 and last_oi.item() == 0 and (date.today() - max_date).days > 30:
                        return {"secid": secid, "asset": asset, "status": "expired", "rows": len(df_existing)}

                    # Запрос новых торговых дней
                    next_date = max_date + timedelta(days=1)
                    if next_date <= date.today():
                        new_rows = fetch_contract_history(secid, from_date=next_date)
                        if new_rows:
                            df_new = _cast_and_clean_df(new_rows)
                            df_combined = (
                                pl.concat([df_existing, df_new], how="diagonal")
                                .unique(subset=["TRADEDATE"], keep="last")
                                .sort("TRADEDATE")
                            )
                            df_combined.write_parquet(target_file, compression="zstd")
                            return {
                                "secid": secid,
                                "asset": asset,
                                "status": "updated",
                                "rows": len(df_combined),
                                "added": len(df_new),
                            }
                        else:
                            return {"secid": secid, "asset": asset, "status": "up_to_date", "rows": len(df_existing)}
                    else:
                        return {"secid": secid, "asset": asset, "status": "up_to_date", "rows": len(df_existing)}
            except Exception as e:
                log.warning("Не удалось инкрементально обновить %s (%s), качаю заново", secid, e)

        # Обычный пропуск, если не флаг update
        try:
            existing_count = pl.scan_parquet(target_file).select(pl.len()).collect().item()
            return {"secid": secid, "asset": asset, "status": "skipped", "rows": existing_count}
        except Exception:
            pass

    # Первичная загрузка
    try:
        raw_rows = fetch_contract_history(secid)
        if not raw_rows:
            return {"secid": secid, "asset": asset, "status": "empty", "rows": 0}

        df = _cast_and_clean_df(raw_rows)
        df.write_parquet(target_file, compression="zstd")
        return {"secid": secid, "asset": asset, "status": "downloaded", "rows": len(df)}

    except Exception as exc:
        log.error("Ошибка при обработке %s (%s): %s", secid, asset, exc)
        return {"secid": secid, "asset": asset, "status": "error", "error": str(exc), "rows": 0}


def discover_contracts_to_download(source_dir: Path) -> dict[str, list[str]]:
    """Находит все инструменты по папкам в data/raw/futures + дополняет каталогом ликвидных."""
    contracts: dict[str, set[str]] = {}

    # 1. Из локальной папки data/raw/futures
    if source_dir.exists():
        for asset_dir in sorted(source_dir.iterdir()):
            if asset_dir.is_dir() and not asset_dir.name.startswith("."):
                tickers = {f.stem for f in asset_dir.glob("*.parquet")}
                if tickers:
                    contracts.setdefault(asset_dir.name, set()).update(tickers)

    # 2. Добавляем дополнительный каталог ликвидных активов
    for asset, tickers in ADDITIONAL_LIQUID_ASSETS.items():
        contracts.setdefault(asset, set()).update(tickers)

    return {k: sorted(v) for k, v in sorted(contracts.items())}


def download_all_futures_daily(
    max_workers: int = 8,
    overwrite: bool = False,
    update: bool = False,
    asset_filter: list[str] | None = None,
) -> None:
    """Основной пайплайн скачивания и обновления дневных данных."""
    paths = Paths(root=PROJECT_ROOT / "data")
    src_dir = paths.raw_futures_min
    dest_dir = paths.raw_futures_daily

    all_contracts = discover_contracts_to_download(src_dir)
    if not all_contracts:
        log.error("Не найдено контрактов для загрузки")
        return

    if asset_filter:
        all_contracts = {k: v for k, v in all_contracts.items() if k in asset_filter}

    tasks = []
    for asset, tickers in all_contracts.items():
        for secid in tickers:
            tasks.append((asset, secid))

    total_tasks = len(tasks)
    mode_str = "ПЕРЕЗАПИСЬ" if overwrite else ("ДОКАЧКА/ОБНОВЛЕНИЕ" if update else "ПРОПУСК СУЩЕСТВУЮЩИХ")
    log.info("Режим: %s | Инструментов: %s в %s активах", mode_str, total_tasks, len(all_contracts))

    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_single_contract, asset, secid, dest_dir, overwrite, update): (asset, secid)
            for asset, secid in tasks
        }

        done_count = 0
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            done_count += 1
            if done_count % 30 == 0 or done_count == total_tasks:
                elapsed = time.time() - t0
                log.info(
                    "Прогресс: %s/%s контрактов (%.1f%%) за %.1f сек...",
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
    expired = sum(1 for r in results if r["status"] == "expired")
    empty = sum(1 for r in results if r["status"] == "empty")
    errors = sum(1 for r in results if r["status"] == "error")
    total_rows = sum(r.get("rows", 0) for r in results)
    added_rows = sum(r.get("added", 0) for r in results)

    print("\n" + "=" * 70)
    print("ИТОГИ ВЫГРУЗКИ / ОБНОВЛЕНИЯ ДНЕВНЫХ ИТОГОВ MOEX FORTS:")
    print(f"  Всего инструментов в каталоге: {total_tasks}")
    print(f"  Вновь скачано:                 {downloaded}")
    print(f"  Обновлено свежими днями:       {updated} (+{added_rows} новых баров)")
    print(f"  Актуально (без изменений):     {up_to_date + skipped + expired}")
    print(f"  Пустых ответов (не торговались): {empty}")
    print(f"  Ошибок:                        {errors}")
    print(f"  Всего торговых дней в базе:    {total_rows:,} строк")
    print(f"  Затраченное время:             {total_time:.1f} сек ({total_time / 60:.2f} мин)")
    print(f"  Папка назначения:              {dest_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Выгрузка и обновление дневной истории фьючерсов с MOEX ISS")
    parser.add_argument("--workers", type=int, default=8, help="Количество параллельных потоков (по умолчанию: 8)")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписывать существующие файлы заново")
    parser.add_argument("--update", action="store_true", help="Инкрементально докачать новые дни для активных контрактов")
    parser.add_argument("--assets", nargs="+", help="Фильтр по активам (например: cocoa euro platinum)")
    args = parser.parse_args()

    download_all_futures_daily(
        max_workers=args.workers,
        overwrite=args.overwrite,
        update=args.update,
        asset_filter=args.assets,
    )
