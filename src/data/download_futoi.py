"""Загрузка данных FUTOI (открытые позиции физлиц и юрлиц) с MOEX Algopack.

FUTOI (Futures Open Interest) публикуется Московской биржей в виде 5-минутных срезов
с разбивкой по категориям участников рынка:
  - FIZ (физические лица)
  - YUR (юридические лица / институционалы)

Особенности FUTOI:
  - Данные агрегированы по БАЗОВОМУ АКТИВУ (Si, BR, RI, GD, CR...), а не по отдельным квартальным сериям.
    Поэтому ряд непрерывен с 2020 года и не требует склейки контрактов!
  - 5-минутные снимки с 10:00 (или 09:00) до 23:50.
  - Поля: tradedate, tradetime, clgroup, pos (нетто), pos_long, pos_short, pos_long_num, pos_short_num.
  - Сервер MOEX жестко ограничивает выдачу 1000 строками на запрос и игнорирует параметр start.
    Данный скрипт запрашивает данные чанками по 2 торговых дня (~700 строк), гарантируя 100% отсутствие усечения данных.

Сохраняет два слоя:
  1. 5-минутный детальный срез: data/raw/futoi/5min/{ticker}.parquet
  2. Дневной агрегат (EOD Pivot): data/raw/futoi/daily/{ticker}.parquet
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib3

# Отключаем предупреждения об отсутствии корпоративных сертификатов Минцифры
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Гарантия импорта src независимо от способа вызова
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
import duckdb
import polars as pl
import requests

from src.core.config import MAX_RETRIES, REQUEST_DELAY, REQUEST_TIMEOUT, Paths

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Базовый URL для аналитических продуктов Algopack
FUTOI_BASE_URL = "https://iss.moex.com/iss/analyticalproducts/futoi"

# Основные 26 ликвидных тикеров под активы нашего проекта
DEFAULT_FUTOI_TICKERS = [
    "Si", "BR", "CR", "Eu", "ED", "GD", "SV", "PT", "PD", "CE", "CC",
    "RI", "MX", "MM", "SR", "GZ", "LK", "YD", "NG", "SF", "NA",
    "CNYRUBF", "EURRUBF", "GLDRUBF", "IMOEXF", "USDRUBF"
]

# Дата начала публикации FUTOI на Московской бирже
FUTOI_START_DATE = "2020-01-03"


def get_auth_headers() -> dict[str, str]:
    """Возвращает заголовки авторизации с токеном из .env."""
    token = os.getenv("ALGOPACK_API", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def get_trading_dates(from_date: str, till_date: str) -> list[str]:
    """Получает список фактических торговых дней из parquet-файлов futures_daily."""
    daily_pattern = str(Paths().raw_futures_daily / "*" / "*.parquet")
    try:
        con = duckdb.connect()
        query = f"""
            SELECT DISTINCT CAST(TRADEDATE AS VARCHAR) AS td
            FROM read_parquet('{daily_pattern}')
            WHERE CAST(TRADEDATE AS VARCHAR) >= '{from_date}'
              AND CAST(TRADEDATE AS VARCHAR) <= '{till_date}'
            ORDER BY td
        """
        rows = con.execute(query).fetchall()
        con.close()
        if rows:
            return [r[0] for r in rows]
    except Exception as exc:
        log.warning("Не удалось прочитать торговые дни из futures_daily (%s), генерируем рабочие дни", exc)

    # Запасной вариант: генерация рабочих дней (пн-пт)
    cur = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(till_date, "%Y-%m-%d").date()
    dates: list[str] = []
    while cur <= end:
        if cur.weekday() < 5:  # Пн-Пт
            dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def fetch_futoi_chunk(
    ticker: str,
    from_d: str,
    till_d: str,
    session: requests.Session,
    headers: dict[str, str],
) -> tuple[list[list[Any]], list[str], bool]:
    """Запрашивает данные FUTOI за диапазон дат from_d..till_d.

    Возвращает (rows, columns, is_cutoff_reached). Сбой (сеть, HTTP не 200) —
    исключение, а не пустой ответ: иначе тикер сохранился бы с дырой в эти дни,
    и следующий --update начал бы уже после неё.
    """
    url = f"{FUTOI_BASE_URL}/securities/{ticker}.json"
    params = {"from": from_d, "till": till_d}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT, verify=False)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                log.error("Тикер %s [%s..%s]: ошибка сети: %s", ticker, from_d, till_d, exc)
                raise
            time.sleep(REQUEST_DELAY * (2**attempt))
            continue
        if resp.status_code == 200:
            payload = resp.json()
            futoi_block = payload.get("futoi", {})
            cols = futoi_block.get("columns", [])
            rows = futoi_block.get("data", [])

            # Проверка на ошибку ограничения бесплатного доступа (T-14 дней)
            if cols == ["ERROR_MESSAGE"] or (rows and len(rows) == 1 and isinstance(rows[0][0], str) and "Free users" in rows[0][0]):
                return [], cols, True

            return rows, cols, False
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            time.sleep(REQUEST_DELAY * (2**attempt))
            continue
        raise requests.HTTPError(f"Тикер {ticker} [{from_d}..{till_d}]: HTTP {resp.status_code}", response=resp)

    raise AssertionError("unreachable")  # каждая попытка либо вернула ответ, либо продолжила, либо бросила


def download_ticker_futoi(
    ticker: str,
    dates: list[str],
    workers: int = 6,
) -> pl.DataFrame | None:
    """Загружает все данные FUTOI для заданного тикера за указанные торговые даты."""
    headers = get_auth_headers()
    session = requests.Session()
    session.headers.update(headers)

    # Формируем пары дат (по 2 торговых дня на запрос)
    date_pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(dates):
        if i + 1 < len(dates):
            date_pairs.append((dates[i], dates[i + 1]))
            i += 2
        else:
            date_pairs.append((dates[i], dates[i]))
            i += 1

    all_rows: list[list[Any]] = []
    col_names: list[str] = []

    def process_pair(pair: tuple[str, str]) -> list[list[Any]]:
        d1, d2 = pair
        # Локальная сессия для каждого потока
        local_s = requests.Session()
        local_s.headers.update(headers)
        rows, cols, is_cutoff = fetch_futoi_chunk(ticker, d1, d2, local_s, headers)
        if is_cutoff:
            return []

        # Если вернулось ровно 1000 строк (лимит MOEX), разбиваем на отдельные дни
        if len(rows) >= 1000 and d1 != d2:
            r1, _, _ = fetch_futoi_chunk(ticker, d1, d1, local_s, headers)
            r2, _, _ = fetch_futoi_chunk(ticker, d2, d2, local_s, headers)
            return r1 + r2

        nonlocal col_names
        if cols and not col_names and cols != ["ERROR_MESSAGE"]:
            col_names = [c.lower() for c in cols]

        return rows

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_pair, pair): pair for pair in date_pairs}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                all_rows.extend(res)

    if not all_rows:
        return None

    if not col_names:
        col_names = [
            "sess_id", "seqnum", "tradedate", "tradetime", "ticker",
            "clgroup", "pos", "pos_long", "pos_short", "pos_long_num",
            "pos_short_num", "systime", "trade_session_date"
        ]

    # Строим Polars DataFrame
    df = pl.DataFrame(all_rows, schema=col_names[:len(all_rows[0])], orient="row")

    # Нормализация типов
    df = (
        df.with_columns(
            timestamp=pl.concat_str([pl.col("tradedate"), pl.lit(" "), pl.col("tradetime")]).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            tradedate=pl.col("tradedate").str.to_date("%Y-%m-%d"),
            ticker=pl.lit(ticker),
            pos=pl.col("pos").cast(pl.Int64),
            pos_long=pl.col("pos_long").cast(pl.Int64),
            pos_short=pl.col("pos_short").cast(pl.Int64),
            pos_long_num=pl.col("pos_long_num").cast(pl.Int64),
            pos_short_num=pl.col("pos_short_num").cast(pl.Int64),
        )
        .unique(subset=["tradedate", "tradetime", "clgroup"])
        .sort(["timestamp", "clgroup"])
    )

    return df


def build_daily_pivot(df_5min: pl.DataFrame, ticker: str) -> pl.DataFrame:
    """Формирует дневной агрегат (EOD Pivot) из 5-минутных данных."""
    # Берем последний срез каждого дня для каждой группы (EOD snapshot)
    eod = (
        df_5min.sort(["tradedate", "tradetime"])
        .group_by(["tradedate", "clgroup"])
        .last()
    )

    # Разворачиваем FIZ / YUR в широкую таблицу
    pivot = eod.pivot(
        index="tradedate",
        on="clgroup",
        values=["pos", "pos_long", "pos_short", "pos_long_num", "pos_short_num"],
        aggregate_function="first",
    )

    # Рассчитываем производные метрики открытого интереса и соотношения участников
    daily = (
        pivot.with_columns(
            ticker=pl.lit(ticker),
            # Суммарный открытый интерес в контрактах (long_FIZ + long_YUR)
            oi_total=(pl.col("pos_long_FIZ") + pl.col("pos_long_YUR")),
        )
        .with_columns(
            # Доля чистой позиции физлиц относительно суммарного ОИ
            fiz_net_ratio=pl.when(pl.col("oi_total") > 0)
            .then(pl.col("pos_FIZ") / pl.col("oi_total"))
            .otherwise(0.0),
            # Доля чистой позиции юрлиц относительно суммарного ОИ
            yur_net_ratio=pl.when(pl.col("oi_total") > 0)
            .then(pl.col("pos_YUR") / pl.col("oi_total"))
            .otherwise(0.0),
            # Доля лонгистов среди физлиц по количеству счетов
            fiz_long_num_ratio=pl.when((pl.col("pos_long_num_FIZ") + pl.col("pos_short_num_FIZ")) > 0)
            .then(pl.col("pos_long_num_FIZ") / (pl.col("pos_long_num_FIZ") + pl.col("pos_short_num_FIZ")))
            .otherwise(0.0),
            # Доля лонгистов среди юрлиц по количеству счетов
            yur_long_num_ratio=pl.when((pl.col("pos_long_num_YUR") + pl.col("pos_short_num_YUR")) > 0)
            .then(pl.col("pos_long_num_YUR") / (pl.col("pos_long_num_YUR") + pl.col("pos_short_num_YUR")))
            .otherwise(0.0),
        )
        .sort("tradedate")
    )

    return daily


def update_ticker(
    idx: int,
    total: int,
    ticker: str,
    paths: Paths,
    args: argparse.Namespace,
    from_date: str,
    till_date: str,
) -> int:
    """Скачивает или докачивает FUTOI одного тикера: 1 — файл обновлён, 0 — нечего обновлять.

    Сбой загрузки — исключение до записи: файл тикера не меняется.
    """
    file_5min = paths.raw_futoi_5min / f"{ticker}.parquet"
    file_daily = paths.raw_futoi_daily / f"{ticker}.parquet"

    current_from = from_date
    existing_5min: pl.DataFrame | None = None

    if file_5min.exists() and not args.force:
        if not args.update:
            log.info("[%d/%d] Тикер %s: уже существует, пропускаем (используйте --update для докачки)", idx, total, ticker)
            return 0
        else:
            existing_5min = pl.read_parquet(file_5min)
            max_dt = existing_5min["tradedate"].max()
            if max_dt:
                next_day = max_dt + timedelta(days=1)
                current_from = next_day.strftime("%Y-%m-%d")
                if current_from > till_date:
                    log.info("[%d/%d] Тикер %s: актуален (последняя дата %s)", idx, total, ticker, max_dt)
                    return 0

    log.info("[%d/%d] Тикер %s: получение дат с %s по %s...", idx, total, ticker, current_from, till_date)
    dates = get_trading_dates(current_from, till_date)
    if not dates:
        log.info("  Нет новых торговых дат для %s", ticker)
        return 0

    log.info("  Загрузка %d торговых дней (%d запросов)...", len(dates), (len(dates) + 1) // 2)
    t0 = time.time()
    new_df = download_ticker_futoi(ticker, dates, workers=args.workers)
    elapsed = time.time() - t0

    if new_df is None or len(new_df) == 0:
        log.warning("  Тикер %s: нет данных за указанный период", ticker)
        return 0

    if existing_5min is not None:
        full_5min = (
            pl.concat([existing_5min, new_df])
            .unique(subset=["tradedate", "tradetime", "clgroup"])
            .sort(["timestamp", "clgroup"])
        )
    else:
        full_5min = new_df

    # Сохранение 5-минутного среза
    full_5min.write_parquet(file_5min, compression="zstd")

    # Формирование и сохранение дневного агрегата
    daily_df = build_daily_pivot(full_5min, ticker)
    daily_df.write_parquet(file_daily, compression="zstd")

    mb_5min = file_5min.stat().st_size / (1024 * 1024)
    log.info(
        "  ✓ %s: %d строк (5min: %.2f MB, daily: %d дней) за %.1fс",
        ticker, len(full_5min), mb_5min, len(daily_df), elapsed
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка данных FUTOI с MOEX Algopack")
    parser.add_argument(
        "--tickers",
        type=str,
        default="default",
        help="Список тикеров через запятую (например, Si,BR,GD), 'default' (26 основных) или 'all'",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=FUTOI_START_DATE,
        help=f"Начальная дата (по умолчанию {FUTOI_START_DATE})",
    )
    parser.add_argument(
        "--till-date",
        type=str,
        default=None,
        help="Конечная дата (по умолчанию сегодня минус 14 дней для бесплатного тарифа)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Количество потоков загрузки (по умолчанию 6)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Докачать только новые дни для уже имеющихся файлов",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие файлы",
    )

    args = parser.parse_args()
    # Один писатель в data/ за раз: данные уже пишет пульт или другой скрипт —
    # выходим с сообщением, кто именно (src/data/pipeline/lock.py).
    from data.pipeline.lock import hold_data_lock

    hold_data_lock("download_futoi.py")
    paths = Paths()

    paths.raw_futoi_5min.mkdir(parents=True, exist_ok=True)
    paths.raw_futoi_daily.mkdir(parents=True, exist_ok=True)

    # Определение списка тикеров
    if args.tickers == "default":
        tickers = DEFAULT_FUTOI_TICKERS
    elif args.tickers == "all":
        # Получаем полный перечень из текущего среза MOEX
        s = requests.Session()
        s.headers.update(get_auth_headers())
        test_d = (date.today() - timedelta(days=15)).strftime("%Y-%m-%d")
        r = s.get(f"{FUTOI_BASE_URL}/securities.json?date={test_d}", verify=False).json()
        cols = r.get("futoi", {}).get("columns", [])
        if "ticker" in cols:
            idx = cols.index("ticker")
            tickers = sorted(list(set(row[idx] for row in r["futoi"]["data"])))
        else:
            tickers = DEFAULT_FUTOI_TICKERS
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    # Определение конечной даты (бесплатный тариф ограничен T - 14 дней)
    if args.till_date:
        till_date = args.till_date
    else:
        max_free_date = date.today() - timedelta(days=14)
        till_date = max_free_date.strftime("%Y-%m-%d")

    from_date = args.from_date
    log.info("Параметры запуска FUTOI:")
    log.info("  Тикеры (%d): %s", len(tickers), ", ".join(tickers[:10]) + ("..." if len(tickers) > 10 else ""))
    log.info("  Диапазон дат: %s -> %s", from_date, till_date)
    log.info("  Потоки: %d | Режим: %s", args.workers, "UPDATE" if args.update else ("FORCE" if args.force else "RESUME"))

    total_downloaded = 0
    errors = 0
    t_start = time.time()

    for idx, ticker in enumerate(tickers, start=1):
        try:
            total_downloaded += update_ticker(idx, len(tickers), ticker, paths, args, from_date, till_date)
        except Exception as exc:  # noqa: BLE001 — сбой одного тикера не должен останавливать остальные
            errors += 1
            log.error("  ✗ Тикер %s: %s — файл не изменён, следующий --update докачает этот же период", ticker, exc)

    total_time = time.time() - t_start
    log.info(
        "Завершена загрузка FUTOI: обновлено %d тикеров, ошибок %d, за %.1fс", total_downloaded, errors, total_time
    )
    # Код 1 при ошибках: пульт и `python -m data.pipeline` покажут, что шаг не удался.
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
