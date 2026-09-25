"""Скрипт конвертации сырых CSV-файлов MOEX ALGOPACK SuperCandles в формат Parquet.

Использует DuckDB для прямого, высокоскоростного стриминга и сжатия ZSTD.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb


def convert_csv_to_parquet(
    base_dir: Path = Path("data/raw/algopack/eq"),
    datasets: list[str] | None = None,
    years: list[int] | None = None,
) -> None:
    if datasets is None:
        datasets = ["tradestats", "orderstats", "obstats"]
    if years is None:
        years = [2020, 2021, 2022, 2023]

    con = duckdb.connect()

    total_csv_bytes = 0
    total_parquet_bytes = 0
    total_rows = 0
    start_total_time = time.time()

    print("=" * 70)
    print("НАЧАЛО КОНВЕРТАЦИИ ALGOPACK SUPERCANDLES (CSV -> PARQUET)")
    print("=" * 70)

    for dataset in datasets:
        target_dir = base_dir / dataset
        target_dir.mkdir(parents=True, exist_ok=True)

        for year in years:
            folder_name = f"{dataset}_{year}"
            csv_path = base_dir / folder_name / f"{folder_name}.csv"
            parquet_path = target_dir / f"{folder_name}.parquet"

            if not csv_path.exists():
                print(f"[ПРОПУСК] Файл не найден: {csv_path}")
                continue

            csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
            total_csv_bytes += csv_path.stat().st_size

            print(f"\n[Обработка] {folder_name} (исходный CSV: {csv_size_mb:.1f} MB)...")
            t0 = time.time()

            # Конвертация через DuckDB COPY напрямую в Parquet с ZSTD компрессией
            query = f"""
                COPY (
                    SELECT * FROM read_csv('{csv_path.as_posix()}', delim=';', header=true, auto_detect=true)
                ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
            con.execute(query)

            # Проверка количества строк и размера
            row_count = con.execute(f"SELECT COUNT(*) FROM '{parquet_path.as_posix()}'").fetchone()[0]
            parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
            total_parquet_bytes += parquet_path.stat().st_size
            total_rows += row_count
            elapsed = time.time() - t0

            ratio = (parquet_size_mb / csv_size_mb) * 100 if csv_size_mb > 0 else 0
            print(
                f"  -> Готово за {elapsed:.2f} сек: {row_count:,} строк | "
                f"{parquet_size_mb:.1f} MB (сжатие до {ratio:.1f}%)"
            )

    total_time = time.time() - start_total_time
    total_csv_gb = total_csv_bytes / (1024**3)
    total_parquet_gb = total_parquet_bytes / (1024**3)
    savings_gb = total_csv_gb - total_parquet_gb

    print("\n" + "=" * 70)
    print("ИТОГИ КОНВЕРТАЦИИ:")
    print(f"  Всего строк:            {total_rows:,}")
    print(f"  Исходный объем CSV:     {total_csv_gb:.2f} ГБ")
    print(f"  Итоговый объем Parquet: {total_parquet_gb:.2f} ГБ")
    print(f"  Сэкономлено места:      {savings_gb:.2f} ГБ (-{(1 - total_parquet_gb / total_csv_gb) * 100:.1f}%)")
    print(f"  Общее время:            {total_time:.1f} сек ({total_time / 60:.2f} мин)")
    print("=" * 70)


if __name__ == "__main__":
    convert_csv_to_parquet()
