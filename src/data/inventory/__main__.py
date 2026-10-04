"""Инвентарь в консоли: uv run python -m data.inventory [--problems | --json]"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .build import build_inventory
from .model import SEVERITY, DatasetStatus, InventoryReport

STATUS_TEXT = {
    "ok": "ok",
    "lagging": "ОТСТАЁТ",
    "stale": "УСТАРЕЛ",
    "unchecked": "·",
    "missing": "НЕТ",
    "error": "ОШИБКА",
}
WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


def human_size(size: int) -> str:
    if size == 0:
        return ""  # срез справочника: размера у него нет
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
    return ["    " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows]


def facts_text(row: DatasetStatus) -> str:
    return " · ".join(
        f"{fact.label.lower()} {number(fact.value) if isinstance(fact.value, int) else fact.value}"
        for fact in row.facts
    )


def print_report(report: InventoryReport, problems_only: bool = False) -> None:
    expected = report.expected_date
    datasets = {row.id: row for row in report.datasets}
    print(f"Инвентарь {report.data_root}")
    print(
        f"Ожидаемая дата: {WEEKDAYS[expected.weekday()]} {expected:%d.%m.%Y}"
        f" · проверено {report.generated_at:%d.%m.%Y %H:%M}"
    )
    for section in report.sections:
        printed_header = False
        for instrument in section.instruments:
            rows = [datasets[dataset_id] for dataset_id in instrument.datasets]
            if problems_only:
                rows = [row for row in rows if SEVERITY[row.status] > 0]
            if not rows:
                continue
            if not printed_header:
                print(f"\n=== {section.title} — {section.description}")
                printed_header = True
            print(f"\n  {instrument.title} ({instrument.subtitle})")
            table = [["статус", "набор", "уточнение", "с", "по", "отстаёт", "строк", "размер", "факты", "заметка"]]
            for row in rows:
                table.append(
                    [
                        STATUS_TEXT[row.status],
                        row.title,
                        row.description,
                        row.first_date.isoformat() if row.first_date else "",
                        row.last_date.isoformat() if row.last_date else "",
                        "" if row.lag_days is None else str(row.lag_days),
                        number(row.rows),
                        human_size(row.size_bytes),
                        facts_text(row),
                        row.note or "",
                    ]
                )
            print("\n".join(format_table(table)))
    counts = ", ".join(f"{STATUS_TEXT[status]}: {count}" for status, count in report.summary.items() if count)
    print(f"\nИтого наборов: {len(report.datasets)} ({counts})")


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
