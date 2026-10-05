"""Раздел «Настройки»: то, что пульт помнит между запусками.

Пока одна настройка — порядок инструментов в «Инвентаре данных», заданный
перетаскиванием строк. Файл app/settings/instrument_order.json лежит рядом с
пультом, а не в data/ (там только рыночные данные). Файл один на Мак и
телефон, поэтому порядок везде одинаковый; изменения видно в git.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

router = APIRouter(prefix="/api/settings", tags=["settings"])

ORDER_FILE = "instrument_order.json"


@dataclass
class InstrumentOrder:
    """Порядок инструментов: id раздела → id инструментов сверху вниз.

    Инструмента нет в списке (появился в каталоге позже) — он идёт после
    упорядоченных, в порядке каталога.
    """

    sections: dict[str, list[str]]


def settings_dir(request: Request) -> Path:
    """Папка настроек приложения (в тестах — временная)."""
    return request.app.state.settings_dir


def read_order(folder: Path) -> InstrumentOrder:
    """Сохранённый порядок; файла ещё нет — пустой (порядок каталога)."""
    path = folder / ORDER_FILE
    if not path.is_file():
        return InstrumentOrder(sections={})
    try:
        raw = json.loads(path.read_text())
        return InstrumentOrder(sections={str(key): [str(item) for item in ids] for key, ids in raw["sections"].items()})
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        # Молча начинать с пустого нельзя: следующее сохранение затёрло бы файл.
        raise HTTPException(status_code=500, detail=f"{path} повреждён ({error}); поправьте или удалите его") from error


def write_order(folder: Path, order: InstrumentOrder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(order), ensure_ascii=False, indent=2) + "\n"
    # Сначала временный файл, потом подмена: оборванная запись не испортит прежний порядок.
    with tempfile.NamedTemporaryFile("w", dir=folder, suffix=".tmp", delete=False) as tmp:
        tmp.write(text)
    os.chmod(tmp.name, 0o644)  # временный файл создаётся только для владельца (0600)
    os.replace(tmp.name, folder / ORDER_FILE)


@router.get("/instrument-order")
def get_instrument_order(folder: Annotated[Path, Depends(settings_dir)]) -> InstrumentOrder:
    """Порядок инструментов по разделам."""
    return read_order(folder)


@router.patch("/instrument-order")
def patch_instrument_order(change: InstrumentOrder, folder: Annotated[Path, Depends(settings_dir)]) -> InstrumentOrder:
    """Заменяет порядок в присланных разделах, остальные не трогает.

    Так перестановка акций на телефоне не затрёт порядок фьючерсов, который
    только что поменяли на Маке.
    """
    order = read_order(folder)
    for section, ids in change.sections.items():
        order.sections[section] = list(dict.fromkeys(ids))  # без повторов, порядок первого вхождения
    write_order(folder, order)
    return order
