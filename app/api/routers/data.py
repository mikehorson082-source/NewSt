"""Раздел «Данные»: инвентарь data/. Только чтение."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from data.inventory import InventoryReport, build_inventory

router = APIRouter(prefix="/api/data", tags=["data"])


def data_root(request: Request) -> Path:
    """Папка данных приложения (в тестах — временная)."""
    return request.app.state.data_root


@router.get("/status")
def data_status(root: Annotated[Path, Depends(data_root)]) -> InventoryReport:
    """Инвентарь data/: что лежит, за какие даты и насколько свежее."""
    return build_inventory(root)
