"""Правила проверки свежести: читаются из rules.json рядом с этим модулем.

Правило — сколько торговых дней набор может отставать (allowed_lag) и почему
(why). Ключ правила — вид набора (futures_minutes, futoi_daily…) или имя
файла справочника. Исключение для рынка (instruments.cocoa.futoi_daily)
целиком заменяет общее правило: у исключения своё «почему».

Файл проверяется строго: опечатка в имени рынка или ключа, правило без
«почему», справочник без правила — ошибка с понятным текстом, а не тихое
«правило не применилось».
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.instruments import ASSETS

from .catalog import REFERENCE_FILES

RULES_FILE = Path(__file__).with_name("rules.json")

DATASET_KINDS = ("futures_minutes", "futures_daily", "continuous", "futoi_5min", "futoi_daily", "algopack")

# Справочник с правилом «торгуемых контрактов», а не «отставания».
CONTRACTS_FILE = "futures_contracts.parquet"

# Что можно переопределить для рынка: наборы рынка и справочники, которые
# режутся по asset_code рынка. Срез «индекс IMOEX» общий у mix и mix_mini —
# разные правила у одного среза дали бы разный статус в зависимости от порядка.
PER_MARKET_KEYS = (
    "futures_minutes",
    "futures_daily",
    "continuous",
    "futoi_5min",
    "futoi_daily",
    "margin_rates.parquet",
    "margin_snapshot.parquet",
    CONTRACTS_FILE,
)


class RulesError(ValueError):
    """rules.json не читается или в нём ошибка."""


@dataclass(frozen=True)
class Rule:
    """Правило набора. Оба порога None — свежесть не проверяется."""

    why: str
    allowed_lag: int | None = None  # допустимое отставание в торговых днях
    min_active_contracts: int | None = None  # только у futures_contracts.parquet


@dataclass(frozen=True)
class Rules:
    stale_after: int  # сколько торговых дней сверх допустимого набор «отстаёт», дальше «устарел»
    base: dict[str, Rule]  # вид набора или файл справочника → правило
    instruments: dict[str, dict[str, Rule]] = field(default_factory=dict)  # рынок → исключения

    def rule(self, key: str, market: str | None = None) -> Rule | None:
        """Правило набора: исключение рынка, иначе общее; нет правила — None."""
        if market is not None and key in self.instruments.get(market, {}):
            return self.instruments[market][key]
        return self.base.get(key)


def parse_rule(raw: object, where: str, key: str) -> Rule:
    if not isinstance(raw, dict):
        raise RulesError(f"{where}: ожидался объект {{\"allowed_lag\": …, \"why\": …}}")
    allowed = {"why", "allowed_lag"} | ({"min_active_contracts"} if key == CONTRACTS_FILE else set())
    unknown = set(raw) - allowed
    if unknown:
        raise RulesError(f"{where}: неизвестные поля {sorted(unknown)}; можно {sorted(allowed)}")
    why = raw.get("why")
    if not isinstance(why, str) or not why.strip():
        raise RulesError(f"{where}: нет «why» — напишите, почему правило такое")
    numbers: dict[str, int | None] = {}
    for name in ("allowed_lag", "min_active_contracts"):
        value = raw.get(name)
        # bool — подкласс int: true в JSON не должен стать «1 день».
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise RulesError(f"{where}: {name} — целое число ≥ 0 или null, а не {value!r}")
        numbers[name] = value
    return Rule(why=why.strip(), **numbers)


def parse_rules(raw: object) -> Rules:
    if not isinstance(raw, dict):
        raise RulesError("ожидался объект с разделами stale_after, datasets, reference, instruments")
    unknown = set(raw) - {"about", "stale_after", "datasets", "reference", "instruments"}
    if unknown:
        raise RulesError(f"неизвестные разделы {sorted(unknown)}")

    stale = raw.get("stale_after")
    days = stale.get("days") if isinstance(stale, dict) else None
    if isinstance(days, bool) or not isinstance(days, int) or days < 0:
        raise RulesError("stale_after: нужно {\"days\": целое ≥ 0, \"why\": …}")

    base: dict[str, Rule] = {}
    for section, expected in (
        ("datasets", DATASET_KINDS),
        ("reference", tuple(spec.file for spec in REFERENCE_FILES)),
    ):
        part = raw.get(section)
        if not isinstance(part, dict):
            raise RulesError(f"нет раздела {section}")
        missing = [key for key in expected if key not in part]
        extra = [key for key in part if key not in expected]
        if missing:
            raise RulesError(f"{section}: нет правил для {missing}")
        if extra:
            raise RulesError(f"{section}: неизвестные ключи {extra}; известны {list(expected)}")
        for key, value in part.items():
            base[key] = parse_rule(value, f"{section}.{key}", key)

    instruments: dict[str, dict[str, Rule]] = {}
    markets = raw.get("instruments", {})
    if not isinstance(markets, dict):
        raise RulesError("instruments: ожидался объект {рынок: {набор: правило}}")
    for market, overrides in markets.items():
        if market not in ASSETS:
            raise RulesError(f"instruments.{market}: нет такого рынка в src/core/instruments.py")
        if not isinstance(overrides, dict):
            raise RulesError(f"instruments.{market}: ожидался объект {{набор: правило}}")
        for key in overrides:
            if key not in PER_MARKET_KEYS:
                raise RulesError(f"instruments.{market}.{key}: для рынка можно переопределить только {list(PER_MARKET_KEYS)}")
        instruments[market] = {
            key: parse_rule(value, f"instruments.{market}.{key}", key) for key, value in overrides.items()
        }
    return Rules(stale_after=days, base=base, instruments=instruments)


def load_rules(path: Path = RULES_FILE) -> Rules:
    """Правила из файла. Читается при каждой сборке отчёта: поправил файл — обнови страницу."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RulesError(f"{path} не читается: {error}") from error
    try:
        return parse_rules(raw)
    except RulesError as error:
        raise RulesError(f"{path}: {error}") from error
