"""Торговый календарь, расчётный цикл, разрывы, режимы и сессии.

Правило: торговые дни берутся из фактических данных, а не из системного
календаря. Дни простоя (весь март 2022 года) просто отсутствуют — их нельзя
заполнять ffill, иначе появятся фиктивные нулевые доходности.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import (
    CLOSING_AUCTION_START,
    CONTINUOUS_START,
    EVENING_SESSION_START,
    LONG_GAP_CALENDAR_DAYS,
    MAIN_SESSION_END,
    MAIN_SESSION_START,
    T1_SWITCH_DATE,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- calendar
def trading_calendar(dates) -> pd.DatetimeIndex:
    """Уникальные нормализованные даты, в которые были торги."""
    values = pd.to_datetime(pd.Series(list(dates))).dt.normalize()
    return pd.DatetimeIndex(values.unique()).sort_values()


def settlement_lag(trade_date) -> int:
    """T+2 до 31.07.2023, T+1 начиная с этой даты."""
    return 1 if pd.Timestamp(trade_date) >= T1_SWITCH_DATE else 2


def settlement_date(trade_date, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Дата расчётов по сделке, заключённой в trade_date."""
    ts = pd.Timestamp(trade_date).normalize()
    pos = calendar.searchsorted(ts)
    if pos >= len(calendar) or calendar[pos] != ts:
        raise ValueError(f"{ts.date()} нет в торговом календаре")
    target = pos + settlement_lag(ts)
    if target >= len(calendar):
        raise ValueError("Календарь кончается раньше даты расчётов — продлите его")
    return calendar[target]


def gap_features(dates, long_gap_days: int = LONG_GAP_CALENDAR_DAYS) -> pd.DataFrame:
    """Календарные дни с прошлого торгового дня и флаг долгого разрыва."""
    idx = trading_calendar(dates)
    days = pd.Series(idx, index=idx).diff().dt.days
    return pd.DataFrame(
        {"calendar_days_since_prev": days, "after_long_gap": days > long_gap_days}, index=idx
    )


# ------------------------------------------------------------------ regimes
def load_regimes(path: Path | str) -> pd.DataFrame:
    """regimes.csv: regime,start,end,end_status,fallback_end,short_ban,source,note.

    end_status: closed — дата окончания известна; open — режим действует;
    unknown — дату нужно сверить, пока используется консервативный fallback_end.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("start", "end", "fallback_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["short_ban"] = df["short_ban"].isin(["1", "true", "True"])
    unknown = df["end_status"] == "unknown"
    for name in df.loc[unknown, "regime"]:
        log.warning("Режим %s: дата окончания не сверена, беру fallback_end", name)
    df["effective_end"] = df["end"]
    df.loc[unknown, "effective_end"] = df.loc[unknown, "fallback_end"]
    return df


def regime_flags(dates, regimes: pd.DataFrame) -> pd.DataFrame:
    """Флаг каждого режима на каждую дату (открытый режим — до конца данных)."""
    idx = trading_calendar(dates)
    out = pd.DataFrame(index=idx)
    for row in regimes.itertuples(index=False):
        end = row.effective_end if pd.notna(row.effective_end) else idx.max()
        out[f"regime_{row.regime}"] = (idx >= row.start) & (idx <= end)
    short_ban = np.zeros(len(idx), dtype=bool)
    for row in regimes[regimes["short_ban"]].itertuples(index=False):
        short_ban |= out[f"regime_{row.regime}"].to_numpy()
    out["short_ban"] = short_ban
    return out


def blocked_periods(regimes: pd.DataFrame, names: list[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Периоды, которые таргеты не должны пересекать (например, простой и ограниченный режим 2022)."""
    periods = []
    for row in regimes[regimes["regime"].isin(names)].itertuples(index=False):
        end = row.effective_end if pd.notna(row.effective_end) else pd.Timestamp.max
        periods.append((row.start, end))
    return periods


# ----------------------------------------------------------------- sessions
def assign_trading_date(bar_times: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Торговый день бара: его дата, а для выходных — ближайший следующий торговый день.

    Сессии выходного дня формально входят в торговый день понедельника.
    """
    days = pd.to_datetime(bar_times).dt.normalize()
    pos = calendar.searchsorted(days.to_numpy(), side="left")
    pos = np.clip(pos, 0, len(calendar) - 1)
    assigned = pd.Series(calendar[pos], index=bar_times.index)
    return assigned.where(days <= calendar[-1])


def session_label(bar_times: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """weekend / morning / open_auction / main / close_auction / evening / other.

    Метки по часам регламента после 2022 года; в укороченные дни марта 2022
    аукцион закрытия был в 13:40–13:50, такие дни нужно размечать отдельно.
    """
    ts = pd.to_datetime(bar_times)
    hm = ts.dt.strftime("%H:%M")
    on_calendar = ts.dt.normalize().isin(calendar)
    labels = np.select(
        [
            ~on_calendar,
            hm < MAIN_SESSION_START,
            hm < CONTINUOUS_START,
            hm < CLOSING_AUCTION_START,
            hm < MAIN_SESSION_END,
            hm >= EVENING_SESSION_START,
        ],
        ["weekend", "morning", "open_auction", "main", "close_auction", "evening"],
        default="other",
    )
    return pd.Series(labels, index=bar_times.index)


def session_bounds(bars: pd.DataFrame, calendar: pd.DatetimeIndex, time_col: str = "begin") -> pd.DataFrame:
    """Фактические границы основной сессии по дням: первый/последний бар и флаг укороченного дня."""
    ts = pd.to_datetime(bars[time_col])
    labels = session_label(ts, calendar)
    main = ts[labels.isin(["open_auction", "main", "close_auction"])]
    grouped = main.groupby(main.dt.normalize())
    out = pd.DataFrame({"first_bar": grouped.min(), "last_bar": grouped.max()})
    out["main_minutes"] = (out["last_bar"] - out["first_bar"]).dt.total_seconds() / 60
    full = (pd.Timestamp(f"2000-01-01 {MAIN_SESSION_END}") - pd.Timestamp(f"2000-01-01 {MAIN_SESSION_START}"))
    out["shortened"] = out["main_minutes"] < 0.8 * full.total_seconds() / 60
    return out


def first_full_session_run(bounds: pd.DataFrame, after, min_run: int = 5) -> pd.Timestamp | None:
    """Первая дата после `after`, с которой идёт min_run подряд полных сессий.

    Так из данных выводится конец укороченного режима 2022 года.
    """
    b = bounds[bounds.index >= pd.Timestamp(after)]
    full = ~b["shortened"].to_numpy()
    for i in range(len(full) - min_run + 1):
        if full[i : i + min_run].all():
            return b.index[i]
    return None
