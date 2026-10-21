"""FUTOI: позиции физлиц на сигнале пробоя — есть ли «плохая треть» сделок.

Блок 1 шага 7 (plan.md). Зацепка — journal.md, раздел 6: недельное изменение
позиций физлиц по юаню против доходности следующей недели, Spearman +0.21.
Сначала диагностика без модели, как с ADX и VHF (journal.md, раздел 19):
фильтр помогает, только если находит группу сигналов, которая в среднем не
зарабатывает.

FUTOI — открытые позиции по группам клиентов каждые 5 минут
(`data/raw/futoi/5min`, все контракты актива вместе): CR — cnyrub, Si — si,
Eu — euro, GD — gold. Юрлица — зеркало физлиц (их нетто-позиция = −нетто
физлиц), поэтому смотрим только физлиц.

Доля физлиц — их нетто-позиция ÷ все открытые контракты, в процентах.
Пример: физлица держат 5.31 млн лонгов и 0.84 млн шортов юаня, всего открыто
11.08 млн контрактов — доля (5.31 − 0.84) ÷ 11.08 = +40%.

Признаки — по направлению сделки (у шорта знак обратный), на закрытии
сигнальной свечи:
- за неделю — изменение доли за 7 дней, п.п. Пример: +3 — за неделю физлица
  сдвинулись в сторону нашей сделки на 3% всех открытых контрактов;
- уровень за год — место доли среди её значений за прошлый год. 1 — физлица
  нагружены в сторону сделки сильнее, чем когда-либо за год («толпа с нами»),
  0 — против сделки сильнее всего.
Решение — по ним: так будет при платной подписке, без задержки. Те же
признаки со сдвигом 14 дней (бесплатный тариф отстаёт на 10 торговых дней) —
для сведения.

Запись FUTOI за 10:10 биржа выкладывает в 10:10:09, поэтому запись считается
известной через минуту: свеча 10:00 (закрывается в 10:15) видит запись 10:10.
FUTOI по CR — с 21.04.2022, поэтому у cnyrub уровень за год — с 21.04.2023.

Сделки — портфель paper mode (cnyrub, si, euro, gold, правила 5.5–5.6), вся
история, 1 шаг проскальзывания. Трети — по границам каждого рынка: границы
знают всю историю, для правила это подглядывание, для проверки допустимо.

Ориентир (записан до запуска): FUTOI — кандидат в блок шага 7 (схема 6.3–6.4:
walk-forward, пустышка), если вместе по четырём рынкам хотя бы у одной трети
признака без сдвига R после издержек ≤ 0. Все трети в плюсе — блок не
проверяем дальше и подписку ради FUTOI не покупаем. Треть ≤ 0 только на одном
рынке или только со сдвигом — записываем, но это уже поиск после просмотра.

Ничего не записывает. Запуск (около минуты):
  uv run python strategies/st01/check_futoi.py
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
from check_breakout import closed
from check_trend_indicators import (
    STEP6_FEATURES,
    _mean_se,
    by_market_table,
    similarity_table,
    thirds_table,
    verdict,
)
from entry_features import signal_features
from rules import frozen_breakout_signals
from run_portfolio import RISK_FRACTIONS, portfolio

from backtest.markets import BAR_LENGTH, PROJECT_ROOT
from backtest.portfolio import MarketData, load_markets

FUTOI_DIR = PROJECT_ROOT / "data" / "raw" / "futoi" / "5min"
FUTOI_TICKERS = {"cnyrub": "CR", "si": "Si", "euro": "Eu", "gold": "GD"}
PUBLISH_DELAY = timedelta(minutes=1)  # запись 10:10 выложена в 10:10:09
STALE_AFTER = timedelta(days=4)  # дольше выходных; после конца FUTOI — пусто
DAY = timedelta(days=1)
WEEK = timedelta(days=7)
YEAR = timedelta(days=365)
FREE_LAG = timedelta(days=14)  # бесплатный тариф: 10 торговых дней

CENTRAL = {
    "fiz_week": "физлица за неделю, п.п.",
    "fiz_level": "физлица: уровень за год",
}
LAGGED = {
    "fiz_week_lag": "за неделю, сдвиг 14 дн.",
    "fiz_level_lag": "уровень, сдвиг 14 дн.",
}
FEATURES = {**CENTRAL, **LAGGED}


# ------------------------------------------------------------------ FUTOI


def retail_share(futoi: pl.DataFrame) -> pl.DataFrame:
    """Доля физлиц в процентах и момент, когда она известна.

    `known_at`; `share` — нетто-позиция физлиц ÷ все открытые контракты (сумма
    лонгов физлиц и юрлиц, она же — сумма шортов), > 0 — физлица в целом в
    лонге; `people` — доля физлиц-участников в лонге, %. Пример: в лонге
    юаня 15 798 человек, в шорте 6 851 — 70%.
    """
    people = pl.col("pos_long_num") + pl.col("pos_short_num")
    fiz = futoi.filter(pl.col("clgroup") == "FIZ").select(
        "timestamp",
        "pos",
        pl.col("pos_long").alias("fiz_long"),
        pl.when(people > 0).then(100 * pl.col("pos_long_num") / people).alias("people"),
    )
    yur = futoi.filter(pl.col("clgroup") == "YUR").select(
        "timestamp", pl.col("pos_long").alias("yur_long")
    )
    open_interest = pl.col("fiz_long") + pl.col("yur_long")
    return (
        fiz.join(yur, on="timestamp")
        .filter(open_interest > 0)
        .select(
            (pl.col("timestamp") + PUBLISH_DELAY).alias("known_at"),
            (100 * pl.col("pos") / open_interest).alias("share"),
            pl.col("people").fill_null(np.nan),
        )
        .sort("known_at")
    )


def share_at(known_at: np.ndarray, shares: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Последняя доля, известная к каждому моменту; старше `STALE_AFTER` — NaN."""
    index = np.searchsorted(known_at, times, side="right") - 1
    result = np.full(len(times), np.nan)
    found = index >= 0
    fresh = found.copy()
    fresh[found] = times[found] - known_at[index[found]] <= np.timedelta64(STALE_AFTER)
    result[fresh] = shares[index[fresh]]
    return result


def year_rank(
    known_at: np.ndarray, shares: np.ndarray, times: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Место значения среди долей, известных за год до момента [время − год, время].

    Доля значений меньше нынешнего: 0 — ниже всего за год, 1 — выше всего.
    Пока у ряда нет года истории — NaN. Пример: за год 70 000 записей, 63 000
    из них ниже нынешней доли — место 0.9.
    """
    year = np.timedelta64(YEAR)
    ranks = np.full(len(times), np.nan)
    if not len(known_at):
        return ranks
    starts = np.searchsorted(known_at, times - year, side="left")
    ends = np.searchsorted(known_at, times, side="right")
    for i in np.flatnonzero((times - year >= known_at[0]) & ~np.isnan(values)):
        past = shares[starts[i] : ends[i]]
        if len(past):
            ranks[i] = np.count_nonzero(past < values[i]) / len(past)
    return ranks


def with_retail(
    trades: pl.DataFrame, shares: pl.DataFrame, time_column: str = "signal_time"
) -> pl.DataFrame:
    """Сделки одного рынка с признаками FUTOI по направлению сделки.

    Признак считается на закрытии сигнальной свечи (`time_column` — её начало),
    со сдвигом — на 14 дней раньше. У шорта изменение берётся с минусом, а
    уровень — как 1 − место: рост доли физлиц для шорта — движение против него.
    Кроме `FEATURES` — для check_futoi_expert.py: `fiz_day` — изменение доли
    за сутки, п.п.; `fiz_people` — доля физлиц-участников на стороне сделки, %
    (у шорта — 100 − доля в лонге).
    """
    known_at = shares["known_at"].to_numpy()
    values = shares["share"].to_numpy()
    people = shares["people"].to_numpy()
    closes = (pl.col(time_column).cast(pl.Datetime("us")) + BAR_LENGTH).alias("t")
    times = trades.select(closes)["t"].to_numpy()
    side = trades["side"].to_numpy()
    columns = {}
    for suffix, lag in (("", timedelta(0)), ("_lag", FREE_LAG)):
        moment = times - np.timedelta64(lag)
        now = share_at(known_at, values, moment)
        week_ago = share_at(known_at, values, moment - np.timedelta64(WEEK))
        day_ago = share_at(known_at, values, moment - np.timedelta64(DAY))
        rank = year_rank(known_at, values, moment, now)
        in_long = share_at(known_at, people, moment)
        columns[f"fiz_week{suffix}"] = side * (now - week_ago)
        columns[f"fiz_level{suffix}"] = np.where(side > 0, rank, 1 - rank)
        columns[f"fiz_day{suffix}"] = side * (now - day_ago)
        columns[f"fiz_people{suffix}"] = np.where(side > 0, in_long, 100 - in_long)
    return trades.with_columns(
        [pl.Series(name, values).fill_nan(None) for name, values in columns.items()]
    )


# ------------------------------------------------------------------ сделки


def trades_with_futoi(markets: dict[str, MarketData]) -> pl.DataFrame:
    """Закрытые сделки портфеля с признаками FUTOI и шага 6 на сигнале."""
    _, report = portfolio(markets)
    trades = closed(report.trades)
    frames = []
    for name, data in markets.items():
        shares = retail_share(
            pl.read_parquet(FUTOI_DIR / f"{FUTOI_TICKERS[name]}.parquet")
        )
        step6 = signal_features(
            frozen_breakout_signals(name, data.bars), data.bars
        ).select(
            pl.col("timestamp").cast(trades.schema["signal_time"]).alias("signal_time"),
            *STEP6_FEATURES,
        )
        market_trades = trades.filter(pl.col("market") == name).join(
            step6, on="signal_time", how="left"
        )
        frames.append(with_retail(market_trades, shares))
    return pl.concat(frames).select(
        "market",
        "signal_time",
        "side",
        "r_multiple",
        (pl.col("pnl_before_costs") / pl.col("risk")).alias("r_before"),
        *FEATURES,
        *STEP6_FEATURES,
    )


def main() -> None:
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    trades = trades_with_futoi(load_markets(RISK_FRACTIONS))
    mean, se = _mean_se(trades["r_multiple"])
    print(
        f"\nПортфель paper mode ({', '.join(RISK_FRACTIONS)}), вся история: "
        f"{trades.height} закрытых сделок, {mean:+.3f} ± {se:.3f}R на сделку."
    )
    print("Сделок с признаком по рынкам (у cnyrub уровень за год — с 21.04.2023):")
    print(
        trades.group_by("market")
        .agg(
            pl.len().alias("сделок"), *(pl.col(c).is_not_null().sum() for c in FEATURES)
        )
        .sort(
            pl.col("market").replace_strict(
                {n: i for i, n in enumerate(RISK_FRACTIONS)}
            )
        )
    )

    central = thirds_table(trades, CENTRAL)
    print("\nБез сдвига, вместе по четырём рынкам (по ним решение):")
    print(central)
    print("\nРешение (ориентир: треть с R после издержек ≤ 0):")
    print("\n".join(verdict(central, CENTRAL)))

    print("\nПо рынкам, R после издержек (для сведения; зацепка была по cnyrub):")
    print(by_market_table(trades, CENTRAL))

    lagged = thirds_table(trades, LAGGED)
    print("\nСо сдвигом 14 дней (бесплатный тариф) — для сведения:")
    print(lagged)
    print("\n".join(verdict(lagged, LAGGED)))

    print(
        "\nПохожесть на признаки шага 6 — ранговая корреляция Спирмена по сделкам "
        "(около ±1 — почти тот же признак, около 0 — новое):"
    )
    print(similarity_table(trades, CENTRAL))


if __name__ == "__main__":
    main()
