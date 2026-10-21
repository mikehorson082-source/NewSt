"""FUTOI: три идеи эксперта — размер позиции, признаки модели, недельный оверлей.

Продолжение check_futoi.py: по третям FUTOI все сделки в плюсе, поэтому как
фильтр FUTOI бесполезен. Эксперт предложил использовать его иначе:
1. Размер позиции. Брать все сделки, но рисковать больше, когда толпа за
   сделку, и меньше — когда против.
2. Признаки модели. Добавить FUTOI к CatBoost и линейной модели шага 6.
3. Недельный оверлей. Брать только сделки по направлению недельного потока
   физлиц (эксперт сам ждёт, что сумма R упадёт).

Признаки FUTOI (check_futoi.py; по направлению сделки, без сдвига — как при
подписке; на закрытии сигнальной свечи):
- `fiz_week` — изменение доли физлиц за неделю, п.п. (нетто-изменение);
- `fiz_day` — то же за сутки (дневной поток);
- `fiz_people` — доля физлиц-участников на стороне сделки, %. Пример: в
  лонге юаня 15 798 человек, в шорте 6 851 — для лонга 70%, для шорта 30%.

Правила (записаны до запуска):
1. Размер. Риск сделки ×1.5, если физлица за неделю сдвинулись в сторону
   сделки (`fiz_week` > 0), ×0.5 — если против, ×1 — если данных нет.
   Портфель 5.5–5.6 на всей истории. Размер не меняет R отдельной сделки,
   поэтому смотрим Sharpe портфеля и сумму R с весами (R × множитель).
   Пустышка: те же множители, случайно перемешанные между сигналами.
   Принимаем, если Sharpe выше 95% пустышек и выше, чем без множителей.
2. Модель. Признаки шага 6 + `fiz_day`, `fiz_week`, `fiz_people`; схема
   catboost_filter.py: walk-forward 2024–2026, пропуск худшей трети прогноза,
   пустышка — столько же случайно пропущенных сигналов по годам. Принимаем,
   если Sharpe выше 95% пустышек, не ниже, чем без модели, и не ниже, чем у
   той же модели без FUTOI.
3. Оверлей. Пропускаем сигналы, против которых физлица за неделю сдвинулись
   (`fiz_week` < 0); без данных — берём. Вся история. Пустышка — столько же
   случайно пропущенных сигналов по годам. Принимаем, если Sharpe выше 95%
   пустышек и не ниже, чем без оверлея.

Проверок четыре (размер, оверлей, две модели). Даже если FUTOI бесполезен,
хотя бы одна из четырёх пройдёт планку 95% случайно примерно в 19% случаев.
Поэтому одна прошедшая — ещё не находка: её надо подтвердить в paper mode.

Ничего не записывает. Запуск (CatBoost — в группе ml), около 20 минут:
  uv run --group ml python strategies/st01/check_futoi_expert.py
Быстрый черновой прогон — с меньшей пустышкой: --dummy-runs 50.
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from catboost_filter import (
    CENTRAL_SHARE,
    DUMMY_RUNS,
    FITTERS,
    MODELS,
    START,
    Dummy,
    importance_table,
    kept,
    random_skip,
    run_dummy,
    run_signals,
    skipped_by_year,
    total_r,
    walk_forward,
)
from check_futoi import FUTOI_DIR, FUTOI_TICKERS, retail_share, with_retail
from entry_features import FEATURES, SIGNAL_COLUMNS, build_dataset
from run_baseline import CAPITAL
from run_portfolio import RISK_FRACTIONS, RULES

from backtest.portfolio import MarketData, load_markets, run_multi_portfolio
from backtest.report import Report, build_report, format_value, market_timeline

FUTOI_FEATURES = ("fiz_day", "fiz_week", "fiz_people")
MODEL_FEATURES = (*FEATURES, *FUTOI_FEATURES)
WITH_CROWD = 1.5  # множитель риска, когда физлица за неделю шли в сторону сделки
AGAINST_CROWD = 0.5
ACCEPT_PERCENTILE = 95.0
SEED = 21


# ---------------------------------------------------------------- признаки


def with_futoi(dataset: pl.DataFrame) -> pl.DataFrame:
    """Набор сигналов (`build_dataset`) с признаками `FUTOI_FEATURES`."""
    frames = []
    for name in dataset["market"].unique(maintain_order=True):
        shares = retail_share(
            pl.read_parquet(FUTOI_DIR / f"{FUTOI_TICKERS[name]}.parquet")
        )
        part = dataset.filter(pl.col("market") == name)
        frames.append(
            with_retail(part, shares, time_column="timestamp").select(
                *dataset.columns, *FUTOI_FEATURES
            )
        )
    return pl.concat(frames).sort("timestamp", "market")


# ------------------------------------------------------------------ правила


def crowd_scale(signals: pl.DataFrame) -> pl.DataFrame:
    """Идея 1: `risk_scale` по недельному потоку физлиц.

    Пример: лонг юаня, за неделю доля физлиц выросла на 3 п.п. — риск ×1.5;
    упала — ×0.5; данных FUTOI нет или изменение ровно 0 — ×1.
    """
    week = pl.col("fiz_week")
    return signals.with_columns(
        pl.when(week > 0)
        .then(WITH_CROWD)
        .when(week < 0)
        .then(AGAINST_CROWD)
        .otherwise(1.0)
        .alias("risk_scale")
    )


def with_crowd_only(signals: pl.DataFrame) -> pl.DataFrame:
    """Идея 3: без сигналов, против которых физлица за неделю сдвинулись."""
    return signals.filter(~(pl.col("fiz_week") < 0).fill_null(False))


def skipped_counts(signals: pl.DataFrame, chosen: pl.DataFrame) -> dict[int, int]:
    """Сколько сигналов правило пропустило в каждом году."""
    year = pl.col("timestamp").dt.year()
    before = dict(signals.group_by(year).len().iter_rows())
    after = dict(chosen.group_by(year).len().iter_rows())
    return {int(y): n - after.get(y, 0) for y, n in before.items()}


def shuffled_scale(signals: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Пустышка к размеру: те же множители, случайно розданные сигналам."""
    return signals.with_columns(
        pl.Series("risk_scale", rng.permutation(signals["risk_scale"].to_numpy()))
    )


# ------------------------------------------------------------------ портфель


def run_history(
    signals: pl.DataFrame, markets: dict[str, MarketData], timeline: pl.Series
) -> Report:
    """Портфель 5.5–5.6 на всей истории; `risk_scale` — если есть у сигналов."""
    columns = ["market", *SIGNAL_COLUMNS]
    if "risk_scale" in signals.columns:
        columns.append("risk_scale")
    result = run_multi_portfolio(signals.select(columns), markets, CAPITAL, RULES)
    return build_report(result, timeline)


def weighted_r(report: Report, signals: pl.DataFrame) -> float:
    """Сумма R × множитель риска по закрытым сделкам (без множителя — сумма R)."""
    trades = report.trades.filter(pl.col("exit_reason") != "end")
    if "risk_scale" not in signals.columns:
        return float(trades["r_multiple"].sum())
    scales = signals.select(
        "market",
        pl.col("timestamp").cast(trades.schema["signal_time"]).alias("signal_time"),
        "risk_scale",
    )
    joined = trades.join(scales, on=["market", "signal_time"], how="left")
    return float((joined["r_multiple"] * joined["risk_scale"]).sum())


def summary_row(label: str, report: Report, signals: pl.DataFrame) -> dict[str, Any]:
    s = report.summary
    return {
        "вариант": label,
        "сделок": s["trades"],
        "Sharpe": f"{s['sharpe']:.2f} ± {s['sharpe_se']:.2f}",
        "CAGR": format_value("cagr", s["cagr"]),
        "Max DD": format_value("max_drawdown", s["max_drawdown"]),
        "сумма R": round(total_r(report), 1),
        "сумма R × множитель": round(weighted_r(report, signals), 1),
    }


def history_dummy(
    make: Any, markets: dict[str, MarketData], timeline: pl.Series, runs: int, seed: int
) -> Dummy:
    """Пустышка на всей истории: `make(rng)` — случайный набор сигналов."""
    rng = np.random.default_rng(seed)
    sharpe, expectancy, totals = [], [], []
    for i in range(runs):
        signals = make(rng)
        report = run_history(signals, markets, timeline)
        sharpe.append(report.summary["sharpe"])
        expectancy.append(report.summary["expectancy_r"])
        totals.append(weighted_r(report, signals))
        if (i + 1) % 50 == 0:
            print(f"  пустышка: {i + 1} из {runs}", flush=True)
    return Dummy(np.array(sharpe), np.array(expectancy), np.array(totals))


def walk_forward_report(
    dataset: pl.DataFrame, fitter: Any, markets: dict[str, MarketData]
) -> Report:
    """Модель шага 6 без FUTOI: портфель на сигналах, которые она берёт."""
    run = walk_forward(dataset, fitter)
    return run_signals(kept(run.predictions, CENTRAL_SHARE), markets)


def decision(label: str, sharpe: float, dummy: Dummy, rivals: dict[str, float]) -> str:
    """Строка решения: Sharpe выше 95% пустышек и не ниже каждого соперника."""
    bound = float(np.percentile(dummy.sharpe, ACCEPT_PERCENTILE))
    smarter = sharpe > bound
    parts = [
        (
            f"{label}: Sharpe {sharpe:.2f}; лучше {dummy.beaten(sharpe):.1f}% "
            f"пустышек (95-я перцентиль {bound:.2f}) — {'да' if smarter else 'нет'}"
        )
    ]
    worth = True
    for name, value in rivals.items():
        ok = sharpe >= value
        worth &= ok
        parts.append(f"не ниже, чем {name} ({value:.2f}) — {'да' if ok else 'нет'}")
    verdict = "ПРИНЯТ" if smarter and worth else "НЕ ПРИНЯТ"
    return "; ".join(parts) + f". {verdict}."


def dummy_row(label: str, dummy: Dummy) -> dict[str, Any]:
    return {
        "пустышка": label,
        "Sharpe: медиана": round(float(np.median(dummy.sharpe)), 2),
        "Sharpe: 95-я перцентиль": round(
            float(np.percentile(dummy.sharpe, ACCEPT_PERCENTILE)), 2
        ),
        "сумма R × множитель: медиана": round(float(np.median(dummy.total_r)), 1),
    }


# ---------------------------------------------------------------------- запуск


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dummy-runs",
        type=int,
        default=DUMMY_RUNS,
        help=f"Прогонов пустышки на каждую проверку (по умолчанию {DUMMY_RUNS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    markets = load_markets(RISK_FRACTIONS)
    timeline = market_timeline(data.bars for data in markets.values())
    dataset = with_futoi(build_dataset(markets))
    week = dataset["fiz_week"]
    print(
        f"\nСигналов: {dataset.height} ({', '.join(markets)}). Физлица за неделю "
        f"в сторону сигнала: {(week > 0).sum()}, против: {(week < 0).sum()}, "
        f"без данных: {week.null_count()}."
    )

    # 1 и 3 — вся история, правила без обучения.
    base = run_history(dataset, markets, timeline)
    scaled_signals = crowd_scale(dataset)
    scaled = run_history(scaled_signals, markets, timeline)
    overlay_signals = with_crowd_only(dataset)
    overlay = run_history(overlay_signals, markets, timeline)
    print("\nИдеи 1 и 3 — портфель 5.5–5.6, вся история:")
    print(
        pl.DataFrame(
            [
                summary_row("без FUTOI", base, dataset),
                summary_row("1. размер ×1.5 / ×0.5", scaled, scaled_signals),
                summary_row("3. только по потоку физлиц", overlay, overlay_signals),
            ]
        )
    )
    taken = scaled.trades.join(
        scaled_signals.select(
            "market",
            pl.col("timestamp")
            .cast(scaled.trades.schema["signal_time"])
            .alias("signal_time"),
            "risk_scale",
        ),
        on=["market", "signal_time"],
    )
    print(
        f"Средний множитель риска у сделок идеи 1: {taken['risk_scale'].mean():.2f} "
        "(около 1 — риск в среднем тот же, сравнение честное)."
    )

    print("\nПустышка к идее 1 (множители перемешаны):")
    size_dummy = history_dummy(
        partial(shuffled_scale, scaled_signals),
        markets,
        timeline,
        args.dummy_runs,
        SEED,
    )
    counts = skipped_counts(dataset, overlay_signals)
    print("Пустышка к идее 3 (столько же случайных пропусков по годам):")
    overlay_dummy = history_dummy(
        partial(random_skip, dataset, counts),
        markets,
        timeline,
        args.dummy_runs,
        SEED + 1,
    )

    # 2 — модели, walk-forward 2024–2026.
    test = dataset.filter(pl.col("timestamp").dt.date() >= START)
    model_base = run_signals(test, markets)
    model_rows = [summary_row("без модели", model_base, test)]
    model_lines = []
    importances = {}
    for i, (key, fitter) in enumerate(FITTERS.items()):
        plain = walk_forward_report(dataset, fitter, markets)
        futoi_run = walk_forward(dataset, partial(fitter, features=MODEL_FEATURES))
        chosen = kept(futoi_run.predictions, CENTRAL_SHARE)
        futoi_report = run_signals(chosen, markets)
        model_rows.append(summary_row(f"{MODELS[key]} без FUTOI", plain, test))
        model_rows.append(summary_row(f"{MODELS[key]} + FUTOI", futoi_report, chosen))
        dummy = run_dummy(
            test,
            markets,
            skipped_by_year(futoi_run.predictions, CENTRAL_SHARE),
            args.dummy_runs,
            SEED + 100 * (i + 1),
            f"{MODELS[key]} + FUTOI",
        )
        model_lines.append(
            decision(
                f"{MODELS[key]} + FUTOI",
                futoi_report.summary["sharpe"],
                dummy,
                {
                    "без модели": model_base.summary["sharpe"],
                    f"{MODELS[key]} без FUTOI": plain.summary["sharpe"],
                },
            )
        )
        importances[key] = importance_table(futoi_run)

    print(f"\nИдея 2 — модели, сигналы с {START:%d.%m.%Y}, пропуск худшей трети:")
    print(pl.DataFrame(model_rows))

    print("\nПустышки:")
    print(
        pl.DataFrame(
            [
                dummy_row("к идее 1", size_dummy),
                dummy_row("к идее 3", overlay_dummy),
            ]
        )
    )

    print(
        f"\nРешение (Sharpe выше {ACCEPT_PERCENTILE:g}% пустышек и не ниже соперников):"
    )
    print(
        "- "
        + decision(
            "1. размер",
            scaled.summary["sharpe"],
            size_dummy,
            {"без FUTOI": base.summary["sharpe"]},
        )
    )
    print(
        "- "
        + decision(
            "3. оверлей",
            overlay.summary["sharpe"],
            overlay_dummy,
            {"без FUTOI": base.summary["sharpe"]},
        )
    )
    for line in model_lines:
        print("- " + line)

    print("\nДля сведения: важность признаков моделей с FUTOI (FUTOI — fiz_*).")
    for key, table in importances.items():
        print(f"\n{MODELS[key]}:")
        print(table)


if __name__ == "__main__":
    main()
