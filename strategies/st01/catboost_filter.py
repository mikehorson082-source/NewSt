"""Шаг 6: фильтр входа пробоя — CatBoost и линейная модель (plan.md, 6.1–6.5).

Правила 5.3 не меняются. Модель оценивает, сколько R принесёт сделка по
сигналу, и худшие сигналы пропускаются. Пример: 100 сделок по +0.24R = 24R.
Модель нашла 30 сделок по −0.10R и убрала их — осталось 70 по +0.39R = 27R.
Если же убранные были такие, как все, — потеряли 7R.

Как проверяем (записано до запуска, plan.md 6.3–6.4):
- Набор — все сигналы cnyrub, si, euro, gold с меткой «R сделки в одиночку»
  (`entry_features.py`).
- Walk-forward: модель для 2024 года учится на сигналах до 2024-го, для 2025
  — до 2025-го, для 2026 — до 2026-го. Зазор: в обучение идут только
  сигналы, чья сделка закрылась до начала проверочного года.
- Настройки моделей заданы заранее и не подбираются. Метка для обучения
  обрезана до [−2R, +3R].
- Порог — 1/3-квантиль прогнозов модели на её же обучающих сигналах; на
  проверке не пересчитывается. Соседи 1/4 и 1/2 — только для устойчивости.
- Портфель — правила 5.5–5.6 (`run_portfolio.RULES`), сигналы с 01.01.2024,
  капитал 1 000 000 ₽ на эту дату. Пропущенный сигнал убирается из таблицы
  сигналов — следующий сигнал рынка может стать сделкой.
- Пустышка: в каждом году пропускаем случайно столько же сигналов, сколько
  модель, 200 раз. Модель принимаем, если её Sharpe выше, чем у 95%
  пустышек, и не ниже, чем без модели.
- Перемешанные метки: та же схема, но R в обучении перемешан. Модель не
  должна выделяться из облака пустышки — иначе где-то подглядывание.

Печатает: набор данных, обучение по годам, главную таблицу и решение по
каждой модели, по годам, соседей, перемешанные метки, средний R по третям
прогноза, важность признаков. График — кривые капитала и облако пустышки.

Запуск (CatBoost — в группе зависимостей ml):
  uv run --group ml python strategies/st01/catboost_filter.py
  → strategies/st01/reports/catboost_filter.png
Около 5 минут: по 200 прогонов портфеля пустышки на каждую модель.
Файлы прошлых запусков не затираются: новый получает суффикс _2, _3…
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from entry_features import (
    CATEGORICAL,
    FEATURES,
    SIGNAL_COLUMNS,
    build_dataset,
    training_rows,
)
from matplotlib.figure import Figure
from run_baseline import CAPITAL
from run_portfolio import RISK_FRACTIONS, RULES, STRESS_SLIPPAGE_TICKS, stressed
from tune_breakout import REPORTS, free_path

from backtest.charts import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    SURFACE,
    _dates,
    _rub,
    _style,
)
from backtest.portfolio import MarketData, load_markets, run_multi_portfolio
from backtest.report import Report, build_report, format_value, market_timeline

TEST_YEARS = (2024, 2025, 2026)
START = date(TEST_YEARS[0], 1, 1)
SKIP_SHARES = (1 / 3, 1 / 4, 1 / 2)  # центральная — первая
CENTRAL_SHARE = SKIP_SHARES[0]
UPPER_THIRD = 2 / 3  # для разбивки по третям прогноза
LABEL_CLIP = (-2.0, 3.0)
SEED = 6

# Настройки заданы заранее (plan.md 6.3): стандартные CatBoost (глубина 6,
# 1 000 деревьев, l2 3) рассчитаны на большие данные, у нас ~1 500 сделок
# и много шума — деревья мельче, регуляризация сильнее.
CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "depth": 4,
    "learning_rate": 0.03,
    "iterations": 500,
    "l2_leaf_reg": 10,
    "random_seed": SEED,
    "verbose": 0,
    "allow_writing_files": False,
}
RIDGE_ALPHA = 100.0

DUMMY_RUNS = 200
SHUFFLE_RUNS = 5
ACCEPT_PERCENTILE = 95.0

MODELS = {"catboost": "CatBoost", "linear": "линейная"}
IMPORTANCE_TITLES = {
    "catboost": "важность признаков — среднее |SHAP| на проверке, R",
    "linear": "коэффициенты, R на одно стандартное отклонение признака",
}
NO_MODEL = "без модели"
MODEL_COLORS = {NO_MODEL: MUTED, "catboost": BLUE, "linear": ORANGE}


# ------------------------------------------------------------------ модели


@dataclass(frozen=True)
class Fit:
    """Прогнозы модели на обучении и проверке и важность признаков.

    `importance`: у CatBoost — среднее |SHAP| на проверке (в R), у линейной
    — коэффициенты стандартизованных числовых признаков (R на одно
    стандартное отклонение признака).
    """

    train_pred: np.ndarray
    test_pred: np.ndarray
    importance: dict[str, float]


Fitter = Callable[[pl.DataFrame, pl.DataFrame, np.ndarray], Fit]


def _features(rows: pl.DataFrame, features: tuple[str, ...] = FEATURES):
    """Признаки для моделей: pandas, категории — целые числа."""
    return rows.select(
        *(
            pl.col(name).cast(pl.Int64) if name in CATEGORICAL else pl.col(name)
            for name in features
        )
    ).to_pandas()


def fit_catboost(
    train: pl.DataFrame,
    test: pl.DataFrame,
    target: np.ndarray,
    features: tuple[str, ...] = FEATURES,
) -> Fit:
    from catboost import CatBoostRegressor, Pool  # группа зависимостей ml

    categorical = list(CATEGORICAL)
    train_pool = Pool(_features(train, features), target, cat_features=categorical)
    test_pool = Pool(_features(test, features), cat_features=categorical)
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool)
    shap = model.get_feature_importance(test_pool, type="ShapValues")
    importance = dict(zip(features, np.abs(shap[:, :-1]).mean(axis=0), strict=True))
    return Fit(model.predict(train_pool), model.predict(test_pool), importance)


def fit_linear(
    train: pl.DataFrame,
    test: pl.DataFrame,
    target: np.ndarray,
    features: tuple[str, ...] = FEATURES,
) -> Fit:
    """Гребневая регрессия: стандартизация по обучению, пропуски — медианой."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [name for name in features if name not in CATEGORICAL]
    prepare = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                ),
                numeric,
            ),
            # Час, которого не было в обучении, — нули во всех столбцах часа.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ]
    )
    model = make_pipeline(prepare, Ridge(alpha=RIDGE_ALPHA))
    x_train = _features(train, features)
    model.fit(x_train, target)
    coefficients = model[-1].coef_[: len(numeric)]
    importance = dict(zip(numeric, coefficients, strict=True))
    return Fit(
        model.predict(x_train), model.predict(_features(test, features)), importance
    )


FITTERS: dict[str, Fitter] = {"catboost": fit_catboost, "linear": fit_linear}


# ---------------------------------------------------------------- walk-forward


def threshold_column(share: float) -> str:
    return f"threshold_{share:.3f}"


def target(train: pl.DataFrame, rng: np.random.Generator | None = None) -> np.ndarray:
    """Метка обучения: R, обрезанный до `LABEL_CLIP`; с `rng` — перемешанный."""
    values = train["r"].clip(*LABEL_CLIP).to_numpy()
    return values if rng is None else rng.permutation(values)


@dataclass(frozen=True)
class WalkForward:
    # Сигналы проверочных лет: прогноз, пороги, сколько строк было в обучении.
    predictions: pl.DataFrame
    # Важность признаков по годам: feature, year, value.
    importance: pl.DataFrame


def walk_forward(
    dataset: pl.DataFrame, fitter: Fitter, shuffle_seed: int | None = None
) -> WalkForward:
    """Прогнозы на каждом проверочном году моделью, обученной только на прошлом.

    Пороги — квантили прогнозов модели на её обучающих сигналах: на проверке
    они не пересчитываются (ML4T гл. 19). `shuffle_seed` — метки обучения
    перемешаны (проверка на подглядывание).
    """
    rng = None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
    parts = []
    importance = []
    for year in TEST_YEARS:
        # Ряды — naive MSK, как в data/continuous_15m.
        train = training_rows(dataset, datetime(year, 1, 1))  # noqa: DTZ001
        test = dataset.filter(pl.col("timestamp").dt.year() == year)
        if train.is_empty() or test.is_empty():
            raise ValueError(f"{year}: нет сигналов для обучения или проверки")
        fit = fitter(train, test, target(train, rng))
        quantiles = {
            threshold_column(share): float(np.quantile(fit.train_pred, share))
            for share in (*SKIP_SHARES, UPPER_THIRD)
        }
        parts.append(
            test.with_columns(
                pl.lit(year).alias("year"),
                pl.Series("prediction", np.asarray(fit.test_pred, dtype=float)),
                pl.lit(train.height).alias("train_rows"),
                *(pl.lit(value).alias(name) for name, value in quantiles.items()),
            )
        )
        importance.extend(
            {"feature": name, "year": year, "value": float(value)}
            for name, value in fit.importance.items()
        )
    return WalkForward(pl.concat(parts), pl.DataFrame(importance))


def kept(predictions: pl.DataFrame, share: float) -> pl.DataFrame:
    """Сигналы, которые модель берёт: прогноз не ниже порога обучения."""
    return predictions.filter(pl.col("prediction") >= pl.col(threshold_column(share)))


def skipped_by_year(predictions: pl.DataFrame, share: float) -> dict[int, int]:
    """Сколько сигналов модель пропустила в каждом году."""
    rows = predictions.group_by("year").agg(
        (pl.col("prediction") < pl.col(threshold_column(share))).sum().alias("n")
    )
    return {int(row["year"]): int(row["n"]) for row in rows.iter_rows(named=True)}


def random_skip(
    signals: pl.DataFrame, counts: dict[int, int], rng: np.random.Generator
) -> pl.DataFrame:
    """Пустышка: в каждом году убрать случайно `counts[год]` сигналов."""
    years = signals["timestamp"].dt.year().to_numpy()
    keep = np.ones(signals.height, dtype=bool)
    for year, count in counts.items():
        rows = np.flatnonzero(years == year)
        keep[rng.choice(rows, size=count, replace=False)] = False
    return signals.filter(pl.Series(keep))


# ------------------------------------------------------------------- портфель


def run_signals(signals: pl.DataFrame, markets: dict[str, MarketData]) -> Report:
    """Портфель 5.5–5.6 на этих сигналах; кривая капитала — с `START`."""
    result = run_multi_portfolio(
        signals.select("market", *SIGNAL_COLUMNS), markets, CAPITAL, RULES
    )
    timeline = market_timeline(data.bars for data in markets.values())
    return build_report(result, timeline.filter(timeline.dt.date() >= START))


def total_r(report: Report) -> float:
    return float(
        report.trades.filter(pl.col("exit_reason") != "end")["r_multiple"].sum()
    )


def summary_row(
    label: str, report: Report, skipped: float | None, hard: Report | None = None
) -> dict[str, Any]:
    s = report.summary
    row = {
        "вариант": label,
        "пропущено сигналов, %": "—" if skipped is None else f"{skipped * 100:.1f}",
        "сделок": s["trades"],
        "expectancy, R": round(s["expectancy_r"], 3),
        "сумма, R": round(total_r(report), 1),
        "Sharpe": f"{s['sharpe']:.2f} ± {s['sharpe_se']:.2f}",
        "CAGR": format_value("cagr", s["cagr"]),
        "Max DD": format_value("max_drawdown", s["max_drawdown"]),
    }
    if hard is not None:
        row[f"Sharpe при {STRESS_SLIPPAGE_TICKS:g} шагах"] = round(
            hard.summary["sharpe"], 2
        )
    return row


@dataclass(frozen=True)
class Dummy:
    sharpe: np.ndarray
    expectancy: np.ndarray
    total_r: np.ndarray

    def beaten(self, sharpe: float) -> float:
        """Доля пустышек с Sharpe ниже этого, %."""
        return float((self.sharpe < sharpe).mean() * 100)


def run_dummy(
    signals: pl.DataFrame,
    markets: dict[str, MarketData],
    counts: dict[int, int],
    runs: int,
    seed: int,
    label: str,
) -> Dummy:
    rng = np.random.default_rng(seed)
    sharpe, expectancy, totals = [], [], []
    for i in range(runs):
        report = run_signals(random_skip(signals, counts, rng), markets)
        sharpe.append(report.summary["sharpe"])
        expectancy.append(report.summary["expectancy_r"])
        totals.append(total_r(report))
        if (i + 1) % 50 == 0:
            print(f"  пустышка ({label}): {i + 1} из {runs}", flush=True)
    return Dummy(np.array(sharpe), np.array(expectancy), np.array(totals))


# ------------------------------------------------------------------- таблицы


def dataset_table(dataset: pl.DataFrame) -> pl.DataFrame:
    return (
        dataset.group_by(pl.col("timestamp").dt.year().alias("год"))
        .agg(
            pl.len().alias("сигналов"),
            pl.col("r").mean().round(3).alias("средний R сигнала"),
            (pl.col("r") > 0).mean().mul(100).round(1).alias("в плюс, %"),
        )
        .sort("год")
    )


def training_table(runs: dict[str, WalkForward]) -> pl.DataFrame:
    rows = []
    for year in TEST_YEARS:
        row: dict[str, Any] = {"проверочный год": year}
        for key, run in runs.items():
            part = run.predictions.filter(pl.col("year") == year)
            skipped = (
                part["prediction"] < part[threshold_column(CENTRAL_SHARE)]
            ).mean()
            row["обучение, сигналов"] = int(part["train_rows"][0])
            row["проверка, сигналов"] = part.height
            row[f"{MODELS[key]}: пропущено, %"] = round(float(skipped) * 100, 1)
        rows.append(row)
    return pl.DataFrame(rows)


def by_year_table(reports: dict[str, Report]) -> pl.DataFrame:
    """По строке на год: expectancy и Sharpe каждого варианта рядом."""
    table: pl.DataFrame | None = None
    for label, report in reports.items():
        part = report.by_year.select(
            pl.col("year").alias("год"),
            pl.col("expectancy_r").round(3).alias(f"R: {label}"),
            pl.col("sharpe").round(2).alias(f"Sharpe: {label}"),
        )
        table = (
            part
            if table is None
            else table.join(part, on="год", how="full", coalesce=True)
        )
    assert table is not None
    r_columns = [c for c in table.columns if c.startswith("R: ")]
    sharpe_columns = [c for c in table.columns if c.startswith("Sharpe: ")]
    return table.select("год", *r_columns, *sharpe_columns).sort("год")


def thirds_table(runs: dict[str, WalkForward]) -> pl.DataFrame:
    """Средний R сигналов на проверке по третям прогноза (пороги — обучения)."""
    low = pl.col(threshold_column(CENTRAL_SHARE))
    high = pl.col(threshold_column(UPPER_THIRD))
    parts = []
    for key, run in runs.items():
        parts.append(
            run.predictions.filter(pl.col("exit_reason") != "end")
            .with_columns(
                pl.when(pl.col("prediction") < low)
                .then(pl.lit("1 низ"))
                .when(pl.col("prediction") > high)
                .then(pl.lit("3 верх"))
                .otherwise(pl.lit("2 середина"))
                .alias("треть прогноза")
            )
            .group_by("треть прогноза")
            .agg(
                pl.len().alias("сигналов"),
                pl.col("prediction").mean().round(3).alias("прогноз, R"),
                pl.col("r").mean().round(3).alias("факт, R"),
                (pl.col("r").std() / pl.len().sqrt()).round(3).alias("± погрешность"),
            )
            .with_columns(pl.lit(MODELS[key]).alias("модель"))
        )
    return (
        pl.concat(parts)
        .select("модель", "треть прогноза", pl.exclude("модель", "треть прогноза"))
        .sort("модель", "треть прогноза")
    )


def importance_table(run: WalkForward) -> pl.DataFrame:
    table = run.importance.pivot(on="year", index="feature", values="value")
    years = [str(year) for year in TEST_YEARS]
    return (
        table.with_columns(pl.mean_horizontal(*years).alias("среднее"))
        .sort(pl.col("среднее").abs(), descending=True)
        .with_columns(pl.exclude("feature").round(3))
    )


def verdict(label: str, sharpe: float, base: float, dummy: Dummy) -> str:
    bound = float(np.percentile(dummy.sharpe, ACCEPT_PERCENTILE))
    smarter = sharpe > bound
    worth = sharpe >= base
    decision = "ПРИНЯТ" if smarter and worth else "НЕ ПРИНЯТ"
    return (
        f"{label}: Sharpe {sharpe:.2f}; лучше {dummy.beaten(sharpe):.1f}% пустышек "
        f"(95-я перцентиль {bound:.2f}) — {'да' if smarter else 'нет'}; "
        f"не ниже, чем без модели ({base:.2f}) — {'да' if worth else 'нет'}. "
        f"{decision}."
    )


# -------------------------------------------------------------------- график


def save_chart(
    reports: dict[str, Report], dummies: dict[str, Dummy], base_sharpe: float
) -> Path:
    fig = Figure(figsize=(14, 11), dpi=110, facecolor=SURFACE)
    ax = fig.add_subplot(2, 1, 1)
    _style(ax, f"Капитал с {START:%d.%m.%Y}: порог — худшая треть прогноза")
    for key, report in reports.items():
        ax.plot(
            report.curve["timestamp"].to_numpy(),
            report.curve["equity"].to_numpy(),
            color=MODEL_COLORS[key],
            lw=1.8 if key != NO_MODEL else 1.2,
            label=MODELS.get(key, key),
        )
    ax.axhline(CAPITAL, color=MUTED, lw=0.6)
    ax.yaxis.set_major_formatter(_rub)
    _dates(ax)
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_SECONDARY)

    ax = fig.add_subplot(2, 1, 2)
    _style(ax, "Sharpe: модель против случайных пропусков того же числа сигналов")
    for key, dummy in dummies.items():
        color = MODEL_COLORS[key]
        ax.hist(
            dummy.sharpe,
            bins=30,
            color=color,
            alpha=0.35,
            label=f"пустышка {MODELS[key]}",
        )
        ax.axvline(
            reports[key].summary["sharpe"], color=color, lw=2.0, label=MODELS[key]
        )
    ax.axvline(base_sharpe, color=AQUA, lw=2.0, ls="--", label=NO_MODEL)
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle(
        "Шаг 6: фильтр входа пробоя (cnyrub, si, euro, gold)",
        x=0.06,
        ha="left",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = free_path(REPORTS / "catboost_filter.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------- запуск


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dummy-runs",
        type=int,
        default=DUMMY_RUNS,
        help=f"Прогонов пустышки на модель (по умолчанию {DUMMY_RUNS}).",
    )
    parser.add_argument(
        "--shuffles",
        type=int,
        default=SHUFFLE_RUNS,
        help=f"Прогонов с перемешанными метками на модель ({SHUFFLE_RUNS}).",
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
    dataset = build_dataset(markets)
    print(
        f"\nНабор: {dataset.height} сигналов ({', '.join(markets)}), метка — R сделки."
    )
    print(dataset_table(dataset))

    runs = {key: walk_forward(dataset, fitter) for key, fitter in FITTERS.items()}
    print("\nОбучение по годам (зазор: сделка закрылась до начала года):")
    print(training_table(runs))

    test = dataset.filter(pl.col("timestamp").dt.date() >= START)
    hard_markets = stressed(markets)
    base = run_signals(test, markets)
    base_sharpe = base.summary["sharpe"]
    reports: dict[str, Report] = {NO_MODEL: base}
    rows = [summary_row(NO_MODEL, base, None, run_signals(test, hard_markets))]
    dummies: dict[str, Dummy] = {}
    dummy_rows = []
    for i, (key, run) in enumerate(runs.items()):
        chosen = kept(run.predictions, CENTRAL_SHARE)
        reports[key] = run_signals(chosen, markets)
        share = 1 - chosen.height / run.predictions.height
        rows.append(
            summary_row(
                f"{MODELS[key]}, треть",
                reports[key],
                share,
                run_signals(chosen, hard_markets),
            )
        )
        counts = skipped_by_year(run.predictions, CENTRAL_SHARE)
        dummies[key] = run_dummy(
            test, markets, counts, args.dummy_runs, SEED + 100 * (i + 1), MODELS[key]
        )
        dummy = dummies[key]
        dummy_rows.append(
            {
                "вариант": f"пустышка к «{MODELS[key]}»",
                "пропущено сигналов, %": f"{share * 100:.1f}",
                "Sharpe: медиана": round(float(np.median(dummy.sharpe)), 2),
                "Sharpe: 95-я перцентиль": round(
                    float(np.percentile(dummy.sharpe, ACCEPT_PERCENTILE)), 2
                ),
                "expectancy, R: медиана": round(float(np.median(dummy.expectancy)), 3),
                "сумма, R: медиана": round(float(np.median(dummy.total_r)), 1),
            }
        )

    print(
        f"\nПортфель с {START:%d.%m.%Y} до конца данных, правила 5.5–5.6. "
        "Сделки, закрытые концом данных, в R не входят."
    )
    print(pl.DataFrame(rows))
    print("\nПустышка: те же годы, столько же случайно пропущенных сигналов.")
    print(pl.DataFrame(dummy_rows))

    print(
        f"\nРешение по ориентиру 6.4 (Sharpe выше {ACCEPT_PERCENTILE:g}% пустышек "
        "и не ниже, чем без модели):"
    )
    for key in runs:
        print(
            "- "
            + verdict(
                MODELS[key], reports[key].summary["sharpe"], base_sharpe, dummies[key]
            )
        )

    print("\nДля сведения, без решения.")
    print("\nПо годам (сделки — по году выхода):")
    print(by_year_table({MODELS.get(k, k): r for k, r in reports.items()}))

    neighbor_rows = []
    for share in SKIP_SHARES[1:]:
        for key, run in runs.items():
            chosen = kept(run.predictions, share)
            neighbor_rows.append(
                summary_row(
                    f"{MODELS[key]}, порог {share:.2f}",
                    run_signals(chosen, markets),
                    1 - chosen.height / run.predictions.height,
                )
            )
    print("\nСоседние пороги (только устойчивость):")
    print(pl.DataFrame(neighbor_rows))

    print(
        "\nПеремешанные метки: модель учится на перемешанном R — не должна выделяться."
    )
    shuffle_rows = []
    for key, fitter in FITTERS.items():
        for k in range(args.shuffles):
            run = walk_forward(dataset, fitter, shuffle_seed=SEED + 1000 + k)
            chosen = kept(run.predictions, CENTRAL_SHARE)
            sharpe = run_signals(chosen, markets).summary["sharpe"]
            shuffle_rows.append(
                {
                    "модель": MODELS[key],
                    "перемешивание": k + 1,
                    "пропущено сигналов, %": round(
                        (1 - chosen.height / run.predictions.height) * 100, 1
                    ),
                    "Sharpe": round(sharpe, 2),
                    "лучше пустышек, %": round(dummies[key].beaten(sharpe), 1),
                }
            )
    if shuffle_rows:
        print(pl.DataFrame(shuffle_rows))

    print("\nСредний R сигналов на проверке по третям прогноза (пороги — по обучению):")
    print(thirds_table(runs))
    for key, run in runs.items():
        print(f"\n{MODELS[key]}: {IMPORTANCE_TITLES[key]}")
        print(importance_table(run))

    path = save_chart(reports, dummies, base_sharpe)
    print(f"\nГрафик: {path}")


if __name__ == "__main__":
    main()
