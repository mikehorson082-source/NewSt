"""
trend_overlay.py — ядро тренд-оверлея на вечных фьючерсах MOEX (ТЗ v2.1).

Капитал целиком лежит в LQDT, фьючерсы — надстройка поверх него.
Сигнал считается по доходности ПОЗИЦИИ (с фандингом и дивпоправкой),
размер — от риск-бюджета, лоты округляются с буфером, сверху —
портфельные лимиты: волатильность, ГО, стресс по сценариям со знаком.

Модуль ничего не торгует и не ходит в сеть: на вход — дневные данные
(расчётные цены вечернего клиринга, фандинг, номинал лота, ГО),
на выход — целевые лоты и текст для Telegram.
Зависимости: numpy, pandas.

Подготовка данных по каждому вечному контракту:
    df  = merge_weekend_sessions(df)                   # сессии выходного дня → в понедельник
    check_funding_units(df["price"], df["funding"], k2) # фандинг ≤ K2 × цена, иначе не те единицы
    ret = perp_excess_returns(df["price"], df["funding"], df.get("div_adj"))

Раз в неделю, после вечернего клиринга пятницы:
    vol_d = daily_vol(ret, p);  fc = trend_forecast(ret, vol_d, p)
    table, stats = rebalance(date, ret, fc, vol_d, lot_value, go_per_lot, current, weights, shocks, p)
    send_to_telegram(format_message(date, table, stats, cash_est))
Каждый торговый день: enforce_limits(...) по текущим позициям и оценка рублёвого
остатка (cash_est += variation_margin_rub(...)); сообщение — только если лимит нарушен
или остаток ушёл в минус.

Изменения v2.1 (по ревью): сумма доходностей вместо логарифма в сигнале; буфер
зависит от размера позиции, но коридор не уже ±0.5 лота; стресс по двум сценариям
со знаком; склейка сессий выходного дня; проверка единиц фандинга; оценка
рублёвого остатка и тревога при минусе.
Изменения v2.2 (второе ревью): склейка выходных со следующим ТОРГОВЫМ днём (праздники);
проверка фандинга снизу (funding_plausibility); проверка обеспечения (collateral_ratio);
лаг исполнения в бэктесте (exec_lag); порог безубыточных издержек в отчёте;
критерий запуска (launch_decision).

Запуск без аргументов: самопроверка и бэктесты на СИНТЕТИКЕ —
с трендами и на чистом шуме. Цифры проверяют код, а не стратегию.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

TD = 252  # торговых дней в году (после склейки выходных сессий)


@dataclass
class Params:
    capital_rub: float = 500_000
    target_vol: float = 0.04            # СРЕДНИЙ годовой риск оверлея, доля капитала (4% = 20 тыс. ₽)
    lookbacks: tuple = (21, 63, 126)     # ~1, 3, 6 месяцев
    lb_weights: tuple = (0.2, 0.4, 0.4)
    forecast_scalar: float = 1.45        # при случайном блуждании даёт средний |F| ≈ 1
    forecast_cap: float = 2.0            # потолок сигнала: не больше 2× среднего риска
    vol_halflife: int = 20               # EWMA-вол-ть, полураспад в днях
    vol_long_window: int = 504           # «длинная» вол-ть, ~2 года
    vol_short_weight: float = 0.7        # σ = 0.7·короткая + 0.3·длинная
    buffer_max: float = 0.25             # коридор = ±(0.5 + min(buffer_max, buffer_frac·N₁)) лота,
    buffer_frac: float = 0.05            # N₁ — лотов при |F| = 1; уже ±0.5 коридор не бывает
    idm_cap: float = 2.0                 # потолок множителя диверсификации
    corr_window: int = 504
    cov_window: int = 126
    vol_cap_mult: float = 1.5            # портфельная вол-ть не выше 1.5 × целевой
    max_go_frac: float = 0.15            # ГО не более 15% капитала
    max_stress_frac: float = 0.12        # убыток в худшем из двух стресс-сценариев ≤ 12% капитала
    rebalance_weekday: int | None = 4    # 4 = пятница (исполнение в понедельник); None = каждый день


# ---------- 1. Подготовка данных и доходность позиции ----------

def merge_weekend_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Склеивает бары сессий выходного дня со следующим ТОРГОВЫМ днём, который есть
    в данных (понедельник-праздник → вторник). df: index = даты, колонка price
    (расчётная цена) и, если есть, funding, div_adj. Цена — последняя в группе,
    фандинг и дивпоправка суммируются: доходность первого торгового дня после
    выходных = от пятницы до него со всем фандингом между. Выходные в самом конце
    данных остаются отдельной строкой до следующей загрузки."""
    d = pd.DatetimeIndex(df.index)
    regular = d[d.weekday < 5]
    pos = regular.searchsorted(d)                           # первый будний день ≥ даты
    key = pd.DatetimeIndex([regular[i] if i < len(regular) else x for i, x in zip(pos, d)])
    agg = {c: ("last" if c == "price" else "sum") for c in df.columns}
    out = df.groupby(key).agg(agg)
    out.index.name = df.index.name
    return out


def check_funding_units(price: pd.Series, funding: pd.Series, k2: float, tol: float = 1.05) -> None:
    """Фандинг по спецификации не превышает L2 = K2 × предыдущая расчётная цена.
    Если превышает — единицы перепутаны (например, рубли на контракт вместо пунктов цены).
    K2 = 0.0015 для IMOEXF, RGBIF и GLDRUBF; 0.006 для SP500F и QQQF;
    0.0035 для CNYRUBF (по публикациям биржи 2024 г., сверить текущее значение)."""
    cap = k2 * price.shift(1).reindex(funding.index)
    bad = funding.abs() > tol * cap
    if bad.any():
        raise ValueError(f"фандинг выше K2×цена в {int(bad.sum())} дн., первый "
                         f"{pd.Timestamp(bad.idxmax()):%Y-%m-%d}: проверьте единицы")


def funding_plausibility(price: pd.Series, funding: pd.Series,
                         lo: float = 0.01, hi: float = 0.40) -> tuple[float, bool]:
    """Проверка снизу: годовой фандинг в долях цены для CNYRUBF и GLDRUBF должен быть
    сопоставим с разницей ставок — не около нуля и не сотни процентов. Возвращает
    (годовой фандинг, в пределах ли [lo, hi]). Для IMOEXF и RGBIF фандинг может быть
    около нуля законно — к ним не применять."""
    rate = float((funding / price.shift(1).reindex(funding.index)).mean() * TD)
    return rate, lo <= abs(rate) <= hi


def collateral_ratio(lqdt_value: float, cash: float, go_total: float, lqdt_discount: float) -> float:
    """Покрытие ГО обеспечением: (LQDT с дисконтом брокера + рубли) / ГО.
    Бот предупреждает, если меньше 3 — например, после продаж LQDT под вармаржу."""
    return float("inf") if go_total <= 0 else (lqdt_value * (1 - lqdt_discount) + cash) / go_total


def perp_excess_returns(price: pd.Series, funding: pd.Series | None = None,
                        div_adj: pd.Series | None = None) -> pd.Series:
    """Дневная доходность ДЛИННОЙ позиции в вечном фьючерсе — так же, как биржа
    считает вариационную маржу: (ΔP − фандинг + дивпоправка) / P.

    price   — расчётная цена вечернего клиринга, в пунктах цены;
    funding — фандинг за день в тех же пунктах: > 0 — покупатель платит продавцу;
    div_adj — дивидендная поправка в пунктах (IMOEXF, вечные на акции): начисляется покупателю.
    """
    dp = price.diff()
    if funding is not None:
        dp = dp - funding.reindex(price.index).fillna(0.0)
    if div_adj is not None:
        dp = dp + div_adj.reindex(price.index).fillna(0.0)
    return dp / price.shift(1)


def quarterly_excess_returns(prices: pd.DataFrame, expiries: dict,
                             roll_days: int = 3) -> pd.Series:
    """Склейка квартальных контрактов ПО ДОХОДНОСТЯМ, а не по ценам
    (для истории до появления вечных: MXI/MIX, CR/Si, RGBI, GL/GD и т.п.).

    prices   — index = даты, columns = коды контрактов, значения = расчётные цены;
    expiries — {код контракта: дата экспирации}.
    Позиция переезжает в следующий контракт на закрытии дня за roll_days торговых
    дней до экспирации. Доходность дня t = изменение цены контракта, который держали
    на закрытии t-1, поэтому carry (схождение к споту) учтён автоматически.
    """
    dates = prices.index
    order = sorted(expiries, key=lambda c: pd.Timestamp(expiries[c]))
    roll_pos = [min(max(dates.searchsorted(pd.Timestamp(expiries[c])) - roll_days, 0), len(dates) - 1)
                for c in order]
    roll_dates = dates[roll_pos].values
    held_idx = np.minimum(np.searchsorted(roll_dates, dates.values, side="right"), len(order) - 1)
    cols = prices.columns.get_indexer(np.array(order, dtype=object)[held_idx])
    if (cols < 0).any():
        raise ValueError("в prices нет колонки для одного из контрактов из expiries")
    P = prices.to_numpy(dtype=float)
    r = np.full(len(dates), np.nan)
    rows = np.arange(1, len(dates))
    r[1:] = P[rows, cols[:-1]] / P[rows - 1, cols[:-1]] - 1.0
    return pd.Series(r, index=dates, name="ret")


# ---------- 2. Волатильность и сигнал ----------

def daily_vol(ret: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Дневная σ: 0.7·EWMA + 0.3·«длинная». Длинная часть не даёт раздуть позицию
    после затишья — для рубля это важнее любой тонкой модели волатильности."""
    r2 = ret.pow(2)
    short = np.sqrt(r2.ewm(halflife=p.vol_halflife, min_periods=20).mean())
    long_ = np.sqrt(r2.rolling(p.vol_long_window, min_periods=60).mean()).fillna(short)
    return p.vol_short_weight * short + (1 - p.vol_short_weight) * long_


def raw_trend(ret: pd.DataFrame, vol_d: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Σ w_k·z_k / Σ w_k, где z_k = сумма доходностей позиции за k дней / (σ·√k).
    Простая сумма = P&L позиции постоянного номинала; логарифм тут ничего не даёт."""
    raw = None
    for k, w in zip(p.lookbacks, p.lb_weights):
        z = ret.rolling(k, min_periods=int(0.8 * k)).sum() / (vol_d * np.sqrt(k))
        raw = w * z if raw is None else raw + w * z
    return raw / sum(p.lb_weights)


def trend_forecast(ret: pd.DataFrame, vol_d: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Непрерывный сигнал вместо голосования знаков: F = clip(scalar·raw, ±cap).
    |F| = 1 — тренд обычной силы, на нём держим целевой риск; |F| = 2 — потолок."""
    return (p.forecast_scalar * raw_trend(ret, vol_d, p)).clip(-p.forecast_cap, p.forecast_cap)


def estimate_forecast_scalar(ret: pd.DataFrame, vol_d: pd.DataFrame, p: Params) -> float:
    """Скаляр, при котором средний |F| по истории всех инструментов = 1.
    Это нормировка масштаба, а не подгонка под доходность."""
    return float(1.0 / raw_trend(ret, vol_d, p).abs().stack().mean())


# ---------- 3. Размер позиции и лимиты ----------

def diversification_multiplier(corr: pd.DataFrame, w: pd.Series, cap: float) -> float:
    """IDM = 1/√(wᵀCw): во сколько поднять риск на инструмент, чтобы портфель в сумме
    вышел на целевую вол-ть. Отрицательные корреляции обнуляем — так осторожнее."""
    c = corr.clip(lower=0.0).fillna(0.0).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(c, 1.0)
    wv = w.to_numpy(dtype=float)
    return float(min(1.0 / np.sqrt(wv @ c @ wv), cap))


def buffered_round(ideal: pd.Series, current: pd.Series, extra=0.25) -> pd.Series:
    """Целые лоты с гистерезисом. Коридор = ideal ± (0.5 + extra): пока текущая позиция
    внутри, не торгуем, иначе идём к ближайшей границе коридора (минимум оборота).
    extra ≥ 0 обязательно: в коридоре уже одного лота может не оказаться целого числа,
    и позиция будет прыгать 1 → 2 → 1 каждую неделю. NaN в ideal — позицию не трогаем."""
    if np.isscalar(extra):
        extra = pd.Series(float(extra), index=ideal.index)
    half = 0.5 + extra.reindex(ideal.index).fillna(0.0).clip(lower=0.0)
    cur = current.reindex(ideal.index).fillna(0).astype(float)
    lo, hi = ideal - half, ideal + half
    new = cur.copy()
    up, down = cur < lo, cur > hi
    new[up] = np.ceil(lo[up])
    new[down] = np.floor(hi[down])
    return new.astype(int)


def enforce_limits(lots: pd.Series, lot_value: pd.Series, cov_ann: pd.DataFrame,
                   go_per_lot: pd.Series, shocks: pd.Series, p: Params) -> tuple[pd.Series, dict]:
    """Жадно срезаем по одному лоту у главного источника нарушения, пока не выполнено:
    портфельная вол-ть ≤ vol_cap_mult·цель, ГО ≤ max_go_frac, стресс ≤ max_stress_frac.

    Стресс: shocks — движения в сценарии «кризис рубля», доли номинала СО ЗНАКОМ
    (юань и золото вверх, индекс и ОФЗ вниз); второй сценарий — зеркальный.
    Убыток в худшем из двух = |Σ номинал_i × shock_i|: разнонаправленные позиции
    гасят друг друга, а однобокая ставка на стабильность рубля режется."""
    tick = list(lots.index)
    x = lots.to_numpy(dtype=int).copy()
    lv = np.nan_to_num(lot_value.reindex(tick).to_numpy(dtype=float))
    go = np.nan_to_num(go_per_lot.reindex(tick).to_numpy(dtype=float))
    sh = np.nan_to_num(shocks.reindex(tick).to_numpy(dtype=float))
    cov = np.nan_to_num(cov_ann.reindex(index=tick, columns=tick).to_numpy(dtype=float))
    vol_lim = p.vol_cap_mult * p.target_vol * p.capital_rub
    go_lim = p.max_go_frac * p.capital_rub
    st_lim = p.max_stress_frac * p.capital_rub
    cut = {"vol": 0, "go": 0, "stress": 0}
    while True:
        e = x * lv                                   # номинал позиций в рублях, со знаком
        port_vol = float(np.sqrt(max(e @ cov @ e, 0.0)))
        go_used = float(np.abs(x) @ go)
        pnl_crisis = float(e @ sh)                   # P&L в сценарии кризиса; зеркальный = −pnl
        stress = abs(pnl_crisis)
        if port_vol <= vol_lim and go_used <= go_lim and stress <= st_lim:
            break
        if port_vol > vol_lim:
            contrib, why = e * (cov @ e), "vol"      # вклад в дисперсию портфеля
        elif stress > st_lim:
            contrib, why = np.sign(pnl_crisis) * e * sh, "stress"   # вклад в убыток худшего сценария
        else:
            contrib, why = np.abs(x) * go, "go"
        i = int(np.argmax(np.where(x != 0, contrib, -np.inf)))
        x[i] -= int(np.sign(x[i]))
        cut[why] += 1
    return pd.Series(x, index=tick), dict(port_vol_rub=port_vol, go_rub=go_used,
                                          stress_rub=stress, cut=cut)


def variation_margin_rub(lots: pd.Series, lot_value_prev: pd.Series, ret: pd.Series) -> float:
    """Оценка вариационной маржи за день (₽, со знаком) по позициям на вчерашний клиринг.
    Бот копит её к последнему известному рублёвому остатку, чтобы поймать уход в минус."""
    return float((lots * lot_value_prev.reindex(lots.index) * ret.reindex(lots.index)).fillna(0.0).sum())


# ---------- 4. Решение на дату ----------

def rebalance(date, ret: pd.DataFrame, fc: pd.DataFrame, vol_d: pd.DataFrame,
              lot_value: pd.DataFrame, go_per_lot: pd.DataFrame, current: pd.Series,
              weights: pd.Series, shocks: pd.Series, p: Params) -> tuple[pd.DataFrame, dict]:
    """Целевые лоты по данным до date включительно. current — фактические лоты из state.db."""
    hist = ret.loc[:date]
    vol_ann = vol_d.loc[date] * np.sqrt(TD)
    lv = lot_value.loc[date]
    f = fc.loc[date]
    active = f.notna() & vol_ann.notna() & lv.notna()
    w = weights.reindex(ret.columns).where(active)
    w = w / w.sum()
    idm = 1.0
    if active.sum() > 1:
        corr = hist.loc[:, active].tail(p.corr_window).corr()
        idm = diversification_multiplier(corr, w[active], p.idm_cap)
    budget = p.capital_rub * p.target_vol * idm * w          # ₽/год риска при |F| = 1
    n1 = budget / (vol_ann * lv)                              # лотов при |F| = 1
    ideal = f * n1
    extra = np.minimum(p.buffer_max, p.buffer_frac * n1.abs())
    target = buffered_round(ideal, current, extra)
    cov_ann = hist.tail(p.cov_window).cov() * TD
    target, stats = enforce_limits(target, lv, cov_ann, go_per_lot.loc[date], shocks, p)
    cur = current.reindex(ret.columns).fillna(0).astype(int)
    table = pd.DataFrame({"forecast": f, "vol_ann": vol_ann, "n1": n1, "ideal": ideal,
                          "current": cur, "target": target, "delta": target - cur})
    stats["idm"] = idm
    return table, stats


def format_message(date, table: pd.DataFrame, stats: dict,
                   cash_est: float | None = None, cash_reserve: float = 20_000,
                   coll_ratio: float | None = None, coll_min: float = 3.0) -> str:
    """Текст для Telegram: только то, что нужно сделать руками."""
    def rub(x):
        return f"{x:,.0f}".replace(",", " ")
    lines = [f"Оверлей на {pd.Timestamp(date):%d.%m.%Y}"]
    todo = table[table["delta"] != 0]
    if todo.empty:
        lines.append("Без изменений.")
    for tk, r in todo.iterrows():
        side = "Купить" if r["delta"] > 0 else "Продать"
        lines.append(f"• {side} {abs(int(r['delta']))} {tk}: {int(r['current'])} → {int(r['target'])}"
                     f" (сигнал {r['forecast']:+.2f})")
    no_data = [str(t) for t in table.index[table["ideal"].isna()]]
    if no_data:
        lines.append("Нет данных, позиция не трогалась: " + ", ".join(no_data))
    if stats.get("cut", {}).get("stress"):
        lines.append(f"Стресс-лимит срезал {stats['cut']['stress']} лот(а).")
    lines.append(f"Риск ≈ {rub(stats['port_vol_rub'])} ₽/год · ГО ≈ {rub(stats['go_rub'])} ₽"
                 f" · стресс ≈ {rub(stats['stress_rub'])} ₽")
    if cash_est is not None and cash_est < 0:
        lines.append(f"Рублёвый остаток ≈ {rub(cash_est)} ₽: продайте LQDT примерно на "
                     f"{rub(cash_reserve - cash_est)} ₽ на следующий торговый день.")
    if coll_ratio is not None and coll_ratio < coll_min:
        lines.append(f"Обеспечение покрывает ГО всего в {coll_ratio:.1f} раза (норма ≥ {coll_min:.0f}): "
                     f"сократите позиции или не продавайте LQDT.")
    return "\n".join(lines)


# ---------- 5. Мини-бэктест ----------

def backtest(ret: pd.DataFrame, lot_value: pd.DataFrame, go_per_lot: pd.DataFrame,
             weights: pd.Series, shocks: pd.Series, cost_per_lot: pd.Series,
             p: Params, warmup: int = 252, exec_lag: int = 1):
    """Дневной P&L оверлея в рублях (поверх дохода LQDT). Решение на закрытии дня t
    исполняется на закрытии t+exec_lag (по умолчанию t+1): это и лаг ручного исполнения,
    и защита от заглядывания в будущее; exec_lag=2 — проверка устойчивости к задержке.
    В дни ребаланса — полный пересчёт, в остальные — только проверка риск-лимитов (внеплановое сообщение бота). Возвращает pnl, costs, positions, events."""
    vol_d = daily_vol(ret, p)
    fc = trend_forecast(ret, vol_d, p)
    R, LV = ret.fillna(0.0), lot_value.fillna(0.0)
    tick = list(ret.columns)
    cpl = cost_per_lot.reindex(tick).fillna(0.0)
    pos = pd.Series(0, index=tick, dtype=int)
    pending, due = None, None
    pnl, costs, hist, events = np.zeros(len(ret)), np.zeros(len(ret)), [], []
    for k, d in enumerate(ret.index):
        if k > 0:
            pnl[k] = float((pos * LV.iloc[k - 1] * R.iloc[k]).sum())
        if pending is not None and k >= due:
            costs[k] = float(((pending - pos).abs() * cpl).sum())
            pos, pending = pending, None
        if k >= warmup:
            if p.rebalance_weekday is None or d.weekday() == p.rebalance_weekday:
                table, stats = rebalance(d, ret, fc, vol_d, lot_value, go_per_lot, pos, weights, shocks, p)
                target, kind = table["target"], "rebalance"
            else:
                cov_ann = ret.loc[:d].tail(p.cov_window).cov() * TD
                target, stats = enforce_limits(pos, lot_value.loc[d], cov_ann, go_per_lot.loc[d], shocks, p)
                kind = "alert"
            changed = bool((target != pos).any())
            if kind == "rebalance" or changed:
                events.append({"date": d, "kind": kind, "stress_cut": stats["cut"]["stress"],
                               "total_cut": sum(stats["cut"].values())})
            if changed:
                pending, due = target.astype(int), k + exec_lag
        hist.append(pos.to_numpy().copy())
    positions = pd.DataFrame(hist, index=ret.index, columns=tick)
    return (pd.Series(pnl, index=ret.index), pd.Series(costs, index=ret.index),
            positions, pd.DataFrame(events))


def summary(pnl: pd.Series, costs: pd.Series, positions: pd.DataFrame,
            events: pd.DataFrame, p: Params) -> dict:
    started = positions.abs().sum(axis=1).cumsum() > 0
    net = (pnl - costs)[started]
    years = len(net) / TD
    ann, vol = net.mean() * TD, net.std() * np.sqrt(TD)
    equity = net.cumsum()
    max_dd = float((equity - equity.cummax()).min())
    out = {
        "лет": round(years, 1),
        "доход сверх LQDT, ₽/год": round(ann),
        "вол-ть, ₽/год": round(vol),
        "Sharpe": round(ann / vol, 2) if vol > 0 else float("nan"),
        "макс. просадка, ₽": round(max_dd),
        "макс. просадка, % капитала": round(100 * max_dd / p.capital_rub, 1),
        "оборот, лотов/год": round(float(positions.diff().abs().sum().sum()) / years),
        "издержки, ₽/год": round(float(costs[started].sum()) / years),
    }
    turnover = float(positions.diff().abs().sum().sum()) / years
    gross = float(pnl[started].sum()) / years
    out["безубыточные издержки, ₽/лот"] = round(gross / turnover) if turnover > 0 else float("nan")
    if len(events):
        reb = events[events["kind"] == "rebalance"]
        out["ребалансов со срезкой по стрессу, %"] = round(100 * float((reb["stress_cut"] > 0).mean()))
        out["внеплановых сообщений о лимитах, в год"] = round(float((events["kind"] == "alert").sum()) / years, 1)
    return out


def launch_decision(sharpe_cost2x: float, neighbor_sharpes, loss_2022_rub: float, p: Params) -> dict:
    """Критерий запуска на реальные деньги, зафиксированный ДО бэктеста на истории.
    sharpe_cost2x   — Sharpe надстройки на истории при издержках ×2;
    neighbor_sharpes — Sharpe на соседних наборах параметров (окна, буфер, полураспад, риск);
    loss_2022_rub   — убыток надстройки за февраль–март 2022 (положительное число).
    Не прошли — остаёмся на чистом LQDT; выигрыш от структуры капитала сохраняется."""
    ns = np.asarray(list(neighbor_sharpes), dtype=float)
    checks = {
        "Sharpe при издержках ×2 ≥ 0.3": bool(sharpe_cost2x >= 0.3),
        "плюс на ≥ 2/3 соседних наборов": bool(len(ns) > 0 and np.mean(ns > 0) >= 2 / 3),
        "убыток 2022 ≤ стресс-лимита": bool(loss_2022_rub <= p.max_stress_frac * p.capital_rub),
    }
    checks["запуск"] = all(checks.values())
    return checks


# ---------- 6. Самопроверка на синтетике ----------

TICKERS = ["CNYRUBF", "GLDRUBF", "IMOEXF", "RGBIF"]


def _synthetic(n_days: int, trend_strength: float, seed: int):
    """4 инструмента с правдоподобными вол-тями и корреляциями. Тренды — режимы
    дрейфа по 3–9 месяцев со случайным Sharpe ~ N(0, trend_strength)."""
    rng = np.random.default_rng(seed)
    vol = np.array([0.11, 0.20, 0.22, 0.09])
    corr = np.array([[1.0, 0.5, -0.1, -0.1],
                     [0.5, 1.0, 0.0, 0.0],
                     [-0.1, 0.0, 1.0, 0.4],
                     [-0.1, 0.0, 0.4, 1.0]])
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    drift = np.zeros((n_days, 4))
    for j in range(4):
        t = 0
        while t < n_days:
            n = int(rng.integers(63, 190))
            drift[t:t + n, j] = rng.normal(0.0, trend_strength) * vol[j] / TD
            t += n
    noise = rng.standard_normal((n_days, 4)) @ np.linalg.cholesky(corr).T * vol / np.sqrt(TD)
    ret = pd.DataFrame(drift + noise, index=dates, columns=TICKERS)
    ret.iloc[0] = np.nan
    lot_value = (1 + ret.fillna(0)).cumprod() * np.array([12_500, 10_000, 21_000, 11_000])
    return ret, lot_value, lot_value * 0.15


def _self_check():
    # старая формула v1: внутренняя сумма никогда не равна 0, поэтому «кэш» недостижим
    sums = {round(0.2 * a + 0.4 * b + 0.4 * c, 2) for a, b, c in itertools.product((-1, 1), repeat=3)}
    assert 0 not in sums, sums
    # буфер: вход при идеале > 0.75 лота; из лонга выходим в ноль, а не сразу в шорт
    got = buffered_round(pd.Series({"A": 1.2, "B": 0.7, "C": -0.3, "D": -2.6}),
                         pd.Series({"A": 0, "B": 0, "C": 1, "D": -1}), 0.25).to_dict()
    assert got == {"A": 1, "B": 0, "C": 0, "D": -2}, got
    # коридор ±0.5 (минимум): идеал 1.5 не гоняет позицию ни из 1, ни из 2
    for start in (1, 2):
        pos = pd.Series({"A": start})
        for _ in range(5):
            pos = buffered_round(pd.Series({"A": 1.5}), pos, 0.0)
        assert pos["A"] == start, (start, pos["A"])
    # стресс со знаком: лонг юаня + лонг индекса — хедж; шорт юаня + лонг индекса — однобоко
    p = Params()
    sh = pd.Series({"CNY": 0.35, "IMX": -0.35})
    lv = pd.Series({"CNY": 10_000.0, "IMX": 10_000.0})
    zero_cov = pd.DataFrame(0.0, index=lv.index, columns=lv.index)
    _, s1 = enforce_limits(pd.Series({"CNY": 1, "IMX": 1}), lv, zero_cov, lv * 0, sh, p)
    _, s2 = enforce_limits(pd.Series({"CNY": -1, "IMX": 1}), lv, zero_cov, lv * 0, sh, p)
    assert np.isclose(s1["stress_rub"], 0.0) and np.isclose(s2["stress_rub"], 7_000.0), (s1, s2)
    # склейка выходных: пт, сб, вс, пн → пт и пн; цена понедельника, фандинг суммой
    idx = pd.to_datetime(["2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21"])
    m = merge_weekend_sessions(pd.DataFrame({"price": [100, 101, 102, 103], "funding": [0.1, 0.2, 0.3, 0.4]}, index=idx))
    assert list(m.index) == [idx[0], idx[3]] and m["price"].iloc[1] == 103 and np.isclose(m["funding"].iloc[1], 0.9)
    # понедельник — праздник: выходные склеиваются со вторником, строки с датой праздника нет
    idx2 = pd.to_datetime(["2026-09-18", "2026-09-19", "2026-09-20", "2026-09-22"])
    m2 = merge_weekend_sessions(pd.DataFrame({"price": [100, 101, 102, 103], "funding": [0.1, 0.2, 0.3, 0.4]}, index=idx2))
    assert list(m2.index) == [idx2[0], idx2[3]] and np.isclose(m2["funding"].iloc[1], 0.9), m2
    # проверка единиц фандинга: 0.1 п. при цене 100 и K2 = 0.15% проходит, 5 п. — нет
    price = pd.Series([100.0, 100.0, 100.0], index=idx[:3])
    check_funding_units(price, pd.Series([0.0, 0.1, 0.1], index=idx[:3]), 0.0015)
    try:
        check_funding_units(price, pd.Series([0.0, 5.0, 0.1], index=idx[:3]), 0.0015)
        raise AssertionError("проверка единиц не сработала")
    except ValueError:
        pass
    # склейка квартальных по доходностям: контанго 2% не даёт скачка на ролле
    dates = pd.bdate_range("2024-03-01", periods=30)
    h = pd.Series(100.0 * 1.001 ** np.arange(30), index=dates)
    r = quarterly_excess_returns(pd.DataFrame({"H": h, "M": h * 1.02}),
                                 {"H": "2024-03-21", "M": "2024-06-20"}, roll_days=3)
    assert np.allclose(r.dropna(), 0.001)
    # фандинг: цена стоит, фандинг 0.5 п. при цене 100 → лонг теряет 0.5% в день
    assert np.isclose(perp_excess_returns(h * 0 + 100, h * 0 + 0.5).iloc[-1], -0.005)


if __name__ == "__main__":
    _self_check()
    p = Params()
    weights = pd.Series(1.0, index=TICKERS)                  # равный риск-бюджет
    # сценарий «кризис рубля», доли номинала со знаком; второй сценарий — зеркальный
    shocks = pd.Series({"CNYRUBF": 0.35, "GLDRUBF": 0.30, "IMOEXF": -0.35, "RGBIF": -0.15})
    cost_per_lot = pd.Series({"CNYRUBF": 5.0, "GLDRUBF": 5.0, "IMOEXF": 10.0, "RGBIF": 5.0})  # подставьте свои

    ret, lv, go = _synthetic(2300, trend_strength=0.8, seed=1)
    pnl, costs, pos, events = backtest(ret, lv, go, weights, shocks, cost_per_lot, p)
    print("С трендами (СИНТЕТИКА — проверка кода, не стратегии):")
    for key, val in summary(pnl, costs, pos, events, p).items():
        print(f"  {key}: {val}")

    for label, kw, cpl in (("издержки ×2", {}, cost_per_lot * 2), ("лаг исполнения 2 дня", {"exec_lag": 2}, cost_per_lot)):
        s = summary(*backtest(ret, lv, go, weights, shocks, cpl, p, **kw), p)
        print(f"  {label}: Sharpe {s['Sharpe']}, доход {s['доход сверх LQDT, ₽/год']} ₽/год")

    sharpes = []
    for seed in range(1, 6):
        r0, lv0, go0 = _synthetic(2300, trend_strength=0.0, seed=seed)
        sharpes.append(summary(*backtest(r0, lv0, go0, weights, shocks, cost_per_lot, p), p)["Sharpe"])
    print(f"\nЧистый шум, 5 прогонов: Sharpe {[float(x) for x in sharpes]}, среднее {np.mean(sharpes):+.2f}")
    print("  (должно быть около нуля или чуть ниже из-за издержек; заметный плюс на шуме = заглядывание в будущее)")

    d = ret.index[-1]
    vol_d = daily_vol(ret, p)
    fc = trend_forecast(ret, vol_d, p)
    table, stats = rebalance(d, ret, fc, vol_d, lv, go, pos.iloc[-1], weights, shocks, p)
    print("\n" + table.round(2).to_string())
    print("\n" + format_message(d, table, stats, cash_est=-12_300))
