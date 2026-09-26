"""Регрессионные проверки. Запуск пользователем: uv run pytest src/data/test_build_continuous_futures.py"""

# raw содержит naive timestamp MSK; фикстуры воспроизводят эту схему намеренно.
# ruff: noqa: DTZ001

import math
from datetime import date, datetime

import polars as pl
import pytest

from data.build_continuous_futures import (
    add_etf_nav,
    assign_trading_dates,
    available_assets,
    build_selection,
    load_contract_metadata,
    parse_contract_dates,
    registry_name,
    require_derived_path,
    resample_15_minutes,
    select_ready_assets,
    select_requested_assets,
    synchronized_closes,
)


@pytest.mark.parametrize("first_trade", [None, ""])
def test_first_trade_is_optional(first_trade):
    assert parse_contract_dates(
        "TEST", {"LSTTRADE": "2026-09-17", "FRSTTRADE": first_trade}
    ) == {"LSTTRADE": date(2026, 9, 17)}


def test_missing_first_trade_field():
    assert parse_contract_dates("TEST", {"LSTTRADE": "2026-09-17"}) == {
        "LSTTRADE": date(2026, 9, 17)
    }


def test_missing_last_trade_has_contract_in_error():
    with pytest.raises(ValueError, match="TEST:.*LSTTRADE"):
        parse_contract_dates("TEST", {"FRSTTRADE": "2025-01-01"})


def test_local_reference_without_optional_first_trade(tmp_path):
    path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {"ticker": ["TEST"], "last_trade_date": [date(2026, 9, 17)]}
    ).write_parquet(path)
    original = path.read_bytes()
    assert load_contract_metadata(["TEST"], path) == {
        "TEST": {"LSTTRADE": date(2026, 9, 17)}
    }
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            {"ticker": ["OTHER"], "last_trade_date": [date(2026, 9, 17)]},
            "нет контрактов TEST",
        ),
        (
            {"ticker": ["TEST", "TEST"], "last_trade_date": [date(2026, 9, 17)] * 2},
            "неоднозначные",
        ),
        ({"ticker": ["TEST"], "last_trade_date": [None]}, "TEST:.*LSTTRADE"),
        ({"ticker": ["TEST"]}, "отсутствуют столбцы last_trade_date"),
        (
            {
                "ticker": ["TEST"],
                "last_trade_date": [date(2026, 9, 17)],
                "first_trade_date": [date(2026, 9, 18)],
            },
            "некорректные сроки",
        ),
    ],
)
def test_invalid_local_reference_is_not_silently_ignored(tmp_path, rows, message):
    path = tmp_path / "metadata.parquet"
    pl.DataFrame(rows).write_parquet(path)
    with pytest.raises(ValueError, match=message):
        load_contract_metadata(["TEST"], path)


def test_missing_local_reference(tmp_path):
    with pytest.raises(FileNotFoundError, match="Нет справочника"):
        load_contract_metadata(["TEST"], tmp_path / "missing.parquet")


def market(volumes):
    return pl.DataFrame(
        [
            {
                "asset": "TEST",
                "date": date(2026, 9, day),
                "contract": contract,
                "daily_volume": volume,
                "first_open": 100.0,
                "last_close": 100.0,
                "trading_minutes": 1,
            }
            for day, contracts in volumes.items()
            for contract, volume in contracts.items()
        ]
    )


def metadata(old_end=date(2026, 9, 30)):
    return {
        "OLD": {"FRSTTRADE": date(2026, 1, 1), "LSTTRADE": old_end},
        "NEW": {"FRSTTRADE": date(2026, 1, 1), "LSTTRADE": date(2026, 12, 31)},
    }


def prices(tmp_path, day=10):
    paths = {contract: tmp_path / f"{contract}.parquet" for contract in ("OLD", "NEW")}
    for contract, close in (("OLD", 100.0), ("NEW", 110.0)):
        pl.DataFrame(
            {"timestamp": [datetime(2026, 9, day, 23, 49)], "close": [close]}
        ).write_parquet(paths[contract])
    return paths


def test_initial_selection_ignores_first_day_volume():
    daily = market({9: {"OLD": 100, "NEW": 10}, 10: {"OLD": 1, "NEW": 1000}})
    selected, rolls = build_selection("TEST", daily, date(2026, 9, 10), metadata(), {})
    assert selected["contract"].to_list() == ["OLD"]
    assert selected["decision_date"][0] == date(2026, 9, 9)
    assert rolls.is_empty()


def test_missing_selected_contract_does_not_trigger_future_based_fallback():
    daily = market(
        {9: {"OLD": 100, "NEW": 10}, 10: {"OLD": 100, "NEW": 10}, 11: {"NEW": 1000}}
    )
    with pytest.raises(ValueError, match="не торговался"):
        build_selection("TEST", daily, date(2026, 9, 10), metadata(), {})


def test_explicit_seven_day_buffer_rolls_even_when_old_contract_has_more_volume(
    tmp_path,
):
    daily = market(
        {
            9: {"OLD": 100, "NEW": 10},
            10: {"OLD": 100, "NEW": 10},
            11: {"OLD": 100, "NEW": 10},
        }
    )
    selected, rolls = build_selection(
        "TEST",
        daily,
        date(2026, 9, 10),
        metadata(date(2026, 9, 18)),
        prices(tmp_path),
        roll_days=7,
    )
    assert selected["contract"].to_list() == ["OLD", "NEW"]
    assert rolls["reason"][0] == "last_trade_deadline"
    assert rolls["adjustment_delta"][0] == -10.0


def test_default_keeps_liquid_front_during_last_week():
    daily = market(
        {
            9: {"OLD": 100, "NEW": 10},
            10: {"OLD": 100, "NEW": 10},
            11: {"OLD": 100, "NEW": 10},
        }
    )
    selected, rolls = build_selection(
        "TEST", daily, date(2026, 9, 10), metadata(date(2026, 9, 18)), {}
    )
    assert selected["contract"].to_list() == ["OLD", "OLD"]
    assert rolls.is_empty()


@pytest.mark.parametrize(
    "old_volume,new_volume,reason",
    [
        (100, 10, "last_trade_deadline"),
        (100, 100, "last_trade_deadline"),
        (10, 100, "next_avg_volume_exceeded_current"),
    ],
)
def test_last_trade_day_and_volume_priority(tmp_path, old_volume, new_volume, reason):
    daily = market(
        {
            9: {"OLD": 100, "NEW": 50},
            10: {"OLD": old_volume, "NEW": new_volume},
            11: {"NEW": 10},
        }
    )
    selected, rolls = build_selection(
        "TEST", daily, date(2026, 9, 10), metadata(date(2026, 9, 11)), prices(tmp_path)
    )
    assert selected["contract"].to_list() == ["OLD", "NEW"]
    assert rolls["reason"][0] == reason
    assert rolls["deadline_reached"][0] is True
    assert rolls["roll_days_before_last_trade"][0] == 0


def test_last_trade_deadline_requires_candidate_known_yesterday():
    daily = market({9: {"OLD": 100}, 10: {"OLD": 100}, 11: {"NEW": 1000}})
    with pytest.raises(ValueError, match="нет следующего контракта"):
        build_selection(
            "TEST", daily, date(2026, 9, 10), metadata(date(2026, 9, 11)), {}
        )


def test_equal_volume_does_not_roll_before_deadline():
    daily = market(
        {
            9: {"OLD": 100, "NEW": 10},
            10: {"OLD": 100, "NEW": 100},
            11: {"OLD": 100, "NEW": 10},
        }
    )
    selected, rolls = build_selection("TEST", daily, date(2026, 9, 10), metadata(), {})
    assert selected["contract"].to_list() == ["OLD", "OLD"]
    assert rolls.is_empty()


def test_cny_september_2022_volume_signal_is_not_masked_by_buffer(tmp_path):
    # Объемы дня решения из raw CRU2/CRZ2 за 2022-09-07.
    daily = pl.DataFrame(
        [
            {
                "asset": "CNYRUB",
                "date": day,
                "contract": contract,
                "daily_volume": volume,
            }
            for day, values in [
                (date(2022, 9, 6), {"CRU2": 100, "CRZ2": 10}),
                (date(2022, 9, 7), {"CRU2": 176838, "CRZ2": 202371}),
                (date(2022, 9, 8), {"CRU2": 100, "CRZ2": 10}),
            ]
            for contract, volume in values.items()
        ]
    )
    paths = {
        contract: tmp_path / f"{contract}.parquet" for contract in ("CRU2", "CRZ2")
    }
    for contract, close in [("CRU2", 8.77), ("CRZ2", 9.0)]:
        pl.DataFrame(
            {"timestamp": [datetime(2022, 9, 7, 23, 47)], "close": [close]}
        ).write_parquet(paths[contract])
    selected, rolls = build_selection(
        "CNYRUB",
        daily,
        date(2022, 9, 7),
        {
            "CRU2": {"LSTTRADE": date(2022, 9, 15)},
            "CRZ2": {"LSTTRADE": date(2022, 12, 15)},
        },
        paths,
    )
    assert selected["contract"].to_list() == ["CRU2", "CRZ2"]
    assert rolls["roll_date"][0] == date(2022, 9, 8)
    assert rolls["reason"][0] == "next_avg_volume_exceeded_current"
    assert rolls["volume_window_days"][0] == 3
    assert rolls["deadline_reached"][0] is False


def test_volume_roll_uses_completed_days_not_today(tmp_path):
    daily = market(
        {
            9: {"OLD": 100, "NEW": 100},
            10: {"OLD": 10, "NEW": 100},
            11: {"OLD": 1000, "NEW": 1},
        }
    )
    selected, rolls = build_selection(
        "TEST",
        daily,
        date(2026, 9, 10),
        metadata(),
        prices(tmp_path),
    )
    assert selected["contract"].to_list() == ["OLD", "NEW"]
    assert rolls["reason"][0] == "next_avg_volume_exceeded_current"


@pytest.mark.parametrize(
    "window,expected_roll",
    [(3, False), (1, True)],
)
def test_volume_window_smooths_single_day_flip(tmp_path, window, expected_roll):
    daily = market(
        {
            9: {"OLD": 1_000_000, "NEW": 100_000},
            10: {"OLD": 10_000, "NEW": 20_000},
            11: {"OLD": 900_000, "NEW": 100_000},
        }
    )
    selected, rolls = build_selection(
        "TEST",
        daily,
        date(2026, 9, 10),
        metadata(),
        prices(tmp_path),
        volume_window_days=window,
    )
    if expected_roll:
        assert selected["contract"].to_list() == ["OLD", "NEW"]
        assert rolls.height == 1
        assert rolls["reason"][0] == "next_avg_volume_exceeded_current"
        assert rolls["new_avg_volume"][0] == 20_000.0
    else:
        assert selected["contract"].to_list() == ["OLD", "OLD"]
        assert rolls.is_empty()


def test_initial_selection_requires_warmup():
    with pytest.raises(ValueError, match="для инициализации"):
        build_selection(
            "TEST", market({10: {"OLD": 100}}), date(2026, 9, 10), metadata(), {}
        )


def test_initial_warmup_uses_first_day_when_no_prior_history():
    daily = market({10: {"OLD": 100, "NEW": 10}, 11: {"OLD": 100, "NEW": 10}})
    selected, rolls = build_selection("TEST", daily, date(2026, 9, 10), metadata(), {})
    assert selected["date"].to_list() == [date(2026, 9, 11)]
    assert selected["contract"].to_list() == ["OLD"]
    assert selected["decision_date"][0] == date(2026, 9, 10)
    assert rolls.is_empty()


def test_synchronized_price_uses_common_minute(tmp_path):
    paths = prices(tmp_path)
    pl.DataFrame(
        {
            "timestamp": [datetime(2026, 9, 10, 23, 49), datetime(2026, 9, 10, 23, 50)],
            "close": [110.0, 120.0],
        }
    ).write_parquet(paths["NEW"])
    old, new, timestamp, age_minutes = synchronized_closes(
        paths["OLD"], paths["NEW"], date(2026, 9, 10)
    )
    assert (old, new) == (100.0, 110.0)
    assert timestamp == datetime(2026, 9, 10, 23, 49)
    assert age_minutes == 1.0


@pytest.mark.parametrize(
    "hour, minute, expected_age", [(23, 23, 26.0), (13, 59, 590.0)]
)
def test_stale_synchronized_price_keeps_common_time_and_reports_age(
    tmp_path, hour, minute, expected_age
):
    paths = prices(tmp_path)
    pl.DataFrame(
        {"timestamp": [datetime(2026, 9, 10, hour, minute)], "close": [100.0]}
    ).write_parquet(paths["OLD"])
    pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 9, 10, hour, minute),
                datetime(2026, 9, 10, 23, 49),
            ],
            "close": [110.0, 120.0],
        }
    ).write_parquet(paths["NEW"])
    old, new, timestamp, age_minutes = synchronized_closes(
        paths["OLD"], paths["NEW"], date(2026, 9, 10)
    )
    assert (old, new) == (100.0, 110.0)
    assert timestamp == datetime(2026, 9, 10, hour, minute)
    assert age_minutes == expected_age


def test_no_common_minute_is_still_rejected(tmp_path):
    paths = prices(tmp_path)
    pl.DataFrame(
        {"timestamp": [datetime(2026, 9, 10, 23, 48)], "close": [110.0]}
    ).write_parquet(paths["NEW"])
    with pytest.raises(ValueError, match="Нет общей минуты"):
        synchronized_closes(paths["OLD"], paths["NEW"], date(2026, 9, 10))


def test_weekends_keep_calendar_dates():
    daily = market({11: {"OLD": 100}, 12: {"OLD": 20}, 14: {"OLD": 150}}).rename(
        {"date": "calendar_date"}
    )
    combined, mapping = assign_trading_dates(daily)
    assert combined.height == 3
    assert mapping.filter(pl.col("calendar_date") != pl.col("date")).is_empty()


def test_returns_do_not_use_adjustment_or_cross_roll():
    rows = []
    for minute, contract, price, adjustment in (
        (0, "OLD", 100.0, -50.0),
        (15, "OLD", 101.0, -50.0),
        (30, "NEW", 210.0, -159.0),
        (45, "NEW", 212.0, -159.0),
    ):
        row = {
            "asset": "TEST",
            "contract": contract,
            "date": date(2026, 9, 10),
            "timestamp": datetime(2026, 9, 10, 10, minute),
            "volume": 2,
            "value": 100.0,
            "price_adjustment": adjustment,
            "is_roll_day": contract == "NEW",
        }
        row.update({column: price for column in ("open", "high", "low", "close")})
        row.update(
            {
                f"{column}_adjusted": price + adjustment
                for column in ("open", "high", "low", "close")
            }
        )
        rows.append(row)
    bars = resample_15_minutes(pl.DataFrame(rows))
    assert bars["close"].to_list() == [100.0, 101.0, 210.0, 212.0]
    assert bars["close_adjusted"].to_list() == [50.0, 51.0, 51.0, 53.0]
    assert bars["log_return"][0] is None
    assert bars["log_return"][1] == pytest.approx(math.log(101 / 100))
    assert bars["log_return"][2] is None
    assert bars["log_return"][3] == pytest.approx(math.log(212 / 210))
    assert bars["is_first_bar_after_roll"].to_list() == [False, False, True, False]
    assert bars["has_all_15_minutes"].to_list() == [False] * 4


def test_output_path_cannot_target_raw():
    from data.build_continuous_futures import RAW_DIR

    with pytest.raises(ValueError, match="только для чтения"):
        require_derived_path(RAW_DIR)


def test_output_path_outside_data_and_reference_are_rejected(tmp_path):
    from data.build_continuous_futures import REFERENCE_DIR

    with pytest.raises(ValueError, match="внутри data"):
        require_derived_path(tmp_path)
    with pytest.raises(ValueError, match="только для чтения"):
        require_derived_path(REFERENCE_DIR)


def test_etf_nav_roll_uses_close_over_open_and_charges_cost():
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 9, 10, 10, 0),
                datetime(2026, 9, 10, 10, 15),
                datetime(2026, 9, 10, 10, 30),
                datetime(2026, 9, 10, 10, 45),
            ],
            "contract": ["OLD", "OLD", "NEW", "NEW"],
            "open": [100.0, 101.0, 200.0, 212.0],
            "close": [101.0, 102.0, 210.0, 213.0],
            "is_first_bar_after_roll": [False, False, True, False],
        }
    )
    nav = add_etf_nav(bars, roll_cost_bps=10.0)["nav"].to_list()
    step2 = 102.0 / 101.0
    step3 = step2 * (1.0 - 10.0 / 10_000.0) * (210.0 / 200.0)
    step4 = step3 * (213.0 / 210.0)
    assert nav == pytest.approx([1.0, step2, step3, step4])


def test_etf_nav_does_not_see_the_basis_gap_on_roll():
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 9, 10, 10, 0),
                datetime(2026, 9, 10, 10, 15),
            ],
            "contract": ["OLD", "NEW"],
            "open": [100.0, 150.0],
            "close": [101.0, 150.0],
            "is_first_bar_after_roll": [False, True],
        }
    )
    assert add_etf_nav(bars)["nav"].to_list() == pytest.approx([1.0, 1.0])


def test_etf_nav_rejects_invalid_cost():
    bars = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 9, 10, 10, 0)],
            "contract": ["OLD"],
            "open": [100.0],
            "close": [101.0],
            "is_first_bar_after_roll": [False],
        }
    )
    with pytest.raises(ValueError, match="Издержка перехода"):
        add_etf_nav(bars, roll_cost_bps=-1.0)


def test_registry_alias_and_name_resolve_to_same_directory():
    assert registry_name("MXI") == "mix_mini"
    assert registry_name("mix_mini") == "mix_mini"
    assert registry_name("brent") == "brent"
    assert registry_name("rts") == "rts"


def test_registry_rejects_unknown_and_perpetual():
    with pytest.raises(ValueError, match="Неизвестный актив"):
        registry_name("no_such_asset")
    with pytest.raises(ValueError, match="вечный"):
        registry_name("perpetual")


def test_available_assets_has_plan_markets_and_no_perpetual():
    names = available_assets()
    assert "perpetual" not in names
    assert {"MXI", "CNYRUB", "BRENT", "GOLD"} <= set(names)
    assert {"brent", "rts", "mix_mini"} <= set(names)


def test_select_ready_assets_skips_missing_raw_and_reference(tmp_path):
    (tmp_path / "brent").mkdir()
    (tmp_path / "brent" / "BRM2.parquet").write_bytes(b"")
    (tmp_path / "gold").mkdir()
    (tmp_path / "gold" / "GDM2.parquet").write_bytes(b"")
    candidates = {"BRENT": "brent", "GOLD": "gold", "MXI": "mix_mini"}
    ready, skipped = select_ready_assets(candidates, {"BRM2"}, raw_dir=tmp_path)
    assert ready == {"BRENT": "brent"}
    assert {label for label, _ in skipped} == {"GOLD", "MXI"}


def test_select_requested_assets_all_is_default_scope():
    requested, process_all = select_requested_assets(["all"])
    assert process_all is True
    assert "perpetual" not in requested
    assert {"brent", "rts", "mix_mini"} <= set(requested)


def test_select_requested_assets_plan_is_four_markets():
    requested, process_all = select_requested_assets(["plan"])
    assert process_all is False
    assert requested == ["MXI", "CNYRUB", "BRENT", "GOLD"]


def test_select_requested_assets_explicit_and_conflicts():
    assert select_requested_assets(["MXI", "brent"]) == (["MXI", "brent"], False)
    with pytest.raises(ValueError, match="отдельно"):
        select_requested_assets(["all", "MXI"])
    with pytest.raises(ValueError, match="Повторные"):
        select_requested_assets(["MXI", "MXI"])


def test_select_ready_assets_case_insensitive(tmp_path):
    (tmp_path / "si").mkdir()
    (tmp_path / "si" / "SiZ7.parquet").write_bytes(b"")
    candidates = {"SI": "si"}
    ready, skipped = select_ready_assets(candidates, {"SIZ7"}, raw_dir=tmp_path)
    assert ready == {"SI": "si"}
    assert skipped == []


def test_load_contract_metadata_case_insensitive(tmp_path):
    path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {"ticker": ["SIZ7"], "last_trade_date": [date(2027, 12, 16)]}
    ).write_parquet(path)
    res = load_contract_metadata(["SiZ7"], path)
    assert "SiZ7" in res
    assert res["SiZ7"] == {"LSTTRADE": date(2027, 12, 16)}

