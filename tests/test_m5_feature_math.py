"""Pure M5 feature calculations and no-lookahead boundaries."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from volguard.config import FeatureConfig, SurfaceConfig
from volguard.features.calendar import calendar_features
from volguard.features.market import build_market_features
from volguard.features.realized import build_realized_features
from volguard.features.surface_factors import SurfaceQualityError, build_surface_features

_GOLDEN = Path(__file__).parent / "fixtures" / "surface_golden.parquet"
_TS = pl.Datetime("ms", "UTC")


def test_surface_features_accept_complete_certified_grid() -> None:
    rows = pl.read_parquet(_GOLDEN).with_columns(
        pl.lit(0).alias("arb_butterfly_post"),
        pl.lit(0).alias("arb_calendar_post"),
        pl.when(pl.col("butterfly_ok").is_not_null())
        .then(pl.lit(True))
        .otherwise(pl.col("butterfly_ok"))
        .alias("butterfly_ok"),
        pl.when(pl.col("calendar_ok").is_not_null())
        .then(pl.lit(True))
        .otherwise(pl.col("calendar_ok"))
        .alias("calendar_ok"),
    )
    snap_date = rows["snap_date"][0]

    result = build_surface_features(rows, snap_date, SurfaceConfig())

    assert len(cast(list[float], result["grid_w"])) == 54
    assert len(cast(list[float], result["grid_k"])) == 54
    assert result["surface_available"] is True
    assert result["surface_source_ts"] == datetime.combine(
        snap_date, datetime.min.time(), tzinfo=UTC
    ).replace(hour=8, minute=5)
    assert cast(float, result["atm_iv_30d"]) > 0.0
    assert result["curvature_30d"] is not None


def test_surface_features_accept_soft_fit_warnings_and_weight_cells() -> None:
    rows = pl.read_parquet(_GOLDEN).with_columns(
        pl.lit(2).alias("arb_butterfly_post"),
        pl.lit(3).alias("arb_calendar_post"),
        pl.when(pl.col("record_kind") == "param")
        .then(pl.lit(False))
        .otherwise(pl.col("butterfly_ok"))
        .alias("butterfly_ok"),
        pl.when(pl.col("record_kind") == "param")
        .then(pl.lit(False))
        .otherwise(pl.col("calendar_ok"))
        .alias("calendar_ok"),
    )

    result = build_surface_features(rows, rows["snap_date"][0], SurfaceConfig(), FeatureConfig())

    assert result["surface_m4_butterfly_post"] == 2
    assert result["surface_m4_calendar_post"] == 3
    assert result["surface_all_butterfly_certified"] is False
    assert result["surface_all_calendar_certified"] is False
    assert cast(list[str], result["surface_quality_flags"])[:4] == [
        "m4_butterfly_post",
        "m4_calendar_post",
        "butterfly_certification_failed",
        "calendar_certification_failed",
    ]
    assert len(cast(list[float], result["grid_fit_rmse"])) == 54
    assert len(cast(list[bool], result["grid_extrap_flag"])) == 54
    assert len(cast(list[float], result["grid_quality_weight"])) == 54
    assert all(0.0 <= value <= 1.0 for value in cast(list[float], result["grid_quality_weight"]))
    assert result["surface_quality_n_obs_reference"] == 5
    interp = cast(list[bool], result["grid_interp_flag"])
    extrap = cast(list[bool], result["grid_extrap_flag"])
    assert result["surface_interp_fraction"] == pytest.approx(
        sum(
            is_interp and not is_extrap for is_interp, is_extrap in zip(interp, extrap, strict=True)
        )
        / 54
    )


def test_surface_features_reject_mispartitioned_date_and_missing_certification() -> None:
    rows = pl.read_parquet(_GOLDEN).with_columns(
        pl.lit(0).alias("arb_butterfly_post"),
        pl.lit(0).alias("arb_calendar_post"),
        pl.lit(None, dtype=pl.Boolean).alias("butterfly_ok"),
        pl.lit(None, dtype=pl.Boolean).alias("calendar_ok"),
    )
    actual_day = rows["snap_date"][0]
    with pytest.raises(SurfaceQualityError, match="partition date"):
        build_surface_features(
            rows, actual_day + timedelta(days=1), SurfaceConfig(), FeatureConfig()
        )
    with pytest.raises(SurfaceQualityError, match="certification"):
        build_surface_features(rows, actual_day, SurfaceConfig(), FeatureConfig())


def test_surface_features_accept_false_but_reject_null_parameter_certification() -> None:
    rows = pl.read_parquet(_GOLDEN)
    first_param_tau = rows.filter(pl.col("record_kind") == "param")["tau"].min()
    rows = rows.with_columns(
        pl.lit(0).alias("arb_butterfly_post"),
        pl.lit(0).alias("arb_calendar_post"),
        pl.when((pl.col("record_kind") == "param") & (pl.col("tau") != first_param_tau))
        .then(pl.lit(True))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("butterfly_ok"),
        pl.when(pl.col("record_kind") == "param")
        .then(pl.lit(True))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("calendar_ok"),
    )

    with pytest.raises(SurfaceQualityError, match="certification"):
        build_surface_features(rows, rows["snap_date"][0], SurfaceConfig(), FeatureConfig())

    valid_false = rows.with_columns(
        pl.when(pl.col("record_kind") == "param")
        .then(pl.lit(False))
        .otherwise(pl.col("butterfly_ok"))
        .alias("butterfly_ok")
    )
    result = build_surface_features(
        valid_false, valid_false["snap_date"][0], SurfaceConfig(), FeatureConfig()
    )
    assert result["surface_all_butterfly_certified"] is False


def test_surface_features_reject_invalid_native_axis_and_parameter_fit() -> None:
    rows = pl.read_parquet(_GOLDEN)
    first_tau = SurfaceConfig().tenor_grid_days[0] / 365.0
    bad_axis = rows.with_columns(
        pl.when(
            (pl.col("record_kind") == "grid")
            & (pl.col("tau") - first_tau).abs().lt(1e-9)
            & (pl.col("moneyness") == 0.0)
        )
        .then(pl.lit(-999.0))
        .otherwise(pl.col("grid_k"))
        .alias("grid_k")
    )
    with pytest.raises(SurfaceQualityError, match="strictly increasing"):
        build_surface_features(bad_axis, bad_axis["snap_date"][0], SurfaceConfig(), FeatureConfig())

    bad_param = rows.with_columns(
        pl.when(pl.col("record_kind") == "param")
        .then(pl.lit(float("nan")))
        .otherwise(pl.col("rmse"))
        .alias("rmse")
    )
    with pytest.raises(SurfaceQualityError, match="rmse"):
        build_surface_features(
            bad_param, bad_param["snap_date"][0], SurfaceConfig(), FeatureConfig()
        )


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("grid_w", float("inf"), "invalid values"),
        ("cell_n_obs", -1, "input contract"),
    ],
)
def test_surface_features_reject_invalid_grid_values(
    column: str, bad_value: float | int, message: str
) -> None:
    rows = pl.read_parquet(_GOLDEN).with_columns(
        pl.when((pl.col("record_kind") == "grid") & (pl.col("moneyness") == 0.0))
        .then(pl.lit(bad_value))
        .otherwise(pl.col(column))
        .alias(column)
    )
    with pytest.raises(SurfaceQualityError, match=message):
        build_surface_features(rows, rows["snap_date"][0], SurfaceConfig(), FeatureConfig())


def test_surface_features_reject_inconsistent_m4_post_counts() -> None:
    rows = (
        pl.read_parquet(_GOLDEN)
        .with_row_index("row_index")
        .with_columns(
            pl.when(pl.col("row_index") == 0)
            .then(pl.col("arb_calendar_post") + 1)
            .otherwise(pl.col("arb_calendar_post"))
            .alias("arb_calendar_post")
        )
        .drop("row_index")
    )
    with pytest.raises(SurfaceQualityError, match="consistent"):
        build_surface_features(rows, rows["snap_date"][0], SurfaceConfig(), FeatureConfig())


def test_surface_features_are_invariant_to_long_format_row_order() -> None:
    rows = pl.read_parquet(_GOLDEN)
    shuffled = rows.reverse()

    expected = build_surface_features(rows, rows["snap_date"][0], SurfaceConfig(), FeatureConfig())
    actual = build_surface_features(
        shuffled, shuffled["snap_date"][0], SurfaceConfig(), FeatureConfig()
    )

    quality_fields = {
        name for name in expected if name.startswith("surface_") or name.startswith("grid_")
    }
    assert {name: actual[name] for name in quality_fields} == {
        name: expected[name] for name in quality_fields
    }


def test_surface_features_reject_duplicate_parameter_taus() -> None:
    rows = pl.read_parquet(_GOLDEN)
    params = rows.filter(pl.col("record_kind") == "param")
    duplicate_tau = params["tau"][0]
    second_tau = params["tau"][1]
    duplicated = rows.with_columns(
        pl.when((pl.col("record_kind") == "param") & (pl.col("tau") == second_tau))
        .then(pl.lit(duplicate_tau))
        .otherwise(pl.col("tau"))
        .alias("tau")
    )
    with pytest.raises(SurfaceQualityError, match="unique"):
        build_surface_features(
            duplicated, duplicated["snap_date"][0], SurfaceConfig(), FeatureConfig()
        )


def test_surface_features_flag_low_support_and_ordinary_interpolation() -> None:
    rows = pl.read_parquet(_GOLDEN).with_columns(
        pl.when(pl.col("record_kind") == "grid")
        .then(pl.lit(True))
        .otherwise(pl.col("interp_flag"))
        .alias("interp_flag")
    )
    result = build_surface_features(
        rows,
        rows["snap_date"][0],
        SurfaceConfig(),
        FeatureConfig(quality_n_obs_reference=10_000),
    )
    flags = cast(list[str], result["surface_quality_flags"])
    assert "low_observation_support" in flags
    assert "interpolated_grid_cells" in flags
    assert flags.index("low_observation_support") < flags.index("interpolated_grid_cells")


@pytest.mark.parametrize("mutation", ["unknown_record_kind", "missing_upstream_column"])
def test_surface_features_reject_invalid_full_m4_contract(mutation: str) -> None:
    rows = pl.read_parquet(_GOLDEN)
    if mutation == "unknown_record_kind":
        rows = (
            rows.with_row_index("row_index")
            .with_columns(
                pl.when(pl.col("row_index") == 0)
                .then(pl.lit("unknown"))
                .otherwise(pl.col("record_kind"))
                .alias("record_kind")
            )
            .drop("row_index")
        )
    else:
        rows = rows.drop("vega_sum")
    with pytest.raises(SurfaceQualityError, match="input contract"):
        build_surface_features(rows, rows["snap_date"][0], SurfaceConfig(), FeatureConfig())


def test_surface_quality_weight_decreases_with_support_interpolation_and_rmse() -> None:
    rows = pl.read_parquet(_GOLDEN)
    params = rows.filter(pl.col("record_kind") == "param")
    first_tau = params["tau"][0]
    rows = rows.with_columns(
        pl.when((pl.col("record_kind") == "param") & (pl.col("tau") == first_tau))
        .then(pl.lit(0.1))
        .otherwise(pl.col("rmse"))
        .alias("rmse")
    )
    result = build_surface_features(rows, rows["snap_date"][0], SurfaceConfig(), FeatureConfig())
    weights = cast(list[float], result["grid_quality_weight"])
    counts = cast(list[int], result["grid_cell_n_obs"])
    interp = cast(list[bool], result["grid_interp_flag"])
    extrap = cast(list[bool], result["grid_extrap_flag"])
    rmses = cast(list[float], result["grid_fit_rmse"])
    expected = [
        min(n_obs / 5.0, 1.0)
        * (0.0 if is_extrap else 0.5 if is_interp else 1.0)
        * 2 ** (-rmse / 0.05)
        for n_obs, is_interp, is_extrap, rmse in zip(counts, interp, extrap, rmses, strict=True)
    ]
    assert weights == pytest.approx(expected)


def test_surface_features_use_configured_snap_time() -> None:
    rows = pl.read_parquet(_GOLDEN).with_columns(
        pl.lit(0).alias("arb_butterfly_post"),
        pl.lit(0).alias("arb_calendar_post"),
        pl.when(pl.col("butterfly_ok").is_not_null())
        .then(pl.lit(True))
        .otherwise(pl.col("butterfly_ok"))
        .alias("butterfly_ok"),
        pl.when(pl.col("calendar_ok").is_not_null())
        .then(pl.lit(True))
        .otherwise(pl.col("calendar_ok"))
        .alias("calendar_ok"),
    )
    result = build_surface_features(
        rows,
        rows["snap_date"][0],
        SurfaceConfig(snap_hour_utc=7, snap_minute_utc=15),
        FeatureConfig(),
    )
    assert cast(datetime, result["snap_ts"]).time() == datetime.min.time().replace(
        hour=7, minute=15
    )


def test_realized_features_exclude_incomplete_hourly_candle() -> None:
    snap_day = date(2024, 1, 2)
    complete_times = [
        datetime(2024, 1, 1, 8, 0, tzinfo=UTC) + timedelta(hours=offset) for offset in range(24)
    ]
    rows = pl.DataFrame(
        {
            "ts": [*complete_times, datetime(2024, 1, 2, 8, 0, tzinfo=UTC)],
            "instrument": ["BTC-PERPETUAL"] * 25,
            "open": [100.0] * 24 + [200.0],
            "high": [110.0] * 24 + [220.0],
            "low": [90.0] * 24 + [180.0],
            "close": [105.0] * 24 + [210.0],
            "volume": [1.0] * 25,
        }
    ).with_columns(pl.col("ts").cast(_TS))
    cfg = FeatureConfig(jump_lookback_days=1)

    result = build_realized_features(rows, [snap_day], cfg)[snap_day]

    assert result["underlying_source_ts"] == datetime(2024, 1, 2, 8, 0, tzinfo=UTC)
    assert result["underlying_age_s"] == pytest.approx(300.0)
    assert result["underlying_log_range"] == pytest.approx(__import__("math").log(110 / 90))
    assert cast(float, result["rv_parkinson_1d"]) > 0.0


def test_realized_features_reject_incomplete_days_and_do_not_compress_gaps() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    times = [
        datetime(2024, 1, 1, 8, 0, tzinfo=UTC) + timedelta(hours=offset)
        for offset in range(48)
        if offset != 5
    ]
    rows = pl.DataFrame(
        {
            "ts": times,
            "open": [100.0] * len(times),
            "high": [110.0] * len(times),
            "low": [90.0] * len(times),
            "close": [105.0] * len(times),
        }
    ).with_columns(pl.col("ts").cast(_TS))

    result = build_realized_features(
        rows,
        days,
        FeatureConfig(jump_lookback_days=1),
    )

    assert result[days[0]]["underlying_available"] is False
    assert result[days[1]]["underlying_return_1d"] is None


def test_realized_features_treat_stale_complete_day_as_a_history_gap() -> None:
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(4)]
    times: list[datetime] = []
    prices: list[float] = []
    for day, price in zip(days, [100.0, 101.0, 150.0, 102.0], strict=True):
        if day == days[2]:
            # This bucket has the expected count, but its final candle closes
            # at 06:00 and is stale at the 08:05 feature snap.
            first = datetime.combine(day - timedelta(days=1), datetime.min.time(), UTC).replace(
                hour=17, minute=30
            )
            day_times = [first + timedelta(minutes=30 * offset) for offset in range(24)]
        else:
            first = datetime.combine(day - timedelta(days=1), datetime.min.time(), UTC).replace(
                hour=8
            )
            day_times = [first + timedelta(hours=offset) for offset in range(24)]
        times.extend(day_times)
        prices.extend([price] * len(day_times))
    rows = pl.DataFrame(
        {
            "ts": times,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
        }
    ).with_columns(pl.col("ts").cast(_TS))

    result = build_realized_features(rows, days, FeatureConfig(jump_lookback_days=1))

    assert result[days[2]]["underlying_available"] is False
    assert result[days[2]]["jump_flag"] is None
    assert result[days[3]]["underlying_available"] is True
    assert result[days[3]]["underlying_return_1d"] is None
    assert result[days[3]]["jump_flag"] is None


def test_realized_features_use_configured_snap_time() -> None:
    snap_day = date(2024, 1, 2)
    times = [
        datetime(2024, 1, 1, 7, 0, tzinfo=UTC) + timedelta(hours=offset) for offset in range(24)
    ]
    rows = pl.DataFrame(
        {
            "ts": times,
            "open": [100.0] * 24,
            "high": [110.0] * 24,
            "low": [90.0] * 24,
            "close": [105.0] * 24,
        }
    ).with_columns(pl.col("ts").cast(_TS))

    result = build_realized_features(
        rows,
        [snap_day],
        FeatureConfig(jump_lookback_days=1),
        snap_hour_utc=7,
        snap_minute_utc=5,
    )[snap_day]

    assert result["underlying_source_ts"] == datetime(2024, 1, 2, 7, 0, tzinfo=UTC)


def _empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def test_market_features_are_nullable_when_sources_missing() -> None:
    snap = datetime(2024, 1, 2, 8, 5, tzinfo=UTC)
    result = build_market_features(snap, FeatureConfig())
    assert result["dvol"] is None
    assert result["futures_basis_annualized"] is None
    assert result["aggregate_oi"] is None
    assert result["basis_available"] is False
    assert result["oi_available"] is False


def test_market_volume_boundary_and_oi_collector_precedence() -> None:
    snap = datetime(2024, 1, 2, 8, 5, tzinfo=UTC)
    trades = pl.DataFrame(
        {
            "ts": [snap - timedelta(days=1), snap - timedelta(hours=1), snap],
            "cp": ["P", "P", "C"],
            "amount": [100.0, 4.0, 2.0],
        }
    ).with_columns(pl.col("ts").cast(_TS))
    oi = pl.DataFrame(
        {
            "source_ts": [snap - timedelta(hours=1), snap - timedelta(hours=2)],
            "instrument": ["BTC-A", "BTC-A"],
            "open_interest": [10.0, 999.0],
            "source": ["collector", "tardis"],
        }
    ).with_columns(pl.col("source_ts").cast(_TS))

    result = build_market_features(snap, FeatureConfig(), option_trades=trades, oi_snapshots=oi)

    assert result["options_volume_btc"] == pytest.approx(6.0)
    assert result["put_call_volume_ratio"] == pytest.approx(2.0)
    assert result["aggregate_oi"] == pytest.approx(10.0)
    assert result["oi_available"] is True


def test_market_oi_excludes_stale_instruments_and_tardis_must_be_same_day() -> None:
    snap = datetime(2024, 1, 2, 8, 5, tzinfo=UTC)
    oi = pl.DataFrame(
        {
            "source_ts": [
                snap - timedelta(hours=1),
                snap - timedelta(days=2),
                snap - timedelta(hours=12),
            ],
            "instrument": ["BTC-A", "BTC-B", "BTC-C"],
            "open_interest": [10.0, 1_000.0, 500.0],
            "source": ["collector", "collector", "tardis"],
        }
    ).with_columns(pl.col("source_ts").cast(_TS))
    result = build_market_features(snap, FeatureConfig(), oi_snapshots=oi)
    assert result["aggregate_oi"] == pytest.approx(10.0)

    tardis_only = oi.filter(pl.col("source") == "tardis")
    missing = build_market_features(snap, FeatureConfig(), oi_snapshots=tardis_only)
    assert missing["aggregate_oi"] is None
    assert missing["oi_available"] is False


def test_market_dvol_waits_for_candle_close_and_basis_selects_target() -> None:
    snap = datetime(2024, 1, 2, 8, 5, tzinfo=UTC)
    dvol = pl.DataFrame(
        {
            "ts": [snap.replace(hour=7, minute=0), snap.replace(hour=8, minute=0)],
            "close": [60.0, 99.0],
        }
    ).with_columns(pl.col("ts").cast(_TS))
    futures = pl.DataFrame(
        {
            "ts": [snap - timedelta(minutes=5), snap - timedelta(minutes=5)],
            "instrument": ["BTC-12JAN24", "BTC-02FEB24"],
            "price": [101.0, 102.0],
            "index_price": [100.0, 100.0],
        }
    ).with_columns(pl.col("ts").cast(_TS))

    result = build_market_features(snap, FeatureConfig(), dvol=dvol, future_trades=futures)

    assert result["dvol"] == pytest.approx(0.60)
    assert result["futures_basis_annualized"] is not None
    assert result["basis_available"] is True


def test_market_dvol_uses_configured_candle_resolution() -> None:
    snap = datetime(2024, 1, 2, 8, 5, tzinfo=UTC)
    dvol = pl.DataFrame({"ts": [snap.replace(minute=4)], "close": [60.0]}).with_columns(
        pl.col("ts").cast(_TS)
    )

    result = build_market_features(
        snap,
        FeatureConfig(),
        dvol=dvol,
        dvol_resolution_seconds=60,
    )

    assert result["dvol"] == pytest.approx(0.60)
    assert result["dvol_source_ts"] == snap


def test_market_dvol_change_is_null_when_target_observation_is_stale() -> None:
    snap = datetime(2024, 1, 10, 8, 5, tzinfo=UTC)
    dvol = pl.DataFrame(
        {
            "ts": [snap - timedelta(days=20), snap - timedelta(hours=2)],
            "close": [40.0, 60.0],
        }
    ).with_columns(pl.col("ts").cast(_TS))

    result = build_market_features(snap, FeatureConfig(), dvol=dvol)

    assert result["dvol"] == pytest.approx(0.60)
    assert result["dvol_change_5d"] is None


def test_market_features_ignore_nonfinite_and_out_of_domain_values() -> None:
    snap = datetime(2024, 1, 10, 8, 5, tzinfo=UTC)
    dvol = pl.DataFrame({"ts": [snap - timedelta(hours=2)], "close": [float("inf")]})
    funding = pl.DataFrame({"ts": [snap], "interest_8h": [float("nan")]})
    trades = pl.DataFrame({"ts": [snap], "cp": ["C"], "amount": [-1.0]})
    futures = pl.DataFrame(
        {
            "ts": [snap],
            "instrument": ["BTC-09FEB24"],
            "price": [101.0],
            "index_price": [0.0],
        }
    )
    oi = pl.DataFrame(
        {
            "source_ts": [snap],
            "instrument": ["BTC-A"],
            "open_interest": [float("inf")],
            "source": ["collector"],
        }
    )

    result = build_market_features(
        snap,
        FeatureConfig(),
        dvol=dvol,
        funding=funding,
        option_trades=trades,
        future_trades=futures,
        oi_snapshots=oi,
    )

    assert result["dvol"] is None
    assert result["funding_8h"] is None
    assert result["options_volume_btc"] is None
    assert result["futures_basis_annualized"] is None
    assert result["aggregate_oi"] is None


def test_calendar_rolls_expired_friday_and_quarter() -> None:
    after_monthly = datetime(2024, 3, 29, 8, 5, tzinfo=UTC)
    result = calendar_features(after_monthly)
    assert result["day_of_week"] == 4
    assert result["days_to_monthly_expiry"] == 28
    assert result["days_to_quarterly_expiry"] == 91
