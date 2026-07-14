"""M5 daily feature pipeline and CLI integration tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from volguard.cli import app
from volguard.config import DataConfig, EvalConfig, FeatureConfig, SurfaceConfig
from volguard.datasets.schemas import SPLIT_MANIFEST
from volguard.features.pipeline import _assert_safe_output_path, run_features
from volguard.features.schemas import DAILY_FEATURES, FEATURE_QC

_GOLDEN = Path(__file__).parent / "fixtures" / "surface_golden.parquet"


def _surface(tmp_path: Path, *, clean: bool) -> tuple[DataConfig, date]:
    rows = pl.read_parquet(_GOLDEN)
    day = rows["snap_date"][0]
    post = 0 if clean else 1
    rows = rows.with_columns(
        pl.lit(post).alias("arb_butterfly_post"),
        pl.lit(post).alias("arb_calendar_post"),
        pl.when(pl.col("butterfly_ok").is_not_null())
        .then(pl.lit(clean))
        .otherwise(pl.col("butterfly_ok"))
        .alias("butterfly_ok"),
        pl.when(pl.col("calendar_ok").is_not_null())
        .then(pl.lit(clean))
        .otherwise(pl.col("calendar_ok"))
        .alias("calendar_ok"),
    )
    curated = tmp_path / "curated"
    part = curated / "surfaces_daily" / f"date={day.isoformat()}" / "part.parquet"
    part.parent.mkdir(parents=True)
    rows.write_parquet(part)
    return (
        DataConfig(
            raw_dir=tmp_path / "raw",
            curated_dir=curated,
            features_dir=tmp_path / "features",
        ),
        day,
    )


def test_run_features_writes_valid_nullable_daily_row_and_qc(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)

    summary = run_features(
        FeatureConfig(),
        DataConfig.model_validate(data_cfg),
        SurfaceConfig(),
        EvalConfig(),
        start=day,
        end=day,
    )

    assert summary.accepted_dates == (day,)
    part = data_cfg.features_dir / "daily" / f"date={day.isoformat()}" / "part.parquet"
    daily = pl.read_parquet(part)
    DAILY_FEATURES.validate(daily)
    assert daily["basis_available"][0] is False
    assert daily["oi_available"][0] is False
    FEATURE_QC.validate(pl.read_parquet(summary.qc_path))
    assert summary.split_manifest_path is not None
    SPLIT_MANIFEST.validate(pl.read_parquet(summary.split_manifest_path))


def test_warning_rerun_overwrites_stale_feature_part_and_is_audited(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=False)
    stale = data_cfg.features_dir / "daily" / f"date={day.isoformat()}" / "part.parquet"
    stale.parent.mkdir(parents=True)
    pl.DataFrame({"stale": [True]}).write_parquet(stale)

    summary = run_features(
        FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=day, end=day
    )

    assert summary.accepted_dates == (day,)
    assert summary.rejected_dates == ()
    assert stale.exists()
    assert "grid_quality_weight" in pl.read_parquet(stale).columns
    qc = pl.read_parquet(summary.qc_path)
    assert qc["status"].to_list() == ["accepted"]
    assert qc["reason_code"].to_list() == ["surface_quality_warning"]
    assert qc["detail"][0].split(",")[:4] == [
        "m4_butterfly_post",
        "m4_calendar_post",
        "butterfly_certification_failed",
        "calendar_certification_failed",
    ]


def test_structural_rejection_removes_exact_stale_feature_part(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)
    source = data_cfg.curated_dir / "surfaces_daily" / f"date={day}" / "part.parquet"
    rows = pl.read_parquet(source)
    first_grid = rows.filter(pl.col("record_kind") == "grid").row(0, named=True)
    rows.filter(
        ~(
            (pl.col("record_kind") == "grid")
            & (pl.col("tau") == first_grid["tau"])
            & (pl.col("moneyness") == first_grid["moneyness"])
        )
    ).write_parquet(source)
    stale = data_cfg.features_dir / "daily" / f"date={day}" / "part.parquet"
    stale.parent.mkdir(parents=True)
    pl.DataFrame({"stale": [True]}).write_parquet(stale)

    summary = run_features(
        FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=day, end=day
    )

    assert summary.rejected_dates == (day,)
    assert not stale.exists()
    qc = pl.read_parquet(summary.qc_path)
    assert qc["reason_code"].to_list() == ["surface_grid_cell_count"]


def test_bounded_rerun_preserves_unrelated_qc_rows(tmp_path: Path) -> None:
    data_cfg, first_day = _surface(tmp_path, clean=True)
    first_part = data_cfg.curated_dir / "surfaces_daily" / f"date={first_day}" / "part.parquet"
    second_day = first_day + date.resolution
    second_part = data_cfg.curated_dir / "surfaces_daily" / f"date={second_day}" / "part.parquet"
    second_part.parent.mkdir(parents=True)
    pl.read_parquet(first_part).with_columns(pl.lit(second_day).alias("snap_date")).write_parquet(
        second_part
    )

    run_features(
        FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=first_day, end=first_day
    )
    run_features(
        FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=second_day, end=second_day
    )

    qc = pl.read_parquet(data_cfg.features_dir / "qc" / "part.parquet")
    assert qc["snap_date"].to_list() == [first_day, second_day]


def test_bounded_run_loads_prior_collector_oi_within_staleness(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)
    prior_day = day - date.resolution
    path = data_cfg.raw_dir / "ticker_snapshots" / f"tickers_{prior_day}.ndjson"
    path.parent.mkdir(parents=True)
    snapshot = {
        "snap_ts": f"{prior_day.isoformat()}T12:00:00+00:00",
        "book_summary": [{"instrument_name": "BTC-TEST", "open_interest": 12.0}],
    }
    path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")

    run_features(FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=day, end=day)

    daily = pl.read_parquet(data_cfg.features_dir / "daily" / f"date={day}" / "part.parquet")
    assert daily["aggregate_oi"][0] == 12.0
    assert daily["oi_available"][0] is True


def test_trade_inputs_are_read_through_bounded_parquet_ranges(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)
    snap = datetime(day.year, day.month, day.day, 8, 5, tzinfo=UTC)
    trades_path = data_cfg.raw_dir / "trades_options" / "month=2021-05" / "part.parquet"
    trades_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": [snap - timedelta(hours=1), snap + timedelta(hours=1)],
            "cp": ["C", "P"],
            "amount": [2.0, 100.0],
        }
    ).write_parquet(trades_path)

    run_features(FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=day, end=day)

    daily = pl.read_parquet(data_cfg.features_dir / "daily" / f"date={day}" / "part.parquet")
    assert daily["options_volume_btc"][0] == 2.0


def test_bounded_run_includes_first_required_warmup_candle(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)
    first_candle = datetime.combine(day - timedelta(days=24), datetime.min.time(), UTC).replace(
        hour=8
    )
    candle_times = [first_candle + timedelta(hours=offset) for offset in range(24 * 24)]
    prices = [100.0] * (24 * 24)
    underlying = data_cfg.raw_dir / "index_ohlc" / "part.parquet"
    underlying.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": candle_times,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
        }
    ).write_parquet(underlying)

    run_features(FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=day, end=day)

    daily = pl.read_parquet(data_cfg.features_dir / "daily" / f"date={day}" / "part.parquet")
    assert daily["rv_parkinson_22d"][0] is not None
    assert daily["jump_flag"][0] is False


def test_run_features_uses_configured_ohlc_and_dvol_resolutions(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)
    data_cfg = DataConfig.model_validate(
        {
            **data_cfg.model_dump(),
            "ohlc_resolution": "30",
            "dvol_resolution": "60",
        }
    )
    first_candle = datetime.combine(day - timedelta(days=1), datetime.min.time(), UTC).replace(
        hour=8
    )
    candle_times = [first_candle + timedelta(minutes=30 * offset) for offset in range(48)]
    underlying = data_cfg.raw_dir / "index_ohlc" / "part.parquet"
    underlying.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": candle_times,
            "open": [100.0] * 48,
            "high": [101.0] * 48,
            "low": [99.0] * 48,
            "close": [100.0] * 48,
        }
    ).write_parquet(underlying)
    dvol = data_cfg.raw_dir / "dvol" / "part.parquet"
    dvol.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": [datetime(day.year, day.month, day.day, 8, 4, tzinfo=UTC)],
            "close": [60.0],
        }
    ).write_parquet(dvol)

    run_features(FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=day, end=day)

    daily = pl.read_parquet(data_cfg.features_dir / "daily" / f"date={day}" / "part.parquet")
    assert daily["rv_parkinson_1d"][0] is not None
    assert daily["underlying_source_ts"][0] == datetime(day.year, day.month, day.day, 8, tzinfo=UTC)
    assert daily["dvol"][0] == pytest.approx(0.60)
    assert daily["dvol_source_ts"][0] == datetime(day.year, day.month, day.day, 8, 5, tzinfo=UTC)


def test_malformed_surface_is_rejected_and_audited_without_blocking_valid_day(
    tmp_path: Path,
) -> None:
    data_cfg, first_day = _surface(tmp_path, clean=True)
    second_day = first_day + date.resolution
    malformed = data_cfg.curated_dir / "surfaces_daily" / f"date={second_day}" / "part.parquet"
    malformed.parent.mkdir(parents=True)
    pl.DataFrame({"snap_date": [second_day]}).write_parquet(malformed)

    summary = run_features(
        FeatureConfig(),
        data_cfg,
        SurfaceConfig(),
        EvalConfig(),
        start=first_day,
        end=second_day,
    )

    first_output = data_cfg.features_dir / "daily" / f"date={first_day}" / "part.parquet"
    assert first_output.exists()
    assert summary.accepted_dates == (first_day,)
    assert summary.rejected_dates == (second_day,)
    qc = pl.read_parquet(summary.qc_path)
    assert qc.filter(pl.col("snap_date") == second_day)["reason_code"][0] == "surface_input_invalid"


def test_output_path_guard_rejects_escape_from_expected_root(tmp_path: Path) -> None:
    root = tmp_path / "features"
    outside = tmp_path / "outside" / "part.parquet"
    with pytest.raises(ValueError, match="outside expected root"):
        _assert_safe_output_path(outside, root)


def test_requested_range_is_clamped_to_available_surface_span(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)
    summary = run_features(
        FeatureConfig(),
        data_cfg,
        SurfaceConfig(),
        EvalConfig(),
        start=date.min,
        end=date.max,
    )
    assert summary.accepted_dates == (day,)


def test_corrupt_optional_raw_sources_do_not_reject_valid_surface(tmp_path: Path) -> None:
    data_cfg, day = _surface(tmp_path, clean=True)
    bad_dvol = data_cfg.raw_dir / "dvol" / "part.parquet"
    bad_dvol.parent.mkdir(parents=True)
    bad_dvol.write_bytes(b"not parquet")
    underlying = data_cfg.raw_dir / "index_ohlc" / "part.parquet"
    underlying.parent.mkdir(parents=True)
    candle_times = [
        datetime(day.year, day.month, day.day, 8, tzinfo=UTC)
        - timedelta(days=1)
        + timedelta(hours=offset)
        for offset in range(24)
    ]
    lows = [90.0] * 24
    closes = [105.0] * 24
    lows[3] = 0.0
    closes[5] = float("inf")
    pl.DataFrame(
        {
            "ts": candle_times,
            "open": [100.0] * 24,
            "high": [110.0] * 24,
            "low": lows,
            "close": closes,
        }
    ).write_parquet(underlying)
    bad_tardis = data_cfg.raw_dir / "tardis_chain" / f"date={day}" / "part.parquet"
    bad_tardis.parent.mkdir(parents=True)
    bad_tardis.write_bytes(b"not parquet either")

    summary = run_features(
        FeatureConfig(), data_cfg, SurfaceConfig(), EvalConfig(), start=day, end=day
    )

    assert summary.accepted_dates == (day,)
    daily = pl.read_parquet(data_cfg.features_dir / "daily" / f"date={day}" / "part.parquet")
    assert daily["dvol_available"][0] is False
    assert daily["underlying_available"][0] is False
    assert daily["oi_available"][0] is False


def test_features_cli_accepts_date_bounds(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return type("Summary", (), {"accepted_count": 1, "rejected_count": 0})()

    monkeypatch.setattr("volguard.features.pipeline.run_features", fake_run)
    runner = CliRunner()
    result = runner.invoke(app, ["features", "--start", "2024-01-01", "--end", "2024-01-02"])

    assert result.exit_code == 0, result.output
    assert seen == {"start": date(2024, 1, 1), "end": date(2024, 1, 2)}
