"""M5 configuration, immutable values, and table-contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandera.errors
import polars as pl
import pytest
from pydantic import ValidationError

from volguard.config import DataConfig, EvalConfig, FeatureConfig, load_config
from volguard.datasets.schemas import SPLIT_MANIFEST
from volguard.datasets.types import Fold, SurfacePCA, WindowedDataset
from volguard.features.schemas import DAILY_FEATURES, FEATURE_QC, validate_daily_features
from volguard.features.types import FeatureRunSummary

TENORS = (7, 14, 30, 60, 90, 180)
_TS = pl.Datetime("ms", "UTC")


def _daily_row() -> dict[str, object]:
    snap = datetime(2024, 1, 2, 8, 5, tzinfo=UTC)
    row: dict[str, object] = {
        "snap_date": date(2024, 1, 2),
        "snap_ts": snap,
        "grid_signature": "tenors=7,14,30,60,90,180|moneyness=-2,-1.5,-1,-0.5,0,0.5,1,1.5,2",
        "grid_w": [0.04] * 54,
        "grid_k": [0.0] * 54,
        "grid_cell_n_obs": [5] * 54,
        "grid_interp_flag": [False] * 54,
        "grid_fit_rmse": [0.01] * 54,
        "grid_extrap_flag": [False] * 54,
        "grid_quality_weight": [0.8] * 54,
        "surface_mean_fit_rmse": 0.01,
        "surface_max_fit_rmse": 0.02,
        "surface_min_slice_n_obs": 3,
        "surface_total_slice_n_obs": 30,
        "surface_interp_fraction": 0.0,
        "surface_extrap_fraction": 0.0,
        "surface_all_butterfly_certified": True,
        "surface_all_calendar_certified": True,
        "surface_m4_butterfly_post": 0,
        "surface_m4_calendar_post": 0,
        "surface_model_ok": True,
        "surface_model_butterfly_violations": 0,
        "surface_model_vertical_violations": 0,
        "surface_model_calendar_violations": 0,
        "surface_model_max_butterfly_magnitude": 0.0,
        "surface_model_max_vertical_magnitude": 0.0,
        "surface_model_max_calendar_magnitude": 0.0,
        "surface_model_integrated_butterfly_magnitude": 0.0,
        "surface_model_integrated_vertical_magnitude": 0.0,
        "surface_model_integrated_calendar_magnitude": 0.0,
        "surface_model_max_relative_calendar_deficit": 0.0,
        "surface_model_integrated_relative_calendar_deficit": 0.0,
        "surface_model_minimum_overlap_width": 0.1,
        "surface_model_insufficient_overlap_pairs": 0,
        "surface_quality_n_obs_reference": 5,
        "surface_quality_flags": [],
        "atm_term_slope_7_180": 0.1,
        "rv_parkinson_1d": 0.2,
        "rv_parkinson_5d": 0.2,
        "rv_parkinson_22d": 0.2,
        "rv_garman_klass_1d": 0.2,
        "rv_garman_klass_5d": 0.2,
        "rv_garman_klass_22d": 0.2,
        "underlying_return_1d": 0.01,
        "underlying_log_range": 0.02,
        "jump_flag": False,
        "dvol": 0.6,
        "dvol_change_5d": 0.015,
        "funding_8h": 0.0001,
        "futures_basis_annualized": None,
        "aggregate_oi": None,
        "options_volume_btc": 1000.0,
        "put_call_volume_ratio": 0.8,
        "day_of_week": 1,
        "days_to_monthly_expiry": 24,
        "days_to_quarterly_expiry": 87,
        "max_source_ts": snap,
    }
    for tenor in TENORS:
        row[f"atm_iv_{tenor}d"] = 0.6
        row[f"skew_25delta_{tenor}d"] = -0.05
        row[f"curvature_{tenor}d"] = 0.02
        row[f"iv_rv_spread_{tenor}d"] = 0.4
    for source in ("surface", "underlying", "dvol", "funding", "basis", "oi", "volume"):
        available = source not in {"basis", "oi"}
        row[f"{source}_source_ts"] = snap if available else None
        row[f"{source}_age_s"] = 0.0 if available else None
        row[f"{source}_available"] = available
    return row


def _daily_frame(row: dict[str, object] | None = None) -> pl.DataFrame:
    frame = pl.DataFrame([row or _daily_row()])
    ts_cols = [c for c in frame.columns if c.endswith("_ts")]
    return frame.with_columns(pl.col(ts_cols).cast(_TS))


def test_feature_config_yaml_and_validated_defaults() -> None:
    cfg = load_config("features", FeatureConfig)
    assert cfg.realized_horizons_days == [1, 5, 22]
    assert cfg.jump_lookback_days == 22
    assert cfg.pca_components == 3
    assert cfg.basis_target_days == 30
    assert cfg.lookback_days == 20
    assert cfg.forecast_horizon_days == 1
    assert cfg.model_domain_calendar_points == 9
    assert cfg.quality_n_obs_reference == 5
    assert cfg.quality_interp_weight == 0.5
    assert cfg.quality_extrap_weight == 0.0
    assert cfg.quality_rmse_half_life == 0.05


def test_data_config_exposes_typed_candle_resolutions() -> None:
    config = DataConfig(ohlc_resolution=" 30 ", dvol_resolution="3600")

    assert config.ohlc_resolution_minutes == 30
    assert config.dvol_resolution_seconds == 3_600
    assert config.ohlc_resolution == "30"
    assert config.dvol_resolution == "3600"
    assert DataConfig(ohlc_resolution="1D").ohlc_resolution_minutes == 1_440
    daily_dvol = DataConfig(dvol_resolution="1d")
    assert daily_dvol.dvol_resolution == "1D"
    assert daily_dvol.dvol_resolution_seconds == 86_400


@pytest.mark.parametrize(
    ("field", "value"),
    [("ohlc_resolution", "6"), ("dvol_resolution", "1"), ("dvol_resolution", "1800")],
)
def test_data_config_rejects_unsupported_or_unsafe_resolution(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        DataConfig.model_validate({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("realized_horizons_days", [1, 1]),
        ("pca_components", 0),
        ("basis_target_days", 0),
        ("lookback_days", 0),
        ("forecast_horizon_days", 0),
        ("oi_max_age_s", -1.0),
    ],
)
def test_feature_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        FeatureConfig.model_validate({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_domain_calendar_points", 2),
        ("quality_n_obs_reference", 0),
        ("quality_interp_weight", -0.01),
        ("quality_interp_weight", 1.01),
        ("quality_extrap_weight", -0.01),
        ("quality_extrap_weight", 1.01),
        ("quality_rmse_half_life", 0.0),
    ],
)
def test_feature_config_rejects_invalid_quality_weighting_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        FeatureConfig.model_validate({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("realized_horizons_days", [1]),
        ("dvol_change_days", 1),
    ],
)
def test_feature_config_rejects_horizons_outside_frozen_schema(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="frozen daily feature schema"):
        FeatureConfig.model_validate({field: value})


def test_daily_contract_rejects_nonfinite_scalar() -> None:
    row = _daily_row()
    row["dvol"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_daily_features(_daily_frame(row))


def test_eval_config_exposes_fit_tail_and_rejects_impossible_window() -> None:
    cfg = EvalConfig(initial_train_months=18, val_months=2)
    assert cfg.initial_fit_months == 16
    with pytest.raises(ValidationError):
        EvalConfig(initial_train_months=2, val_months=2)
    with pytest.raises(ValidationError):
        EvalConfig(seeds=[1, 1])


def test_immutable_types_copy_and_freeze_numpy_payloads(tmp_path: Path) -> None:
    fold = Fold(
        fold_id=0,
        train_start=date(2021, 1, 1),
        train_end=date(2022, 5, 1),
        validation_start=date(2022, 5, 1),
        validation_end=date(2022, 7, 1),
        test_start=date(2022, 7, 1),
        test_end=date(2022, 9, 1),
        tune_hyperparameters=True,
    )
    source = np.zeros(54)
    pca = SurfacePCA(
        mean=source,
        components=np.ones((3, 54)),
        explained_variance_ratio=np.ones(3) / 3,
        fit_start=date(2021, 1, 1),
        fit_end=date(2022, 5, 1),
    )
    source_quality = np.zeros((2, 20, 6, 9, 5))
    source_weights = np.ones((2, 6, 9))
    dataset = WindowedDataset(
        x_grid=np.zeros((2, 20, 6, 9)),
        x_grid_quality=source_quality,
        x_features=np.zeros((2, 20, 10)),
        x_feature_mask=np.ones((2, 20, 10), dtype=np.bool_),
        y_grid=np.zeros((2, 6, 9)),
        y_grid_weight=source_weights,
        input_dates=tuple(
            tuple(date(2022, 6, 1) + timedelta(days=j) for j in range(20)) for _ in range(2)
        ),
        target_dates=(date(2022, 7, 2), date(2022, 7, 3)),
        max_source_ts=(datetime(2022, 7, 1, 8, 5, tzinfo=UTC),) * 2,
        splits=("test", "test"),
        feature_names=tuple(f"feature_{i}" for i in range(10)),
        quality_channel_names=(
            "log1p_cell_n_obs",
            "is_interpolated",
            "is_extrapolated",
            "fit_rmse",
            "reliability_weight",
        ),
    )
    summary = FeatureRunSummary(
        accepted_dates=(date(2022, 7, 1),),
        rejected_dates=(date(2022, 7, 2),),
        daily_dir=tmp_path / "daily",
        qc_path=tmp_path / "qc.parquet",
        split_manifest_path=None,
    )
    source[0] = 9.0
    source_quality[0, 0, 0, 0, 0] = 9.0
    source_weights[0, 0, 0] = 0.0
    assert pca.mean[0] == 0.0
    assert dataset.x_grid_quality[0, 0, 0, 0, 0] == 0.0
    assert dataset.y_grid_weight[0, 0, 0] == 1.0
    with pytest.raises(FrozenInstanceError):
        fold.fold_id = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        pca.mean[0] = 1.0
    with pytest.raises(ValueError):
        dataset.x_grid[0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        dataset.x_grid_quality[0, 0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        dataset.y_grid_weight[0, 0, 0] = 0.0
    assert summary.accepted_count == summary.rejected_count == 1


def _invalid_quality_channel(channel: int, value: float) -> np.ndarray:
    quality = np.zeros((1, 2, 6, 9, 5))
    quality[..., channel] = value
    return quality


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"x_grid_quality": np.zeros((1, 2, 6, 9))}, "dimensions"),
        ({"x_grid_quality": np.zeros((1, 3, 6, 9, 5))}, "lookback"),
        (
            {
                "x_features": np.zeros((1, 3, 1)),
                "x_feature_mask": np.ones((1, 3, 1), dtype=np.bool_),
            },
            "lookback",
        ),
        ({"x_grid_quality": np.zeros((1, 2, 6, 8, 5))}, "grid"),
        ({"quality_channel_names": ("only_one",)}, "quality_channel_names"),
        ({"x_grid_quality": np.full((1, 2, 6, 9, 5), np.nan)}, "finite"),
        ({"x_grid_quality": _invalid_quality_channel(0, -0.01)}, "log1p_cell_n_obs"),
        ({"x_grid_quality": _invalid_quality_channel(1, 0.5)}, "is_interpolated"),
        ({"x_grid_quality": _invalid_quality_channel(2, 2.0)}, "is_extrapolated"),
        ({"x_grid_quality": _invalid_quality_channel(3, -0.01)}, "fit_rmse"),
        ({"x_grid_quality": _invalid_quality_channel(4, 1.01)}, "reliability_weight"),
        ({"y_grid_weight": np.full((1, 6, 9), -0.01)}, "between 0 and 1"),
        ({"y_grid_weight": np.full((1, 6, 9), np.inf)}, "finite"),
        (
            {
                "x_grid": np.zeros((1, 2, 5, 9)),
                "x_grid_quality": np.zeros((1, 2, 5, 9, 5)),
                "y_grid": np.zeros((1, 5, 9)),
                "y_grid_weight": np.ones((1, 5, 9)),
            },
            "6x9",
        ),
    ],
)
def test_windowed_dataset_rejects_misaligned_or_invalid_quality_arrays(
    override: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "x_grid": np.zeros((1, 2, 6, 9)),
        "x_grid_quality": np.zeros((1, 2, 6, 9, 5)),
        "x_features": np.zeros((1, 2, 1)),
        "x_feature_mask": np.ones((1, 2, 1), dtype=np.bool_),
        "y_grid": np.zeros((1, 6, 9)),
        "y_grid_weight": np.ones((1, 6, 9)),
        "input_dates": ((date(2024, 1, 1), date(2024, 1, 2)),),
        "target_dates": (date(2024, 1, 3),),
        "max_source_ts": (datetime(2024, 1, 2, 8, 5, tzinfo=UTC),),
        "splits": ("train",),
        "feature_names": ("feature",),
        "quality_channel_names": (
            "log1p_cell_n_obs",
            "is_interpolated",
            "is_extrapolated",
            "fit_rmse",
            "reliability_weight",
        ),
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        WindowedDataset(**values)  # type: ignore[arg-type]


def test_fold_rejects_noncontiguous_boundaries() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        Fold(
            fold_id=0,
            train_start=date(2021, 1, 1),
            train_end=date(2022, 5, 1),
            validation_start=date(2022, 4, 1),
            validation_end=date(2022, 7, 1),
            test_start=date(2022, 7, 1),
            test_end=date(2022, 9, 1),
            tune_hyperparameters=True,
        )


def test_daily_schema_accepts_nullable_basis_and_oi() -> None:
    validated = validate_daily_features(_daily_frame())
    assert validated.height == 1
    DAILY_FEATURES.validate(validated)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([0.04] * 53, "54"),
        ([0.04] * 53 + [-0.01], "nonnegative"),
        ([0.04] * 53 + [float("nan")], "finite"),
    ],
)
def test_daily_helper_rejects_invalid_grid(value: object, message: str) -> None:
    row = _daily_row()
    row["grid_w"] = value
    with pytest.raises(ValueError, match=message):
        validate_daily_features(_daily_frame(row))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("grid_fit_rmse", [0.01] * 53, "54"),
        ("grid_fit_rmse", [0.01] * 53 + [-0.01], "nonnegative"),
        ("grid_quality_weight", [0.5] * 53 + [1.01], "between 0 and 1"),
        ("grid_quality_weight", [0.5] * 53 + [float("nan")], "finite"),
        ("grid_extrap_flag", [False] * 53, "54"),
    ],
)
def test_daily_helper_rejects_invalid_quality_lists(
    column: str, value: object, message: str
) -> None:
    row = _daily_row()
    row[column] = value
    with pytest.raises(ValueError, match=message):
        validate_daily_features(_daily_frame(row))


@pytest.mark.parametrize(
    "flags",
    [
        ["unknown_quality_flag"],
        ["m4_calendar_post", "m4_calendar_post"],
        ["calendar_certification_failed", "m4_calendar_post"],
    ],
)
def test_daily_helper_rejects_unknown_duplicate_or_unordered_quality_flags(
    flags: list[str],
) -> None:
    row = _daily_row()
    row["surface_quality_flags"] = flags
    with pytest.raises(ValueError, match="quality flags"):
        validate_daily_features(_daily_frame(row))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("surface_interp_fraction", 0.5),
        ("surface_extrap_fraction", 0.5),
        ("surface_model_ok", False),
        ("surface_quality_flags", ["m4_calendar_post"]),
    ],
)
def test_daily_helper_rejects_incoherent_surface_quality_fields(field: str, value: object) -> None:
    row = _daily_row()
    row[field] = value
    with pytest.raises(ValueError, match=r"surface quality|fraction|model_ok"):
        validate_daily_features(_daily_frame(row))


def test_daily_helper_rejects_missing_low_support_flag() -> None:
    row = _daily_row()
    row["grid_cell_n_obs"] = [4] * 54
    with pytest.raises(ValueError, match="surface quality flags"):
        validate_daily_features(_daily_frame(row))


def test_daily_helper_rejects_lookahead() -> None:
    row = _daily_row()
    row["max_source_ts"] = row["snap_ts"] + timedelta(seconds=1)  # type: ignore[operator]
    with pytest.raises(ValueError, match="max_source_ts"):
        validate_daily_features(_daily_frame(row))


def test_daily_schema_is_strict() -> None:
    row = _daily_row()
    row["surprise"] = 1
    with pytest.raises(pandera.errors.SchemaError):
        DAILY_FEATURES.validate(_daily_frame(row))


def test_qc_and_split_manifest_contracts() -> None:
    snap = datetime(2024, 1, 2, 8, 5, tzinfo=UTC)
    qc = pl.DataFrame(
        {
            "snap_date": [date(2024, 1, 2)],
            "snap_ts": [snap],
            "status": ["rejected"],
            "reason_code": ["surface_grid_cell_count"],
            "detail": ["expected 54 cells, found 53"],
            "input_rows": [53],
            "grid_cells": [53],
            "output_path": [None],
            "max_source_ts": [None],
        },
        schema_overrides={"output_path": pl.String, "max_source_ts": _TS},
    ).with_columns(pl.col("snap_ts").cast(_TS))
    manifest = pl.DataFrame(
        {
            "fold_id": [0],
            "target_date": [date(2022, 5, 2)],
            "split": ["validation"],
            "train_start": [date(2021, 1, 1)],
            "train_end": [date(2022, 3, 1)],
            "validation_start": [date(2022, 3, 1)],
            "validation_end": [date(2022, 5, 1)],
            "test_start": [date(2022, 5, 1)],
            "test_end": [date(2022, 7, 1)],
            "tune_hyperparameters": [True],
        }
    )
    FEATURE_QC.validate(qc)
    SPLIT_MANIFEST.validate(manifest)
