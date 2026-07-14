"""Strict frozen contracts for M5 daily features and audit rows."""

from __future__ import annotations

import hashlib
import json
import math
from typing import cast

import numpy as np
import pandera.polars as pa
import polars as pl

from volguard.features.surface_quality import SURFACE_QUALITY_FLAG_ORDER
from volguard.ingest.schemas import validate

__all__ = [
    "DAILY_FEATURES",
    "FEATURE_QC",
    "grid_signature",
    "validate",
    "validate_daily_features",
]

GRID_CELL_COUNT = 54
TENORS_DAYS = (7, 14, 30, 60, 90, 180)
SOURCE_GROUPS = ("surface", "underlying", "dvol", "funding", "basis", "oi", "volume")
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")


def _float(*, nullable: bool = True, nonnegative: bool = False) -> pa.Column:
    check = pa.Check.ge(0.0) if nonnegative else None
    return pa.Column(pl.Float64, check, nullable=nullable)


_DAILY_COLUMNS: dict[str, pa.Column] = {
    "snap_date": pa.Column(pl.Date),
    "snap_ts": pa.Column(_TS),
    "grid_signature": pa.Column(pl.String),
    "grid_w": pa.Column(pl.List(pl.Float64)),
    "grid_k": pa.Column(pl.List(pl.Float64)),
    "grid_cell_n_obs": pa.Column(pl.List(pl.Int64)),
    "grid_interp_flag": pa.Column(pl.List(pl.Boolean)),
    "grid_fit_rmse": pa.Column(pl.List(pl.Float64)),
    "grid_extrap_flag": pa.Column(pl.List(pl.Boolean)),
    "grid_quality_weight": pa.Column(pl.List(pl.Float64)),
    "surface_mean_fit_rmse": _float(nullable=False, nonnegative=True),
    "surface_max_fit_rmse": _float(nullable=False, nonnegative=True),
    "surface_min_slice_n_obs": pa.Column(pl.Int64, pa.Check.gt(0)),
    "surface_total_slice_n_obs": pa.Column(pl.Int64, pa.Check.gt(0)),
    "surface_interp_fraction": pa.Column(pl.Float64, pa.Check.in_range(0.0, 1.0)),
    "surface_extrap_fraction": pa.Column(pl.Float64, pa.Check.in_range(0.0, 1.0)),
    "surface_all_butterfly_certified": pa.Column(pl.Boolean),
    "surface_all_calendar_certified": pa.Column(pl.Boolean),
    "surface_m4_butterfly_post": pa.Column(pl.Int64, pa.Check.ge(0)),
    "surface_m4_calendar_post": pa.Column(pl.Int64, pa.Check.ge(0)),
    "surface_model_ok": pa.Column(pl.Boolean),
    "surface_model_butterfly_violations": pa.Column(pl.Int64, pa.Check.ge(0)),
    "surface_model_vertical_violations": pa.Column(pl.Int64, pa.Check.ge(0)),
    "surface_model_calendar_violations": pa.Column(pl.Int64, pa.Check.ge(0)),
    "surface_model_max_butterfly_magnitude": _float(nullable=False, nonnegative=True),
    "surface_model_max_vertical_magnitude": _float(nullable=False, nonnegative=True),
    "surface_model_max_calendar_magnitude": _float(nullable=False, nonnegative=True),
    "surface_model_integrated_butterfly_magnitude": _float(nullable=False, nonnegative=True),
    "surface_model_integrated_vertical_magnitude": _float(nullable=False, nonnegative=True),
    "surface_model_integrated_calendar_magnitude": _float(nullable=False, nonnegative=True),
    "surface_model_max_relative_calendar_deficit": _float(nullable=False, nonnegative=True),
    "surface_model_integrated_relative_calendar_deficit": _float(nullable=False, nonnegative=True),
    "surface_model_minimum_overlap_width": _float(nullable=False, nonnegative=True),
    "surface_model_insufficient_overlap_pairs": pa.Column(pl.Int64, pa.Check.ge(0)),
    "surface_quality_n_obs_reference": pa.Column(pl.Int64, pa.Check.gt(0)),
    "surface_quality_flags": pa.Column(pl.List(pl.String)),
}
for _tenor in TENORS_DAYS:
    _DAILY_COLUMNS.update(
        {
            f"atm_iv_{_tenor}d": _float(nonnegative=True),
            f"skew_25delta_{_tenor}d": _float(),
            f"curvature_{_tenor}d": _float(),
            f"iv_rv_spread_{_tenor}d": _float(),
        }
    )
_DAILY_COLUMNS.update(
    {
        "atm_term_slope_7_180": _float(),
        "rv_parkinson_1d": _float(nonnegative=True),
        "rv_parkinson_5d": _float(nonnegative=True),
        "rv_parkinson_22d": _float(nonnegative=True),
        "rv_garman_klass_1d": _float(nonnegative=True),
        "rv_garman_klass_5d": _float(nonnegative=True),
        "rv_garman_klass_22d": _float(nonnegative=True),
        "underlying_return_1d": _float(),
        "underlying_log_range": _float(nonnegative=True),
        "jump_flag": pa.Column(pl.Boolean, nullable=True),
        "dvol": _float(nonnegative=True),
        "dvol_change_5d": _float(),
        "funding_8h": _float(),
        "futures_basis_annualized": _float(),
        "aggregate_oi": _float(nonnegative=True),
        "options_volume_btc": _float(nonnegative=True),
        "put_call_volume_ratio": _float(nonnegative=True),
        "day_of_week": pa.Column(pl.Int64, pa.Check.in_range(0, 6)),
        "days_to_monthly_expiry": pa.Column(pl.Int64, pa.Check.ge(0)),
        "days_to_quarterly_expiry": pa.Column(pl.Int64, pa.Check.ge(0)),
    }
)
for _source in SOURCE_GROUPS:
    _DAILY_COLUMNS.update(
        {
            f"{_source}_source_ts": pa.Column(_TS, nullable=True),
            f"{_source}_age_s": _float(nonnegative=True),
            f"{_source}_available": pa.Column(pl.Boolean),
        }
    )
_DAILY_COLUMNS["max_source_ts"] = pa.Column(_TS)

DAILY_FEATURES = pa.DataFrameSchema(_DAILY_COLUMNS, strict=True, coerce=True)

FEATURE_QC = pa.DataFrameSchema(
    {
        "snap_date": pa.Column(pl.Date),
        "snap_ts": pa.Column(_TS),
        "status": pa.Column(pl.String, pa.Check.isin(["accepted", "rejected"])),
        "reason_code": pa.Column(pl.String),
        "detail": pa.Column(pl.String),
        "input_rows": pa.Column(pl.Int64, pa.Check.ge(0)),
        "grid_cells": pa.Column(pl.Int64, pa.Check.ge(0)),
        "output_path": pa.Column(pl.String, nullable=True),
        "max_source_ts": pa.Column(_TS, nullable=True),
    },
    strict=True,
    coerce=True,
)


def grid_signature(tenors_days: list[float], moneyness: list[float]) -> str:
    """Hash the explicitly ordered grid axes."""
    payload = json.dumps(
        {"tenors_days": tenors_days, "moneyness": moneyness},
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _list(row: dict[str, object], column: str) -> list[object]:
    values = row[column]
    if not isinstance(values, list) or len(values) != GRID_CELL_COUNT:
        raise ValueError(f"{column} must contain exactly 54 cells")
    return values


def _validate_grid_lists(
    row: dict[str, object],
) -> tuple[np.ndarray, list[bool], list[bool]]:
    grid_w = np.asarray(_list(row, "grid_w"), dtype=np.float64)
    grid_k = np.asarray(_list(row, "grid_k"), dtype=np.float64)
    n_obs = np.asarray(_list(row, "grid_cell_n_obs"), dtype=np.int64)
    interp = cast(list[bool], _list(row, "grid_interp_flag"))
    fit_rmse = np.asarray(_list(row, "grid_fit_rmse"), dtype=np.float64)
    extrap = cast(list[bool], _list(row, "grid_extrap_flag"))
    quality_weight = np.asarray(_list(row, "grid_quality_weight"), dtype=np.float64)
    if not np.all(np.isfinite(grid_w)):
        raise ValueError("grid_w values must be finite")
    if np.any(grid_w < 0.0):
        raise ValueError("grid_w values must be nonnegative")
    if not np.all(np.isfinite(grid_k)):
        raise ValueError("grid_k values must be finite")
    if np.any(n_obs < 0):
        raise ValueError("grid_cell_n_obs values must be nonnegative")
    if not np.all(np.isfinite(fit_rmse)):
        raise ValueError("grid_fit_rmse values must be finite")
    if np.any(fit_rmse < 0.0):
        raise ValueError("grid_fit_rmse values must be nonnegative")
    if not np.all(np.isfinite(quality_weight)):
        raise ValueError("grid_quality_weight values must be finite")
    if np.any((quality_weight < 0.0) | (quality_weight > 1.0)):
        raise ValueError("grid_quality_weight values must be between 0 and 1")
    return n_obs, interp, extrap


def _implied_quality_flags(
    row: dict[str, object], n_obs: np.ndarray, interp: list[bool], extrap: list[bool]
) -> list[str]:
    reference = cast(int, row["surface_quality_n_obs_reference"])
    conditions = {
        "m4_butterfly_post": cast(int, row["surface_m4_butterfly_post"]) > 0,
        "m4_calendar_post": cast(int, row["surface_m4_calendar_post"]) > 0,
        "butterfly_certification_failed": not cast(bool, row["surface_all_butterfly_certified"]),
        "calendar_certification_failed": not cast(bool, row["surface_all_calendar_certified"]),
        "model_butterfly_violations": cast(int, row["surface_model_butterfly_violations"]) > 0,
        "model_vertical_violations": cast(int, row["surface_model_vertical_violations"]) > 0,
        "model_calendar_violations": cast(int, row["surface_model_calendar_violations"]) > 0,
        "model_insufficient_overlap": cast(int, row["surface_model_insufficient_overlap_pairs"])
        > 0,
        "low_observation_support": bool(np.any(n_obs < reference)),
        "interpolated_grid_cells": any(
            is_interp and not is_extrap for is_interp, is_extrap in zip(interp, extrap, strict=True)
        ),
        "extrapolated_grid_cells": any(extrap),
    }
    return [flag for flag in SURFACE_QUALITY_FLAG_ORDER if conditions[flag]]


def _validate_quality_coherence(
    row: dict[str, object], n_obs: np.ndarray, interp: list[bool], extrap: list[bool]
) -> None:
    quality_flags = cast(list[str], row["surface_quality_flags"])
    if quality_flags != _implied_quality_flags(row, n_obs, interp, extrap):
        raise ValueError("surface quality flags must exactly match surface diagnostics")
    expected_model_ok = not any(
        cast(int, row[column]) > 0
        for column in (
            "surface_model_butterfly_violations",
            "surface_model_vertical_violations",
            "surface_model_calendar_violations",
            "surface_model_insufficient_overlap_pairs",
        )
    )
    if cast(bool, row["surface_model_ok"]) != expected_model_ok:
        raise ValueError("surface_model_ok must match model-domain violation counts")
    expected_interp = float(
        np.mean(
            [
                is_interp and not is_extrap
                for is_interp, is_extrap in zip(interp, extrap, strict=True)
            ]
        )
    )
    if not math.isclose(
        cast(float, row["surface_interp_fraction"]), expected_interp, abs_tol=1e-12
    ):
        raise ValueError("surface_interp_fraction must match grid provenance")
    if not math.isclose(
        cast(float, row["surface_extrap_fraction"]), float(np.mean(extrap)), abs_tol=1e-12
    ):
        raise ValueError("surface_extrap_fraction must match grid provenance")


def validate_daily_features(df: pl.DataFrame) -> pl.DataFrame:
    """Validate the schema, fixed grid shape, and no-lookahead invariant."""
    validated = DAILY_FEATURES.validate(df)
    for row in validated.iter_rows(named=True):
        for column, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{column} must be finite when present")
        n_obs, interp, extrap = _validate_grid_lists(row)
        _validate_quality_coherence(row, n_obs, interp, extrap)
        snap_ts = row["snap_ts"]
        for source in SOURCE_GROUPS:
            source_ts = row[f"{source}_source_ts"]
            if source_ts is not None and source_ts > snap_ts:
                raise ValueError(f"{source}_source_ts must not exceed snap_ts")
        if row["max_source_ts"] > snap_ts:
            raise ValueError("max_source_ts must not exceed snap_ts")
    return validated
