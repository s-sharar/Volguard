"""Strict M4 surface admission and deterministic surface-state features."""

from __future__ import annotations

import math
from dataclasses import fields
from datetime import UTC, date, datetime
from typing import cast

import numpy as np
import pandera.errors
import polars as pl

from volguard.config import FeatureConfig, SurfaceConfig
from volguard.features.schemas import grid_signature
from volguard.features.surface_quality import (
    SURFACE_QUALITY_FLAG_ORDER,
    SurfaceDomainQuality,
    check_model_grid,
)
from volguard.surface.schemas import SURFACES_DAILY


class SurfaceQualityError(ValueError):
    """A reason-coded surface rejection for the M5 audit table."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return np.asarray([0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))) for x in value])


def _interpolate_delta(delta: np.ndarray, iv: np.ndarray, target: float) -> float | None:
    order = np.argsort(delta)
    x, y = delta[order], iv[order]
    if target < x[0] or target > x[-1]:
        return None
    return float(np.interp(target, x, y))


_REQUIRED_COLUMNS = frozenset(
    {
        "snap_date",
        "record_kind",
        "tau",
        "moneyness",
        "grid_k",
        "grid_w",
        "cell_n_obs",
        "interp_flag",
        "rmse",
        "n_obs",
        "svi_a",
        "svi_b",
        "svi_rho",
        "svi_m",
        "svi_sigma",
        "butterfly_ok",
        "calendar_ok",
        "arb_butterfly_post",
        "arb_calendar_post",
    }
)
_SVI_COLUMNS = ("svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma")
_POST_COUNT_COLUMNS = ("arb_butterfly_post", "arb_calendar_post")
_MIN_PARAMETER_SLICES = 2


def _require_columns(rows: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(rows.columns))
    if missing:
        raise SurfaceQualityError(
            "surface_input_invalid", f"surface is missing required columns: {missing}"
        )


def _validate_partition(rows: pl.DataFrame, snap_date: date) -> None:
    partition_dates = rows["snap_date"].unique().to_list()
    if partition_dates != [snap_date]:
        raise SurfaceQualityError(
            "surface_partition_date",
            f"surface partition date {snap_date} does not match input dates {partition_dates}",
        )


def _validate_post_counts(rows: pl.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in _POST_COUNT_COLUMNS:
        values = rows[column].to_list()
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values
        ):
            raise SurfaceQualityError(
                "surface_post_counts", f"{column} must contain nonnegative integer counts"
            )
        normalized = [int(cast(int, value)) for value in values]
        if any(value < 0 for value in normalized):
            raise SurfaceQualityError(
                "surface_post_counts", f"{column} must contain nonnegative integer counts"
            )
        if len(set(normalized)) != 1:
            raise SurfaceQualityError(
                "surface_post_counts", f"{column} must be consistent on every row"
            )
        result[column] = normalized[0]
    return result


def _validate_grid(rows: pl.DataFrame, expected_cells: int) -> pl.DataFrame:
    grid = rows.filter(pl.col("record_kind") == "grid")
    if grid.height != expected_cells:
        raise SurfaceQualityError(
            "surface_grid_cell_count",
            f"expected {expected_cells} grid cells, found {grid.height}",
        )
    if grid.select(["tau", "moneyness"]).unique().height != expected_cells:
        raise SurfaceQualityError("surface_grid_duplicates", "surface grid cells are not unique")
    for column in ("tau", "moneyness", "grid_w", "grid_k", "cell_n_obs", "interp_flag"):
        if grid[column].null_count():
            raise SurfaceQualityError("surface_grid_null", f"{column} contains nulls")
    w = grid["grid_w"].to_numpy()
    k = grid["grid_k"].to_numpy()
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or not np.all(np.isfinite(k)):
        raise SurfaceQualityError("surface_grid_values", "surface grid contains invalid values")
    if np.any(grid["cell_n_obs"].to_numpy() < 0):
        raise SurfaceQualityError(
            "surface_grid_values", "surface grid has negative observation counts"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in grid["cell_n_obs"].to_list()
    ):
        raise SurfaceQualityError(
            "surface_grid_values", "surface grid observation counts must be integers"
        )
    if any(not isinstance(value, bool) for value in grid["interp_flag"].to_list()):
        raise SurfaceQualityError(
            "surface_grid_values", "surface grid interpolation provenance must be boolean"
        )
    return grid


def _validate_params(rows: pl.DataFrame) -> pl.DataFrame:
    params = rows.filter(pl.col("record_kind") == "param").sort("tau")
    if params.height < _MIN_PARAMETER_SLICES:
        raise SurfaceQualityError(
            "surface_parameter_rows", "surface needs at least two parameter slices"
        )
    for column in ("tau", "rmse", "n_obs", *_SVI_COLUMNS):
        if params[column].null_count():
            raise SurfaceQualityError("surface_parameter_rows", f"parameter {column} is null")
    for column in ("butterfly_ok", "calendar_ok"):
        if params[column].null_count():
            raise SurfaceQualityError(
                "surface_certification", f"{column} certification must be nonnull"
            )
        if any(not isinstance(value, bool) for value in params[column].to_list()):
            raise SurfaceQualityError(
                "surface_certification", f"{column} certification must be boolean"
            )
    taus = params["tau"].to_numpy()
    rmse = params["rmse"].to_numpy()
    n_obs = params["n_obs"].to_numpy()
    if not np.all(np.isfinite(taus)) or np.any(taus <= 0.0) or np.any(np.diff(taus) <= 0.0):
        raise SurfaceQualityError(
            "surface_parameter_rows", "parameter tau values must be unique increasing positive"
        )
    if not np.all(np.isfinite(rmse)) or np.any(rmse < 0.0):
        raise SurfaceQualityError(
            "surface_parameter_rows", "parameter rmse values must be finite and nonnegative"
        )
    if np.any(n_obs <= 0) or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in params["n_obs"].to_list()
    ):
        raise SurfaceQualityError("surface_parameter_rows", "parameter n_obs must be positive")
    for column in _SVI_COLUMNS:
        if not np.all(np.isfinite(params[column].to_numpy())):
            raise SurfaceQualityError(
                "surface_parameter_rows", f"parameter {column} must be finite"
            )
    return params


def _ordered_grid(grid: pl.DataFrame, surface_cfg: SurfaceConfig) -> list[dict[str, object]]:
    ordered: list[dict[str, object]] = []
    for tenor_days in surface_cfg.tenor_grid_days:
        tau = tenor_days / 365.0
        for moneyness in surface_cfg.moneyness_grid:
            cell = grid.filter(
                (pl.col("tau") - tau).abs().lt(1e-9)
                & (pl.col("moneyness") - moneyness).abs().lt(1e-9)
            )
            if cell.height != 1:
                raise SurfaceQualityError(
                    "surface_grid_axes",
                    f"missing cell tenor={tenor_days}, moneyness={moneyness}",
                )
            ordered.append(cell.row(0, named=True))
    width = len(surface_cfg.moneyness_grid)
    k = np.asarray([cell["grid_k"] for cell in ordered], dtype=np.float64).reshape(-1, width)
    if not np.all(np.diff(k, axis=1) > 0.0):
        raise SurfaceQualityError(
            "surface_grid_axes", "native grid_k rows must be strictly increasing"
        )
    return ordered


def _grid_quality_fields(
    ordered: list[dict[str, object]],
    params: pl.DataFrame,
    feature_cfg: FeatureConfig,
    surface_cfg: SurfaceConfig,
) -> dict[str, list[float] | list[bool]]:
    param_taus = params["tau"].to_numpy()
    param_rmse = params["rmse"].to_numpy()
    fit_rmse: list[float] = []
    extrapolated: list[bool] = []
    for tenor_days in surface_cfg.tenor_grid_days:
        tau = tenor_days / 365.0
        nearest = int(np.argmin(np.abs(param_taus - tau)))
        fit_rmse.extend([float(param_rmse[nearest])] * len(surface_cfg.moneyness_grid))
        outside = bool(tau < param_taus[0] or tau > param_taus[-1])
        extrapolated.extend([outside] * len(surface_cfg.moneyness_grid))
    weights: list[float] = []
    for cell, rmse, extrap in zip(ordered, fit_rmse, extrapolated, strict=True):
        support = min(
            float(cast(int, cell["cell_n_obs"])) / feature_cfg.quality_n_obs_reference,
            1.0,
        )
        if extrap:
            provenance = feature_cfg.quality_extrap_weight
        elif bool(cell["interp_flag"]):
            provenance = feature_cfg.quality_interp_weight
        else:
            provenance = 1.0
        weights.append(
            float(
                np.clip(
                    support * provenance * 2.0 ** (-rmse / feature_cfg.quality_rmse_half_life),
                    0.0,
                    1.0,
                )
            )
        )
    return {
        "grid_fit_rmse": fit_rmse,
        "grid_extrap_flag": extrapolated,
        "grid_quality_weight": weights,
    }


def _quality_flags(
    *,
    post_counts: dict[str, int],
    butterfly_certified: bool,
    calendar_certified: bool,
    model: SurfaceDomainQuality,
    low_support: bool,
    interpolated: list[bool],
    extrapolated: list[bool],
) -> list[str]:
    conditions = {
        "m4_butterfly_post": post_counts["arb_butterfly_post"] > 0,
        "m4_calendar_post": post_counts["arb_calendar_post"] > 0,
        "butterfly_certification_failed": not butterfly_certified,
        "calendar_certification_failed": not calendar_certified,
        "model_butterfly_violations": model.butterfly_violations > 0,
        "model_vertical_violations": model.vertical_violations > 0,
        "model_calendar_violations": model.calendar_violations > 0,
        "model_insufficient_overlap": model.insufficient_overlap_pairs > 0,
        "low_observation_support": low_support,
        "interpolated_grid_cells": any(interpolated),
        "extrapolated_grid_cells": any(extrapolated),
    }
    return [flag for flag in SURFACE_QUALITY_FLAG_ORDER if conditions[flag]]


def _surface_quality_fields(
    ordered: list[dict[str, object]],
    params: pl.DataFrame,
    post_counts: dict[str, int],
    feature_cfg: FeatureConfig,
    surface_cfg: SurfaceConfig,
) -> dict[str, object]:
    grid_quality = _grid_quality_fields(ordered, params, feature_cfg, surface_cfg)
    width = len(surface_cfg.moneyness_grid)
    k = np.asarray([cell["grid_k"] for cell in ordered], dtype=np.float64).reshape(-1, width)
    w = np.asarray([cell["grid_w"] for cell in ordered], dtype=np.float64).reshape(-1, width)
    taus = np.asarray(surface_cfg.tenor_grid_days, dtype=np.float64) / 365.0
    model = check_model_grid(k, w, taus, calendar_points=feature_cfg.model_domain_calendar_points)
    butterfly_certified = bool(params["butterfly_ok"].all())
    calendar_certified = bool(params["calendar_ok"].all())
    extrapolated = cast(list[bool], grid_quality["grid_extrap_flag"])
    interpolated = [
        bool(cell["interp_flag"]) and not extrap
        for cell, extrap in zip(ordered, extrapolated, strict=True)
    ]
    low_support = any(
        cast(int, cell["cell_n_obs"]) < feature_cfg.quality_n_obs_reference for cell in ordered
    )
    result: dict[str, object] = dict(grid_quality)
    for field in fields(SurfaceDomainQuality):
        result[f"surface_model_{field.name}"] = getattr(model, field.name)
    result.update(
        {
            "surface_mean_fit_rmse": cast(float, params["rmse"].mean()),
            "surface_max_fit_rmse": cast(float, params["rmse"].max()),
            "surface_min_slice_n_obs": cast(int, params["n_obs"].min()),
            "surface_total_slice_n_obs": cast(int, params["n_obs"].sum()),
            "surface_interp_fraction": float(np.mean(interpolated)),
            "surface_extrap_fraction": float(np.mean(extrapolated)),
            "surface_all_butterfly_certified": butterfly_certified,
            "surface_all_calendar_certified": calendar_certified,
            "surface_m4_butterfly_post": post_counts["arb_butterfly_post"],
            "surface_m4_calendar_post": post_counts["arb_calendar_post"],
            "surface_model_ok": model.ok,
            "surface_quality_n_obs_reference": feature_cfg.quality_n_obs_reference,
            "surface_quality_flags": _quality_flags(
                post_counts=post_counts,
                butterfly_certified=butterfly_certified,
                calendar_certified=calendar_certified,
                model=model,
                low_support=low_support,
                interpolated=interpolated,
                extrapolated=extrapolated,
            ),
        }
    )
    return result


def _build_surface_features(
    rows: pl.DataFrame,
    snap_date: date,
    feature_cfg: FeatureConfig,
    surface_cfg: SurfaceConfig,
) -> dict[str, object]:
    """Validate and flatten one complete M4 surface into M5 features."""
    tenors = surface_cfg.tenor_grid_days
    money = surface_cfg.moneyness_grid
    _require_columns(rows)
    _validate_partition(rows, snap_date)
    post_counts = _validate_post_counts(rows)
    grid = _validate_grid(rows, len(tenors) * len(money))
    params = _validate_params(rows)
    ordered = _ordered_grid(grid, surface_cfg)

    result: dict[str, object] = {
        "grid_signature": grid_signature(tenors, money),
        "grid_w": [float(cast(float, cell["grid_w"])) for cell in ordered],
        "grid_k": [float(cast(float, cell["grid_k"])) for cell in ordered],
        "grid_cell_n_obs": [int(cast(int, cell["cell_n_obs"])) for cell in ordered],
        "grid_interp_flag": [bool(cast(bool, cell["interp_flag"])) for cell in ordered],
    }
    result.update(_surface_quality_fields(ordered, params, post_counts, feature_cfg, surface_cfg))
    for tenor_index, tenor_days in enumerate(tenors):
        start = tenor_index * len(money)
        cells = ordered[start : start + len(money)]
        tau = tenor_days / 365.0
        w = np.asarray([cell["grid_w"] for cell in cells], dtype=np.float64)
        k = np.asarray([cell["grid_k"] for cell in cells], dtype=np.float64)
        iv = np.sqrt(w / tau)
        atm_index = money.index(0)
        left_index, right_index = money.index(-1), money.index(1)
        result[f"atm_iv_{int(tenor_days)}d"] = float(iv[atm_index])
        result[f"curvature_{int(tenor_days)}d"] = float(
            0.5 * (iv[left_index] + iv[right_index]) - iv[atm_index]
        )
        root_tau = math.sqrt(tau)
        d1 = -k / np.maximum(iv * root_tau, np.finfo(float).eps) + 0.5 * iv * root_tau
        call_iv = _interpolate_delta(_normal_cdf(d1), iv, 0.25)
        put_iv = _interpolate_delta(_normal_cdf(d1) - 1.0, iv, -0.25)
        result[f"skew_25delta_{int(tenor_days)}d"] = (
            None if call_iv is None or put_iv is None else put_iv - call_iv
        )
    first, last = int(tenors[0]), int(tenors[-1])
    result["atm_term_slope_7_180"] = (
        cast(float, result[f"atm_iv_{last}d"]) - cast(float, result[f"atm_iv_{first}d"])
    ) / ((last - first) / 365.0)
    snap_ts = datetime(
        snap_date.year,
        snap_date.month,
        snap_date.day,
        surface_cfg.snap_hour_utc,
        surface_cfg.snap_minute_utc,
        tzinfo=UTC,
    )
    result.update(
        {
            "snap_date": snap_date,
            "snap_ts": snap_ts,
            "surface_source_ts": snap_ts,
            "surface_age_s": 0.0,
            "surface_available": True,
        }
    )
    return result


def build_surface_features(
    rows: pl.DataFrame,
    snap_date: date,
    surface_cfg: SurfaceConfig,
    feature_cfg: FeatureConfig | None = None,
) -> dict[str, object]:
    """Validate and flatten one complete M4 surface into M5 features."""
    try:
        validated = SURFACES_DAILY.validate(rows)
        return _build_surface_features(
            validated,
            snap_date,
            feature_cfg or FeatureConfig(),
            surface_cfg,
        )
    except SurfaceQualityError:
        raise
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        pandera.errors.SchemaError,
        pl.exceptions.PolarsError,
    ) as exc:
        raise SurfaceQualityError(
            "surface_input_invalid", f"surface input contract is invalid: {exc}"
        ) from exc
