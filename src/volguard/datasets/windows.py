"""Consecutive-calendar supervised windows with masks and fold membership."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import cast

import numpy as np
import polars as pl

from volguard.datasets.leakage import assert_no_feature_leakage
from volguard.datasets.pca import transform_surface_pca
from volguard.datasets.types import (
    GRID_QUALITY_CHANNEL_NAMES,
    Fold,
    Split,
    SurfacePCA,
    WindowedDataset,
)

_GRID_ROWS = 6
_GRID_COLUMNS = 9
_EXCLUDED_FEATURES = {
    "snap_date",
    "snap_ts",
    "max_source_ts",
    "grid_signature",
    "grid_w",
    "grid_k",
    "grid_cell_n_obs",
    "grid_interp_flag",
    "grid_extrap_flag",
    "grid_fit_rmse",
    "grid_quality_weight",
    "surface_quality_flags",
}


def _scalar_feature_names(daily: pl.DataFrame) -> tuple[str, ...]:
    names: list[str] = []
    for name, dtype in daily.schema.items():
        if name in _EXCLUDED_FEATURES or name.endswith("_source_ts"):
            continue
        if dtype.is_numeric() or dtype == pl.Boolean:
            names.append(name)
    return tuple(names)


def _empty_dataset(lookback: int, features: tuple[str, ...]) -> WindowedDataset:
    return WindowedDataset(
        x_grid=np.empty((0, lookback, _GRID_ROWS, _GRID_COLUMNS)),
        x_grid_quality=np.empty(
            (0, lookback, _GRID_ROWS, _GRID_COLUMNS, len(GRID_QUALITY_CHANNEL_NAMES))
        ),
        x_features=np.empty((0, lookback, len(features))),
        x_feature_mask=np.empty((0, lookback, len(features)), dtype=np.bool_),
        y_grid=np.empty((0, _GRID_ROWS, _GRID_COLUMNS)),
        y_grid_weight=np.empty((0, _GRID_ROWS, _GRID_COLUMNS)),
        input_dates=(),
        target_dates=(),
        max_source_ts=(),
        splits=(),
        feature_names=features,
        quality_channel_names=GRID_QUALITY_CHANNEL_NAMES,
    )


def _validated_rows(
    daily: pl.DataFrame, fold: Fold, pca: SurfacePCA
) -> tuple[dict[date, dict[str, object]], tuple[str, ...]]:
    required = {
        "snap_date",
        "snap_ts",
        "max_source_ts",
        "grid_w",
        "grid_cell_n_obs",
        "grid_interp_flag",
        "grid_extrap_flag",
        "grid_fit_rmse",
        "grid_quality_weight",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily features missing window columns: {sorted(missing)}")
    if daily["snap_date"].n_unique() != daily.height:
        raise ValueError("daily features contain duplicate snap_date rows")
    if pca.fit_start is None or pca.fit_end is None:
        raise ValueError("PCA fit-date provenance is required for supervised windows")
    if not fold.train_start <= pca.fit_start < pca.fit_end <= fold.train_end:
        raise ValueError("PCA fit interval must be contained in the fold training range")
    assert_no_feature_leakage(daily)
    ordered = daily.sort("snap_date")
    rows = {cast(date, row["snap_date"]): row for row in ordered.iter_rows(named=True)}
    return rows, _scalar_feature_names(ordered)


def _window_quality(window_rows: list[dict[str, object]], lookback_days: int) -> np.ndarray:
    raw_channels = (
        ("grid_cell_n_obs", np.float64),
        ("grid_interp_flag", np.float64),
        ("grid_extrap_flag", np.float64),
        ("grid_fit_rmse", np.float64),
        ("grid_quality_weight", np.float64),
    )
    channels = [
        np.asarray([row[name] for row in window_rows], dtype=dtype) for name, dtype in raw_channels
    ]
    if any(channel.shape != (lookback_days, 54) for channel in channels):
        raise ValueError("every grid quality list must contain exactly 54 cells")
    channels[0] = np.log1p(channels[0])
    quality = np.stack(channels, axis=-1)
    if not np.all(np.isfinite(quality)):
        raise ValueError("window grid quality values must be finite")
    return quality.reshape(
        lookback_days,
        _GRID_ROWS,
        _GRID_COLUMNS,
        len(GRID_QUALITY_CHANNEL_NAMES),
    )


def make_supervised_windows(
    daily: pl.DataFrame,
    fold: Fold,
    pca: SurfacePCA,
    *,
    lookback_days: int = 20,
    horizon_days: int = 1,
) -> WindowedDataset:
    """Build windows; missing calendar dates drop samples and targets assign splits."""
    if lookback_days < 1 or horizon_days < 1:
        raise ValueError("lookback_days and horizon_days must be positive")
    rows, scalar_names = _validated_rows(daily, fold, pca)
    pca_names = tuple(f"surface_pca_{index + 1}" for index in range(pca.components.shape[0]))
    feature_names = scalar_names + pca_names
    grids: list[np.ndarray] = []
    grid_quality: list[np.ndarray] = []
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_weights: list[np.ndarray] = []
    input_dates: list[tuple[date, ...]] = []
    target_dates: list[date] = []
    max_source_timestamps: list[datetime] = []
    splits: list[Split] = []
    for target_date in rows:
        split = fold.split_for(target_date)
        if split is None:
            continue
        input_end = target_date - timedelta(days=horizon_days)
        first = input_end - timedelta(days=lookback_days - 1)
        calendar_span = tuple(
            first + timedelta(days=offset) for offset in range((target_date - first).days + 1)
        )
        if any(day not in rows for day in calendar_span):
            continue
        window_dates = calendar_span[:lookback_days]
        window_rows = [rows[day] for day in window_dates]
        window_grids = np.asarray([row["grid_w"] for row in window_rows], dtype=np.float64)
        target_grid = np.asarray(rows[target_date]["grid_w"], dtype=np.float64)
        target_weight = np.asarray(rows[target_date]["grid_quality_weight"], dtype=np.float64)
        if window_grids.shape != (lookback_days, 54) or target_grid.shape != (54,):
            raise ValueError("every grid_w value must contain exactly 54 cells")
        if target_weight.shape != (54,):
            raise ValueError("every grid_quality_weight value must contain exactly 54 cells")
        if not np.all(np.isfinite(window_grids)) or not np.all(np.isfinite(target_grid)):
            raise ValueError("window grids must be finite")
        scalar = np.asarray(
            [
                [
                    np.nan if row[name] is None else float(cast(float | int | bool, row[name]))
                    for name in scalar_names
                ]
                for row in window_rows
            ],
            dtype=np.float64,
        )
        scores = transform_surface_pca(pca, window_grids)
        combined = np.concatenate((scalar, scores), axis=1)
        grids.append(window_grids.reshape(lookback_days, _GRID_ROWS, _GRID_COLUMNS))
        grid_quality.append(_window_quality(window_rows, lookback_days))
        features.append(combined)
        masks.append(np.isfinite(combined))
        targets.append(target_grid.reshape(_GRID_ROWS, _GRID_COLUMNS))
        target_weights.append(target_weight.reshape(_GRID_ROWS, _GRID_COLUMNS))
        input_dates.append(tuple(window_dates))
        target_dates.append(target_date)
        max_source_timestamps.append(
            max(cast(datetime, row["max_source_ts"]) for row in window_rows)
        )
        splits.append(split)
    if not grids:
        return _empty_dataset(lookback_days, feature_names)
    return WindowedDataset(
        x_grid=np.stack(grids),
        x_grid_quality=np.stack(grid_quality),
        x_features=np.stack(features),
        x_feature_mask=np.stack(masks),
        y_grid=np.stack(targets),
        y_grid_weight=np.stack(target_weights),
        input_dates=tuple(input_dates),
        target_dates=tuple(target_dates),
        max_source_ts=tuple(max_source_timestamps),
        splits=tuple(splits),
        feature_names=feature_names,
        quality_channel_names=GRID_QUALITY_CHANNEL_NAMES,
    )
