"""Immutable walk-forward, PCA, and supervised-dataset values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Split = Literal["train", "validation", "test"]
_MATRIX_NDIM = 2
_TENSOR_NDIM = 3
_WINDOW_GRID_NDIM = 4
_WINDOW_GRID_QUALITY_NDIM = 5
_GRID_SHAPE = (6, 9)
GRID_QUALITY_CHANNEL_NAMES = (
    "log1p_cell_n_obs",
    "is_interpolated",
    "is_extrapolated",
    "fit_rmse",
    "reliability_weight",
)


def _readonly[T: np.generic](value: NDArray[T], dtype: np.dtype[T]) -> NDArray[T]:
    copied = np.array(value, dtype=dtype, copy=True)
    copied.flags.writeable = False
    return copied


def _validate_quality_values(x_grid_quality: FloatArray, y_grid_weight: FloatArray) -> None:
    if not np.all(np.isfinite(x_grid_quality)):
        raise ValueError("x_grid_quality must be finite")
    if not np.all(np.isfinite(y_grid_weight)):
        raise ValueError("y_grid_weight must be finite")
    if np.any(x_grid_quality[..., 0] < 0.0):
        raise ValueError("log1p_cell_n_obs must be nonnegative")
    for index, name in ((1, "is_interpolated"), (2, "is_extrapolated")):
        values = x_grid_quality[..., index]
        if np.any((values != 0.0) & (values != 1.0)):
            raise ValueError(f"{name} must contain only 0 or 1")
    if np.any(x_grid_quality[..., 3] < 0.0):
        raise ValueError("fit_rmse must be nonnegative")
    reliability = x_grid_quality[..., 4]
    if np.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError("reliability_weight must be between 0 and 1")
    if np.any((y_grid_weight < 0.0) | (y_grid_weight > 1.0)):
        raise ValueError("y_grid_weight must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Fold:
    """One expanding fold with contiguous half-open split ranges."""

    fold_id: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    tune_hyperparameters: bool

    def __post_init__(self) -> None:
        if self.fold_id < 0:
            raise ValueError("fold_id must be nonnegative")
        if not self.train_start < self.train_end:
            raise ValueError("train range must be non-empty")
        if self.train_end != self.validation_start or self.validation_end != self.test_start:
            raise ValueError("train, validation, and test ranges must be contiguous")
        if not self.validation_start < self.validation_end < self.test_end:
            raise ValueError("validation and test ranges must be non-empty")

    def split_for(self, target_date: date) -> Split | None:
        if self.train_start <= target_date < self.train_end:
            return "train"
        if self.validation_start <= target_date < self.validation_end:
            return "validation"
        if self.test_start <= target_date < self.test_end:
            return "test"
        return None


@dataclass(frozen=True, slots=True)
class SurfacePCA:
    """Fold-local PCA state fitted on a recorded training interval."""

    mean: FloatArray
    components: FloatArray
    explained_variance_ratio: FloatArray
    fit_start: date | None
    fit_end: date | None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        components = np.asarray(self.components, dtype=np.float64)
        ratio = np.asarray(self.explained_variance_ratio, dtype=np.float64)
        if mean.ndim != 1 or components.ndim != _MATRIX_NDIM or ratio.ndim != 1:
            raise ValueError("PCA state has invalid dimensions")
        if components.shape != (ratio.size, mean.size):
            raise ValueError("PCA component shape does not match mean/variance ratio")
        if (self.fit_start is None) != (self.fit_end is None):
            raise ValueError("PCA fit provenance must be complete or absent")
        if (
            self.fit_start is not None
            and self.fit_end is not None
            and self.fit_start >= self.fit_end
        ):
            raise ValueError("PCA fit interval is reversed")
        object.__setattr__(self, "mean", _readonly(mean, np.dtype(np.float64)))
        object.__setattr__(self, "components", _readonly(components, np.dtype(np.float64)))
        object.__setattr__(
            self,
            "explained_variance_ratio",
            _readonly(ratio, np.dtype(np.float64)),
        )


@dataclass(frozen=True, slots=True)
class WindowedDataset:
    """Aligned grid/feature windows and next-day grid targets."""

    x_grid: FloatArray
    x_grid_quality: FloatArray
    x_features: FloatArray
    x_feature_mask: BoolArray
    y_grid: FloatArray
    y_grid_weight: FloatArray
    input_dates: tuple[tuple[date, ...], ...]
    target_dates: tuple[date, ...]
    max_source_ts: tuple[datetime, ...]
    splits: tuple[Split, ...]
    feature_names: tuple[str, ...]
    quality_channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        x_grid = np.asarray(self.x_grid, dtype=np.float64)
        x_grid_quality = np.asarray(self.x_grid_quality, dtype=np.float64)
        x_features = np.asarray(self.x_features, dtype=np.float64)
        mask = np.asarray(self.x_feature_mask, dtype=np.bool_)
        y_grid = np.asarray(self.y_grid, dtype=np.float64)
        y_grid_weight = np.asarray(self.y_grid_weight, dtype=np.float64)
        if (
            x_grid.ndim != _WINDOW_GRID_NDIM
            or x_grid_quality.ndim != _WINDOW_GRID_QUALITY_NDIM
            or x_features.ndim != _TENSOR_NDIM
            or y_grid.ndim != _TENSOR_NDIM
            or y_grid_weight.ndim != _TENSOR_NDIM
        ):
            raise ValueError("window arrays have invalid dimensions")
        if (
            x_grid.shape[2:] != _GRID_SHAPE
            or x_grid_quality.shape[2:4] != _GRID_SHAPE
            or y_grid.shape[1:] != _GRID_SHAPE
            or y_grid_weight.shape[1:] != _GRID_SHAPE
        ):
            raise ValueError("grid arrays must use the exact 6x9 spatial shape")
        if mask.shape != x_features.shape:
            raise ValueError("x_feature_mask must align with x_features")
        n = x_grid.shape[0]
        if (
            x_grid_quality.shape[0] != n
            or x_features.shape[0] != n
            or y_grid.shape[0] != n
            or y_grid_weight.shape[0] != n
        ):
            raise ValueError("window arrays must have the same sample count")
        if x_grid_quality.shape[1] != x_grid.shape[1] or x_features.shape[1] != x_grid.shape[1]:
            raise ValueError("window inputs must align with the lookback length")
        if (
            x_grid_quality.shape[2:4] != x_grid.shape[2:4]
            or y_grid.shape[1:] != x_grid.shape[2:]
            or y_grid_weight.shape[1:] != y_grid.shape[1:]
        ):
            raise ValueError("quality and target arrays must align with the grid")
        if (
            len(self.input_dates) != n
            or len(self.target_dates) != n
            or len(self.max_source_ts) != n
            or len(self.splits) != n
        ):
            raise ValueError("window metadata must align with arrays")
        if any(len(dates) != x_grid.shape[1] for dates in self.input_dates):
            raise ValueError("input_dates must align with lookback length")
        if len(self.feature_names) != x_features.shape[2]:
            raise ValueError("feature_names must align with x_features")
        if len(self.quality_channel_names) != x_grid_quality.shape[-1]:
            raise ValueError("quality_channel_names must align with x_grid_quality")
        if self.quality_channel_names != GRID_QUALITY_CHANNEL_NAMES:
            raise ValueError("quality_channel_names must use the fixed channel order")
        _validate_quality_values(x_grid_quality, y_grid_weight)
        object.__setattr__(self, "x_grid", _readonly(x_grid, np.dtype(np.float64)))
        object.__setattr__(
            self,
            "x_grid_quality",
            _readonly(x_grid_quality, np.dtype(np.float64)),
        )
        object.__setattr__(self, "x_features", _readonly(x_features, np.dtype(np.float64)))
        object.__setattr__(self, "x_feature_mask", _readonly(mask, np.dtype(np.bool_)))
        object.__setattr__(self, "y_grid", _readonly(y_grid, np.dtype(np.float64)))
        object.__setattr__(
            self,
            "y_grid_weight",
            _readonly(y_grid_weight, np.dtype(np.float64)),
        )
