"""Immutable model / evaluation contracts for the M6 harness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from volguard.datasets.types import Split

FloatArray = NDArray[np.float64]
ForecastVariant = Literal["raw", "repaired"]
MetricScope = Literal["overall", "fold", "tenor", "moneyness", "cell", "regime"]
_GRID_SHAPE = (6, 9)
_SURFACE_NDIM = 2


def _readonly(value: FloatArray) -> FloatArray:
    copied = np.array(value, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Canonical 6x9 surface axes plus the persisted M5 grid signature."""

    tenors_days: tuple[float, ...]
    moneyness: tuple[float, ...]
    signature: str
    shape: tuple[int, int] = _GRID_SHAPE

    def __post_init__(self) -> None:
        if self.shape != _GRID_SHAPE:
            raise ValueError("grid shape must be exactly (6, 9)")
        if len(self.tenors_days) != self.shape[0]:
            raise ValueError("tenors_days length must match grid rows")
        if len(self.moneyness) != self.shape[1]:
            raise ValueError("moneyness length must match grid columns")
        if not self.signature:
            raise ValueError("signature must be non-empty")

    @classmethod
    def from_axes(
        cls,
        *,
        tenors_days: tuple[float, ...] | list[float],
        moneyness: tuple[float, ...] | list[float],
        signature: str,
    ) -> GridSpec:
        """Build a grid spec from ordered tenor and moneyness axes."""
        return cls(
            tenors_days=tuple(float(v) for v in tenors_days),
            moneyness=tuple(float(v) for v in moneyness),
            signature=signature,
        )


@dataclass(frozen=True, slots=True)
class FittedBaseline:
    """Fold-local fitted baseline coefficients / hyperparameters."""

    model_id: str
    fold_id: int
    train_start: date
    train_end: date
    hyperparameters: Mapping[str, Any]
    state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.fold_id < 0:
            raise ValueError("fold_id must be nonnegative")
        if not self.train_start < self.train_end:
            raise ValueError("training interval must be non-empty")
        object.__setattr__(self, "hyperparameters", dict(self.hyperparameters))
        object.__setattr__(self, "state", dict(self.state))


@dataclass(frozen=True, slots=True)
class ForecastRecord:
    """One issue-date → target-date surface forecast (pre-repair raw grid)."""

    model_id: str
    fold_id: int
    split: Split
    issue_date: date
    target_date: date
    raw_w: FloatArray
    pre_floor_negative_count: int
    pre_floor_min: float

    def __post_init__(self) -> None:
        if self.issue_date >= self.target_date:
            raise ValueError("issue_date must precede target_date")
        if self.fold_id < 0:
            raise ValueError("fold_id must be nonnegative")
        if self.pre_floor_negative_count < 0:
            raise ValueError("pre_floor_negative_count must be nonnegative")
        raw = np.asarray(self.raw_w, dtype=np.float64)
        if raw.shape != _GRID_SHAPE or raw.ndim != _SURFACE_NDIM:
            raise ValueError("raw_w must have shape (6, 9)")
        if not np.all(np.isfinite(raw)):
            raise ValueError("raw_w must be finite")
        if np.any(raw < 0.0):
            raise ValueError("raw_w must be nonnegative after the total-variance floor")
        object.__setattr__(self, "raw_w", _readonly(raw))


@dataclass(frozen=True, slots=True)
class ForecastBatch:
    """Ordered forecasts from one fitted baseline on one fold."""

    model_id: str
    fold_id: int
    records: tuple[ForecastRecord, ...]
    fitted: FittedBaseline

    def __post_init__(self) -> None:
        if self.fold_id != self.fitted.fold_id or self.model_id != self.fitted.model_id:
            raise ValueError("ForecastBatch identity must match FittedBaseline")
        if any(record.fold_id != self.fold_id for record in self.records):
            raise ValueError("forecast records must share the batch fold_id")
        if any(record.model_id != self.model_id for record in self.records):
            raise ValueError("forecast records must share the batch model_id")


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One aggregated metric row (schema for the evaluation harness)."""

    model_id: str
    fold_id: int | None
    split: Split | Literal["all"]
    variant: ForecastVariant
    scope: MetricScope
    metric: str
    value: float
    n: int
    weight_scheme: str
    scope_key: str | None = None

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must be nonnegative")
        if not np.isfinite(self.value):
            raise ValueError("metric value must be finite")
        if not self.metric:
            raise ValueError("metric name must be non-empty")


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Provenance for one training/evaluation run."""

    run_id: str
    created_at: datetime
    model_ids: tuple[str, ...]
    config_hash: str
    data_fingerprint: str
    git_commit: str | None
    seed: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.model_ids:
            raise ValueError("model_ids must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.config_hash or not self.data_fingerprint:
            raise ValueError("config_hash and data_fingerprint are required")
