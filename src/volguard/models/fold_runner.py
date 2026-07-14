"""Walk-forward fold contexts and forecast-before-target execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray

from volguard.config import EvalConfig
from volguard.datasets.types import Fold, Split
from volguard.models.inputs import EvalPanel
from volguard.models.types import FittedBaseline, ForecastBatch, ForecastRecord, GridSpec

FloatArray = NDArray[np.float64]
HyperParams = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FoldContext:
    """Train/validation/test materialization for one expanding fold.

    Thresholds and any fold-local statistics are derived **only** from the
    training interval. ``grid_k`` is retained for post-forecast scoring and is
    never an input to baseline coefficient fitting in this runner.
    """

    fold: Fold
    grid_spec: GridSpec
    dates: tuple[date, ...]
    grid_w: FloatArray
    grid_k: FloatArray
    reliability: FloatArray
    dvol: FloatArray
    features: FloatArray
    feature_names: tuple[str, ...]
    taus: FloatArray
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    dvol_stress_threshold: float | None

    def indices_for(self, split: Split) -> tuple[int, ...]:
        if split == "train":
            return self.train_indices
        if split == "validation":
            return self.validation_indices
        return self.test_indices


class BaselineModel(Protocol):
    """Fold-local baseline: fit on train(+val for tuning), forecast one day ahead."""

    model_id: str

    def fit(
        self,
        ctx: FoldContext,
        *,
        cfg: EvalConfig,
        tune: bool,
        frozen_hyperparameters: HyperParams | None,
    ) -> FittedBaseline: ...

    def predict_next(
        self,
        fitted: FittedBaseline,
        *,
        ctx: FoldContext,
        history_end: int,
        issue_date: date,
        target_date: date,
        cfg: EvalConfig,
    ) -> FloatArray: ...


def _finite_train_dvol(dvol: FloatArray, train_indices: Sequence[int]) -> FloatArray:
    values = np.asarray([dvol[i] for i in train_indices], dtype=np.float64)
    return values[np.isfinite(values)]


def build_fold_context(panel: EvalPanel, fold: Fold, cfg: EvalConfig) -> FoldContext:
    """Slice an eval panel into one fold and compute train-only regime thresholds."""
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for index, snap_date in enumerate(panel.dates):
        split = fold.split_for(snap_date)
        if split == "train":
            train_indices.append(index)
        elif split == "validation":
            validation_indices.append(index)
        elif split == "test":
            test_indices.append(index)
    if not train_indices:
        raise ValueError(f"fold {fold.fold_id} has an empty training interval")

    train_dvol = _finite_train_dvol(panel.dvol, train_indices)
    threshold: float | None
    if train_dvol.size == 0:
        threshold = None
    else:
        threshold = float(np.quantile(train_dvol, cfg.regimes.stress_dvol_quantile))

    taus = np.asarray([day / 365.0 for day in panel.grid_spec.tenors_days], dtype=np.float64)
    return FoldContext(
        fold=fold,
        grid_spec=panel.grid_spec,
        dates=panel.dates,
        grid_w=panel.grid_w,
        grid_k=panel.grid_k,
        reliability=panel.reliability,
        dvol=panel.dvol,
        features=panel.features,
        feature_names=panel.feature_names,
        taus=taus,
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
        test_indices=tuple(test_indices),
        dvol_stress_threshold=threshold,
    )


def apply_w_floor(raw: FloatArray, floor: float) -> tuple[FloatArray, int, float]:
    """Apply the common nonnegative total-variance floor; report pre-floor stats."""
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (6, 9):
        raise ValueError("forecast grid must have shape (6, 9)")
    if not np.all(np.isfinite(array)):
        raise ValueError("forecast grid must be finite")
    negatives = array < 0.0
    count = int(np.count_nonzero(negatives))
    pre_min = float(array.min()) if array.size else 0.0
    floored = np.maximum(array, floor)
    return floored, count, pre_min


def _forecast_splits(fold: Fold) -> tuple[Split, ...]:
    """Out-of-sample splits only — train targets stay out of the scored batch."""
    del fold
    return ("validation", "test")


def run_fold_forecasts(
    model: BaselineModel,
    ctx: FoldContext,
    *,
    cfg: EvalConfig,
    frozen_hyperparameters: HyperParams | None = None,
) -> ForecastBatch:
    """Fit on the training interval, then forecast validation/test days only.

    Coefficients are frozen after ``fit``. Later one-day forecasts may still
    consume earlier realized days (including earlier validation/test outcomes
    once elapsed) through ``history_end``, but train-interval targets are never
    emitted so evaluation metrics stay out of sample.
    """
    tune = bool(ctx.fold.tune_hyperparameters and frozen_hyperparameters is None)
    fitted = model.fit(
        ctx,
        cfg=cfg,
        tune=tune,
        frozen_hyperparameters=frozen_hyperparameters,
    )
    records: list[ForecastRecord] = []
    date_to_index = {snap_date: index for index, snap_date in enumerate(ctx.dates)}

    for split in _forecast_splits(ctx.fold):
        for index in ctx.indices_for(split):
            target_date = ctx.dates[index]
            if index == 0:
                continue
            issue_date = ctx.dates[index - 1]
            predicted = model.predict_next(
                fitted,
                ctx=ctx,
                history_end=index,
                issue_date=issue_date,
                target_date=target_date,
                cfg=cfg,
            )
            floored, neg_count, pre_min = apply_w_floor(predicted, cfg.w_floor)
            records.append(
                ForecastRecord(
                    model_id=model.model_id,
                    fold_id=ctx.fold.fold_id,
                    split=split,
                    issue_date=issue_date,
                    target_date=target_date,
                    raw_w=floored,
                    pre_floor_negative_count=neg_count,
                    pre_floor_min=pre_min,
                )
            )
            assert date_to_index[target_date] == index

    return ForecastBatch(
        model_id=model.model_id,
        fold_id=ctx.fold.fold_id,
        records=tuple(records),
        fitted=fitted,
    )


def folds_from_manifest(panel: EvalPanel) -> tuple[Fold, ...]:
    """Rebuild immutable ``Fold`` objects from the persisted split manifest."""
    frame = panel.split_manifest
    folds: list[Fold] = []
    for fold_id in sorted(frame["fold_id"].unique().to_list()):
        rows = frame.filter(pl.col("fold_id") == fold_id)
        first = rows.row(0, named=True)
        folds.append(
            Fold(
                fold_id=int(first["fold_id"]),
                train_start=first["train_start"],
                train_end=first["train_end"],
                validation_start=first["validation_start"],
                validation_end=first["validation_end"],
                test_start=first["test_start"],
                test_end=first["test_end"],
                tune_hyperparameters=bool(first["tune_hyperparameters"]),
            )
        )
    return tuple(folds)


ForecastRunner = Callable[[BaselineModel, FoldContext], ForecastBatch]
