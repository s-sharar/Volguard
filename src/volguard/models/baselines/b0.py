"""B0 persistence baseline: ``ŵ(t+1) = w(t)``."""

from __future__ import annotations

from datetime import date

import numpy as np
from numpy.typing import NDArray

from volguard.config import EvalConfig
from volguard.models.fold_runner import FoldContext, HyperParams
from volguard.models.types import FittedBaseline

FloatArray = NDArray[np.float64]


class PersistenceBaseline:
    """Exact one-day persistence; no hyperparameters."""

    model_id = "b0"

    def fit(
        self,
        ctx: FoldContext,
        *,
        cfg: EvalConfig,
        tune: bool,
        frozen_hyperparameters: HyperParams | None,
    ) -> FittedBaseline:
        del cfg, tune, frozen_hyperparameters
        return FittedBaseline(
            model_id=self.model_id,
            fold_id=ctx.fold.fold_id,
            train_start=ctx.fold.train_start,
            train_end=ctx.fold.train_end,
            hyperparameters={},
            state={},
        )

    def predict_next(
        self,
        fitted: FittedBaseline,
        *,
        ctx: FoldContext,
        history_end: int,
        issue_date: date,
        target_date: date,
        cfg: EvalConfig,
    ) -> FloatArray:
        del fitted, target_date, cfg
        if history_end <= 0:
            raise ValueError("persistence requires at least one realized surface")
        if ctx.dates[history_end - 1] != issue_date:
            raise ValueError("history must end at the issue date")
        return np.asarray(ctx.grid_w[history_end - 1], dtype=np.float64).copy()
