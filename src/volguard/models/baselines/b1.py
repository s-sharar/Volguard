"""B1 per-cell recursive EWMA baseline."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volguard.config import EvalConfig
from volguard.models.baselines.common import ewma_forecast, train_cell_weights, weighted_mse
from volguard.models.fold_runner import FoldContext, HyperParams
from volguard.models.types import FittedBaseline

FloatArray = NDArray[np.float64]


def _one_step_train_loss(ctx: FoldContext, lam: float, taus: FloatArray) -> float:
    """Training-only weighted one-step MSE for a candidate λ."""
    losses: list[float] = []
    for index in ctx.train_indices:
        if index == 0:
            continue
        history = ctx.grid_w[:index]
        pred = ewma_forecast(history, lam)
        target = ctx.grid_w[index]
        weights = train_cell_weights(
            ctx.reliability[index],
            ctx.grid_k[index],
            target,
            taus,
        )
        losses.append(weighted_mse(pred, target, weights))
    if not losses:
        raise ValueError("need at least two training days to tune EWMA")
    return float(np.mean(losses))


class EWMABaseline:
    """Per-cell recursive EWMA with λ chosen on training-only weighted MSE."""

    model_id = "b1"

    def fit(
        self,
        ctx: FoldContext,
        *,
        cfg: EvalConfig,
        tune: bool,
        frozen_hyperparameters: HyperParams | None,
    ) -> FittedBaseline:
        lambdas = list(cfg.baselines.ewma_lambdas)
        if frozen_hyperparameters is not None and "lambda" in frozen_hyperparameters:
            chosen = float(frozen_hyperparameters["lambda"])  # type: ignore[arg-type]
            trace: list[dict[str, Any]] = [{"lambda": chosen, "source": "frozen"}]
        elif tune or frozen_hyperparameters is None:
            # B1 selects λ with training-only weighted one-step MSE.
            scored = [(_one_step_train_loss(ctx, lam, ctx.taus), lam) for lam in lambdas]
            scored.sort(key=lambda item: (item[0], item[1]))
            chosen = float(scored[0][1])
            trace = [{"lambda": lam, "train_vw_mse": loss} for loss, lam in scored]
        else:
            chosen = float(lambdas[0])
            trace = [{"lambda": chosen, "source": "default"}]
        return FittedBaseline(
            model_id=self.model_id,
            fold_id=ctx.fold.fold_id,
            train_start=ctx.fold.train_start,
            train_end=ctx.fold.train_end,
            hyperparameters={"lambda": chosen},
            state={"tuning_trace": trace},
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
        del target_date, cfg
        if history_end <= 0:
            raise ValueError("EWMA requires at least one realized surface")
        if ctx.dates[history_end - 1] != issue_date:
            raise ValueError("history must end at the issue date")
        lam = float(fitted.hyperparameters["lambda"])
        return ewma_forecast(ctx.grid_w[:history_end], lam)
