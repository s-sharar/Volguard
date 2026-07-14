"""B3 train-only PCA + VAR(1) on scores, deterministic reconstruction."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volguard.config import EvalConfig
from volguard.datasets.pca import fit_surface_pca, transform_surface_pca
from volguard.models.baselines.common import (
    fit_var1,
    flatten_grids,
    reconstruct_surface_pca,
    reshape_grid,
    train_cell_weights,
    weighted_mse,
)
from volguard.models.fold_runner import FoldContext, HyperParams
from volguard.models.types import FittedBaseline

FloatArray = NDArray[np.float64]


def _validation_loss(
    ctx: FoldContext,
    n_components: int,
    train_grids: FloatArray,
    train_dates: list[date],
) -> float:
    pca = fit_surface_pca(train_grids, n_components=n_components, fit_dates=train_dates)
    train_scores = transform_surface_pca(pca, train_grids)
    c, a_mat = fit_var1(train_scores)
    losses: list[float] = []
    for index in ctx.validation_indices:
        if index == 0:
            continue
        # Issue from all history before target using frozen PCA/VAR from train.
        hist = flatten_grids(ctx.grid_w[:index])
        scores = transform_surface_pca(pca, hist)
        next_score = c + a_mat @ scores[-1]
        pred = reshape_grid(reconstruct_surface_pca(pca.mean, pca.components, next_score))
        weights = train_cell_weights(
            ctx.reliability[index],
            ctx.grid_k[index],
            ctx.grid_w[index],
            ctx.taus,
        )
        losses.append(weighted_mse(pred, ctx.grid_w[index], weights))
    if not losses:
        # Fall back to train one-step loss if validation is empty.
        for local in range(1, train_grids.shape[0]):
            next_score = c + a_mat @ train_scores[local - 1]
            pred = reshape_grid(reconstruct_surface_pca(pca.mean, pca.components, next_score))
            target = reshape_grid(train_grids[local])
            k = ctx.grid_k[ctx.train_indices[local]]
            rel = ctx.reliability[ctx.train_indices[local]]
            weights = train_cell_weights(rel, k, target, ctx.taus)
            losses.append(weighted_mse(pred, target, weights))
    return float(np.mean(losses)) if losses else float("inf")


class PCAVARBaseline:
    """Train-only PCA (k in {3,4,5}) + VAR(1); reconstruct deterministically."""

    model_id = "b3"

    def fit(
        self,
        ctx: FoldContext,
        *,
        cfg: EvalConfig,
        tune: bool,
        frozen_hyperparameters: HyperParams | None,
    ) -> FittedBaseline:
        train_idx = list(ctx.train_indices)
        train_grids = flatten_grids(ctx.grid_w[train_idx])
        train_dates = [ctx.dates[i] for i in train_idx]
        candidates = list(cfg.baselines.pca_components)

        if frozen_hyperparameters is not None and "n_components" in frozen_hyperparameters:
            n_components = int(frozen_hyperparameters["n_components"])  # type: ignore[arg-type]
            trace: list[dict[str, Any]] = [{"n_components": n_components, "source": "frozen"}]
        elif tune:
            scored = [(_validation_loss(ctx, n, train_grids, train_dates), n) for n in candidates]
            best_loss = min(loss for loss, _ in scored)
            # Prefer the smallest rank among near-ties (exact low-rank processes).
            near = [
                n
                for loss, n in scored
                if loss <= best_loss + 1e-10 or loss <= best_loss * (1.0 + 1e-8)
            ]
            n_components = int(min(near))
            trace = [{"n_components": n, "validation_vw_mse": loss} for loss, n in scored]
        else:
            n_components = int(candidates[0])
            trace = [{"n_components": n_components, "source": "default"}]

        pca = fit_surface_pca(train_grids, n_components=n_components, fit_dates=train_dates)
        scores = transform_surface_pca(pca, train_grids)
        c, a_mat = fit_var1(scores)
        return FittedBaseline(
            model_id=self.model_id,
            fold_id=ctx.fold.fold_id,
            train_start=ctx.fold.train_start,
            train_end=ctx.fold.train_end,
            hyperparameters={"n_components": n_components},
            state={
                "mean": np.asarray(pca.mean, dtype=np.float64),
                "components": np.asarray(pca.components, dtype=np.float64),
                "var_c": np.asarray(c, dtype=np.float64),
                "var_a": np.asarray(a_mat, dtype=np.float64),
                "tuning_trace": trace,
            },
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
            raise ValueError("PCA-VAR requires at least one realized surface")
        if ctx.dates[history_end - 1] != issue_date:
            raise ValueError("history must end at the issue date")
        mean = np.asarray(fitted.state["mean"], dtype=np.float64)
        components = np.asarray(fitted.state["components"], dtype=np.float64)
        c = np.asarray(fitted.state["var_c"], dtype=np.float64)
        a_mat = np.asarray(fitted.state["var_a"], dtype=np.float64)
        hist = flatten_grids(ctx.grid_w[:history_end])
        # Manual transform with frozen mean/components (avoid SurfacePCA date checks).
        scores = (hist - mean) @ components.T
        next_score = c + a_mat @ scores[-1]
        return reshape_grid(reconstruct_surface_pca(mean, components, next_score))
