"""B4 per-cell weighted ridge using grid lags + RV/DVOL features."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volguard.config import EvalConfig
from volguard.models.baselines.common import ridge_closed_form, train_cell_weights
from volguard.models.fold_runner import FoldContext, HyperParams
from volguard.models.types import FittedBaseline

FloatArray = NDArray[np.float64]
_N_CELLS = 54
_SCALE_EPS = 1e-12


def _rolling_mean(history: FloatArray, window: int) -> FloatArray:
    """Mean of the last ``window`` surfaces in ``history`` (T, 6, 9)."""
    if history.shape[0] == 0:
        raise ValueError("rolling mean needs non-empty history")
    take = history[-window:] if history.shape[0] >= window else history
    return np.mean(take, axis=0)


def _feature_vector(
    ctx: FoldContext,
    history_end: int,
    mean_windows: list[int],
) -> FloatArray:
    """Build the B4 design row at the issue date (history_end - 1)."""
    issue = history_end - 1
    w_t = ctx.grid_w[issue].ravel()
    pieces: list[FloatArray] = [w_t]
    for window in mean_windows:
        pieces.append(_rolling_mean(ctx.grid_w[:history_end], window).ravel())
    # Scalar market features at the issue date.
    feat = ctx.features[issue]
    pieces.append(feat)
    return np.concatenate(pieces, axis=0)


def _impute_and_scale_train(
    rows: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Train-only median impute + missing indicators + z-scale (non-indicator cols).

    Returns ``(x_scaled, medians, scales, miss_template_shape)``.
    """
    x = np.asarray(rows, dtype=np.float64)
    _n_rows, p = x.shape
    medians = np.zeros(p, dtype=np.float64)
    scales = np.ones(p, dtype=np.float64)
    missing = ~np.isfinite(x)
    filled = x.copy()
    for j in range(p):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        med = float(np.median(finite)) if finite.size else 0.0
        medians[j] = med
        filled[:, j] = np.where(np.isfinite(col), col, med)
        std = float(np.std(filled[:, j]))
        scales[j] = std if std > _SCALE_EPS else 1.0
    indicators = missing.astype(np.float64)
    scaled = (filled - medians) / scales
    design = np.concatenate([scaled, indicators], axis=1)
    return design, medians, scales, indicators[:0]  # last unused


def _transform_row(row: FloatArray, medians: FloatArray, scales: FloatArray) -> FloatArray:
    missing = ~np.isfinite(row)
    filled = np.where(np.isfinite(row), row, medians)
    scaled = (filled - medians) / scales
    return np.concatenate([scaled, missing.astype(np.float64)], axis=0)


def _build_supervised(
    ctx: FoldContext,
    indices: list[int],
    mean_windows: list[int],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(X_raw, Y_flat, cell_weights_flat)`` for one-step targets in indices."""
    rows: list[FloatArray] = []
    targets: list[FloatArray] = []
    weights: list[FloatArray] = []
    for index in indices:
        if index == 0:
            continue
        rows.append(_feature_vector(ctx, index, mean_windows))
        targets.append(ctx.grid_w[index].ravel())
        weights.append(
            train_cell_weights(
                ctx.reliability[index],
                ctx.grid_k[index],
                ctx.grid_w[index],
                ctx.taus,
            ).ravel()
        )
    if not rows:
        raise ValueError("ridge needs at least one supervised pair")
    return np.asarray(rows), np.asarray(targets), np.asarray(weights)


class RidgeBaseline:
    """54 weighted ridge heads with train-only median impute / scale / indicators."""

    model_id = "b4"

    def fit(
        self,
        ctx: FoldContext,
        *,
        cfg: EvalConfig,
        tune: bool,
        frozen_hyperparameters: HyperParams | None,
    ) -> FittedBaseline:
        mean_windows = list(cfg.baselines.ridge_mean_windows)
        alphas = list(cfg.baselines.ridge_alphas)
        train_idx = list(ctx.train_indices)
        x_train_raw, y_train, w_train = _build_supervised(ctx, train_idx, mean_windows)
        x_train, medians, scales, _ = _impute_and_scale_train(x_train_raw)

        def _loss_for_alpha(alpha: float, x_raw: FloatArray, y: FloatArray, w: FloatArray) -> float:
            x = np.stack([_transform_row(row, medians, scales) for row in x_raw])
            total = 0.0
            weight_sum = 0.0
            for cell in range(_N_CELLS):
                beta = ridge_closed_form(x, y[:, cell], alpha, sample_weight=w[:, cell])
                pred = x @ beta[:-1] + beta[-1]
                total += float(np.sum(w[:, cell] * (pred - y[:, cell]) ** 2))
                weight_sum += float(np.sum(w[:, cell]))
            return total / weight_sum if weight_sum > 0 else float("inf")

        if frozen_hyperparameters is not None and "alpha" in frozen_hyperparameters:
            alpha = float(frozen_hyperparameters["alpha"])  # type: ignore[arg-type]
            trace: list[dict[str, Any]] = [{"alpha": alpha, "source": "frozen"}]
        elif tune and ctx.validation_indices:
            x_val_raw, y_val, w_val = _build_supervised(
                ctx, list(ctx.validation_indices), mean_windows
            )
            scored = [(_loss_for_alpha(a, x_val_raw, y_val, w_val), a) for a in alphas]
            scored.sort(key=lambda item: (item[0], item[1]))
            alpha = float(scored[0][1])
            trace = [{"alpha": a, "validation_vw_mse": loss} for loss, a in scored]
        else:
            # Training-only alpha selection when validation is unavailable.
            scored = [(_loss_for_alpha(a, x_train_raw, y_train, w_train), a) for a in alphas]
            scored.sort(key=lambda item: (item[0], item[1]))
            alpha = float(scored[0][1])
            trace = [{"alpha": a, "train_vw_mse": loss} for loss, a in scored]

        betas = np.zeros((_N_CELLS, x_train.shape[1] + 1), dtype=np.float64)
        for cell in range(_N_CELLS):
            betas[cell] = ridge_closed_form(
                x_train, y_train[:, cell], alpha, sample_weight=w_train[:, cell]
            )

        return FittedBaseline(
            model_id=self.model_id,
            fold_id=ctx.fold.fold_id,
            train_start=ctx.fold.train_start,
            train_end=ctx.fold.train_end,
            hyperparameters={"alpha": alpha, "mean_windows": mean_windows},
            state={
                "betas": betas,
                "medians": medians,
                "scales": scales,
                "tuning_trace": trace,
                "feature_names": ctx.feature_names,
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
        del target_date
        if history_end <= 0:
            raise ValueError("ridge requires at least one realized surface")
        if ctx.dates[history_end - 1] != issue_date:
            raise ValueError("history must end at the issue date")
        mean_windows = list(fitted.hyperparameters["mean_windows"])  # type: ignore[arg-type]
        raw = _feature_vector(ctx, history_end, mean_windows)
        medians = np.asarray(fitted.state["medians"], dtype=np.float64)
        scales = np.asarray(fitted.state["scales"], dtype=np.float64)
        betas = np.asarray(fitted.state["betas"], dtype=np.float64)
        design = _transform_row(raw, medians, scales)
        pred = betas[:, :-1] @ design + betas[:, -1]
        return pred.reshape(6, 9)
