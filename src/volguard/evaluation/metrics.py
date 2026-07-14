"""Atomic losses, aggregation, and skill vs B0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_GRID_SHAPE = (6, 9)


@dataclass(frozen=True, slots=True)
class SurfaceLoss:
    """Weighted surface-level loss for one forecast day."""

    mse: float
    mae: float
    rmse: float
    weight_sum: float
    n_cells: int


def cell_squared_errors(pred: FloatArray, actual: FloatArray) -> FloatArray:
    pred_a = np.asarray(pred, dtype=np.float64)
    actual_a = np.asarray(actual, dtype=np.float64)
    if pred_a.shape != actual_a.shape:
        raise ValueError("pred and actual must share a shape")
    return (pred_a - actual_a) ** 2


def cell_abs_errors(pred: FloatArray, actual: FloatArray) -> FloatArray:
    pred_a = np.asarray(pred, dtype=np.float64)
    actual_a = np.asarray(actual, dtype=np.float64)
    if pred_a.shape != actual_a.shape:
        raise ValueError("pred and actual must share a shape")
    return np.abs(pred_a - actual_a)


def weighted_surface_loss(pred: FloatArray, actual: FloatArray, weights: FloatArray) -> SurfaceLoss:
    """Single-surface weighted MSE / MAE / RMSE."""
    sq = cell_squared_errors(pred, actual)
    abs_e = cell_abs_errors(pred, actual)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != sq.shape:
        raise ValueError("weights must match pred/actual shape")
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        mse = float(np.mean(sq))
        mae = float(np.mean(abs_e))
    else:
        mse = float(np.sum(w * sq) / weight_sum)
        mae = float(np.sum(w * abs_e) / weight_sum)
    return SurfaceLoss(
        mse=mse,
        mae=mae,
        rmse=float(np.sqrt(mse)),
        weight_sum=weight_sum if weight_sum > 0.0 else float(sq.size),
        n_cells=int(sq.size),
    )


def aggregate_from_atomic(
    squared: FloatArray,
    absolute: FloatArray,
    weights: FloatArray,
) -> SurfaceLoss:
    """Aggregate metric from stacked atomic cell losses (not mean of RMSEs)."""
    sq = np.asarray(squared, dtype=np.float64).ravel()
    abs_e = np.asarray(absolute, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if sq.shape != abs_e.shape or sq.shape != w.shape:
        raise ValueError("atomic arrays must share a length")
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        mse = float(np.mean(sq)) if sq.size else 0.0
        mae = float(np.mean(abs_e)) if abs_e.size else 0.0
        weight_sum = float(sq.size)
    else:
        mse = float(np.sum(w * sq) / weight_sum)
        mae = float(np.sum(w * abs_e) / weight_sum)
    return SurfaceLoss(
        mse=mse,
        mae=mae,
        rmse=float(np.sqrt(mse)),
        weight_sum=weight_sum,
        n_cells=int(sq.size),
    )


def skill_score(model_mse: float, baseline_mse: float) -> float:
    """Skill vs B0: ``1 - MSE_model / MSE_B0``."""
    if not np.isfinite(model_mse) or not np.isfinite(baseline_mse):
        raise ValueError("MSE values must be finite")
    if baseline_mse < 0.0 or model_mse < 0.0:
        raise ValueError("MSE values must be nonnegative")
    if baseline_mse == 0.0:
        return 0.0 if model_mse == 0.0 else float("-inf")
    return 1.0 - model_mse / baseline_mse
