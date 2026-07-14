"""Shared helpers for baseline fitting, weighting, and PCA reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from volguard.curate.blackiv import black76_vega

FloatArray = NDArray[np.float64]
_GRID_SHAPE = (6, 9)
_GRID_CELLS = 54
_GRID_NDIM = 3
_MATRIX_NDIM = 2
_W_FLOOR = 1e-12
_MIN_SERIES = 2
_SINGULAR_EPS = 1e-18


def flatten_grids(grids: FloatArray) -> FloatArray:
    """Reshape ``(n, 6, 9)`` -> ``(n, 54)``."""
    array = np.asarray(grids, dtype=np.float64)
    if array.ndim != _GRID_NDIM or array.shape[1:] != _GRID_SHAPE:
        raise ValueError("grids must have shape (n, 6, 9)")
    return array.reshape(array.shape[0], -1)


def reshape_grid(flat: FloatArray) -> FloatArray:
    """Reshape ``(54,)`` or ``(n, 54)`` to grid form."""
    array = np.asarray(flat, dtype=np.float64)
    if array.ndim == 1:
        if array.size != _GRID_CELLS:
            raise ValueError("flat grid must have 54 cells")
        return array.reshape(_GRID_SHAPE)
    if array.ndim == _MATRIX_NDIM and array.shape[1] == _GRID_CELLS:
        return array.reshape(array.shape[0], *_GRID_SHAPE)
    raise ValueError("unsupported flat grid shape")


def black76_vega_grid(k_grid: FloatArray, w_grid: FloatArray, taus: FloatArray) -> FloatArray:
    """Normalized-forward Black-76 vega on a 6x9 native or shared grid."""
    k_grid = np.asarray(k_grid, dtype=np.float64)
    w_grid = np.asarray(w_grid, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if k_grid.shape != _GRID_SHAPE or w_grid.shape != _GRID_SHAPE:
        raise ValueError("k_grid and w_grid must have shape (6, 9)")
    if taus.shape != (6,):
        raise ValueError("taus must have shape (6,)")
    out = np.zeros(_GRID_SHAPE, dtype=np.float64)
    for j in range(6):
        tau = float(taus[j])
        for i in range(9):
            w = max(float(w_grid[j, i]), _W_FLOOR)
            sigma = float(np.sqrt(w / tau))
            strike = float(np.exp(k_grid[j, i]))
            out[j, i] = black76_vega(1.0, strike, tau, sigma)
    return out


def train_cell_weights(
    reliability: FloatArray,
    k_grid: FloatArray,
    w_grid: FloatArray,
    taus: FloatArray,
) -> FloatArray:
    """Training weights = Black-76 vega x M5 reliability (nonnegative)."""
    vega = black76_vega_grid(k_grid, w_grid, taus)
    weights = np.maximum(vega, 0.0) * np.maximum(reliability, 0.0)
    if not np.any(weights > 0.0):
        return np.ones_like(weights)
    return weights


def weighted_mse(pred: FloatArray, actual: FloatArray, weights: FloatArray) -> float:
    """Scalar weighted MSE over a surface or batch of surfaces."""
    pred_a = np.asarray(pred, dtype=np.float64)
    actual_a = np.asarray(actual, dtype=np.float64)
    weight_a = np.asarray(weights, dtype=np.float64)
    if pred_a.shape != actual_a.shape or weight_a.shape != pred_a.shape:
        raise ValueError("pred, actual, and weights must share a shape")
    residual = pred_a - actual_a
    denom = float(np.sum(weight_a))
    if denom <= 0.0:
        return float(np.mean(residual * residual))
    return float(np.sum(weight_a * residual * residual) / denom)


def reconstruct_surface_pca(
    mean: FloatArray, components: FloatArray, scores: FloatArray
) -> FloatArray:
    """Map PCA scores back to a flattened 54-vector (or batch)."""
    mean_a = np.asarray(mean, dtype=np.float64)
    components_a = np.asarray(components, dtype=np.float64)
    scores_a = np.asarray(scores, dtype=np.float64)
    if scores_a.ndim == 1:
        return mean_a + scores_a @ components_a
    return mean_a + scores_a @ components_a


def ewma_forecast(history: FloatArray, lam: float) -> FloatArray:
    """One-step EWMA forecast from a ``(T, ...)`` history ending at t.

    ``s_t = lam * x_t + (1-lam) * s_{t-1}`` with ``s_0 = x_0``; forecast is ``s_t``.
    ``lam=1`` is exact persistence.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    hist = np.asarray(history, dtype=np.float64)
    if hist.shape[0] == 0:
        raise ValueError("EWMA history must be non-empty")
    state = hist[0].copy()
    for t in range(1, hist.shape[0]):
        state = lam * hist[t] + (1.0 - lam) * state
    return state


def fit_ar1(series: FloatArray) -> tuple[float, float]:
    """OLS AR(1): ``x_{t+1} = c + phi * x_t``. Returns ``(c, phi)``."""
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1 or x.size < _MIN_SERIES:
        raise ValueError("AR(1) needs a 1D series with at least 2 points")
    x0 = x[:-1]
    x1 = x[1:]
    n = x0.size
    sx = float(x0.sum())
    sy = float(x1.sum())
    sxx = float(np.dot(x0, x0))
    sxy = float(np.dot(x0, x1))
    denom = n * sxx - sx * sx
    if abs(denom) < _SINGULAR_EPS:
        phi = 0.0
        c = float(np.mean(x1))
    else:
        phi = (n * sxy - sx * sy) / denom
        c = (sy - phi * sx) / n
    return c, phi


def fit_var1(scores: FloatArray) -> tuple[FloatArray, FloatArray]:
    """OLS VAR(1): ``z_{t+1} = c + A z_t``. Returns ``(c, A)`` with ``A`` shape ``(k,k)``."""
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != _MATRIX_NDIM or z.shape[0] < _MIN_SERIES:
        raise ValueError("VAR(1) needs shape (T, k) with T >= 2")
    z0 = z[:-1]
    z1 = z[1:]
    ones = np.ones((z0.shape[0], 1), dtype=np.float64)
    design = np.concatenate([ones, z0], axis=1)
    # coeffs: (1+k, k)
    coeffs, _, _, _ = np.linalg.lstsq(design, z1, rcond=None)
    c = coeffs[0]
    a_mat = coeffs[1:].T
    return c, a_mat


def ridge_closed_form(
    x: FloatArray, y: FloatArray, alpha: float, sample_weight: FloatArray | None = None
) -> FloatArray:
    """Weighted ridge with intercept absorbed by an added column of ones.

    Returns coefficient vector of length ``n_features + 1`` (intercept last).
    """
    x_a = np.asarray(x, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.float64)
    if x_a.ndim != _MATRIX_NDIM or y_a.ndim != 1 or x_a.shape[0] != y_a.shape[0]:
        raise ValueError("x must be (n, p) and y (n,)")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    n, p = x_a.shape
    if sample_weight is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError("sample_weight must have shape (n,)")
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    x_w = x_a * sqrt_w[:, None]
    y_w = y_a * sqrt_w
    ones = sqrt_w[:, None]
    design = np.concatenate([x_w, ones], axis=1)
    # Penalize only non-intercept columns.
    penalty = alpha * np.eye(p + 1, dtype=np.float64)
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ y_w
    return np.linalg.solve(gram, rhs)
