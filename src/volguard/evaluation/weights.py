"""Evaluation weight schemes from target geometry (distinct from training weights)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from volguard.models.baselines.common import black76_vega_grid

FloatArray = NDArray[np.float64]
WeightScheme = str
_GRID_SHAPE = (6, 9)


def total_variance_to_iv(w_grid: FloatArray, taus: FloatArray) -> FloatArray:
    """Convert total variance ``w`` to IV ``sqrt(w/τ)`` on a 6x9 grid."""
    w = np.asarray(w_grid, dtype=np.float64)
    tau = np.asarray(taus, dtype=np.float64)
    if w.shape != _GRID_SHAPE:
        raise ValueError("w_grid must have shape (6, 9)")
    if tau.shape != (6,):
        raise ValueError("taus must have shape (6,)")
    if np.any(tau <= 0.0):
        raise ValueError("taus must be positive")
    return np.sqrt(np.maximum(w, 0.0) / tau[:, None])


def metric_weights(
    *,
    scheme: WeightScheme,
    reliability: FloatArray,
    k_grid: FloatArray,
    w_grid: FloatArray,
    taus: FloatArray,
) -> FloatArray:
    """Build nonnegative metric weights for one target surface.

    Schemes:
    - ``uniform``: ones
    - ``vega``: Black-76 vega from target ``k`` / ``w``
    - ``vega_reliability``: vega x M5 reliability
    """
    reliability_a = np.asarray(reliability, dtype=np.float64)
    if reliability_a.shape != _GRID_SHAPE:
        raise ValueError("reliability must have shape (6, 9)")
    if scheme == "uniform":
        return np.ones(_GRID_SHAPE, dtype=np.float64)
    if scheme == "vega":
        return np.maximum(black76_vega_grid(k_grid, w_grid, taus), 0.0)
    if scheme == "vega_reliability":
        vega = np.maximum(black76_vega_grid(k_grid, w_grid, taus), 0.0)
        return vega * np.maximum(reliability_a, 0.0)
    raise ValueError(f"unknown weight scheme: {scheme}")
