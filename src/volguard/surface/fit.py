"""Fit raw-SVI slices to observed implied volatilities (vega-weighted).

Given per-expiry observations ``(k_i, iv_i)`` with optional vega weights, find
the five raw-SVI parameters minimizing the vega-weighted squared error in total
variance. Uses ``scipy.optimize.least_squares`` (Trust Region Reflective) with a
parameter-domain reparameterization so the optimizer stays in the valid SVI
region, plus a soft butterfly penalty that pushes fits toward g(k) >= 0.

Fitting is separated from the pure parameterization (``svi.py``) so the math
core stays dependency-light and property-testable; this module is where the one
heavier dependency (scipy.optimize) lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize_scalar

from volguard.surface.svi import SVIParams, svi_g, svi_total_variance

FloatArray = NDArray[np.float64]

_MIN_OBS_FOR_FIT = 5
# SVI is a continuous smile: once params are stored, downstream code prices
# arbitrary strikes, so butterfly checks must not miss arbitrage between quotes.
# Two scales matter. Far wings: a fixed wide range (±5 log-moneyness, K/F in
# ~[0.007, 148], any tradeable Deribit strike) unioned with the observed range.
# Vertex: butterfly pockets sit within O(100·sigma) of the smile minimum and
# their width scales with sigma, so a fixed coarse grid steps right over a thin
# pocket for small sigma; sample ±200·sigma around m at ~0.5·sigma resolution.
_ARB_GRID_HALFWIDTH = 5.0
_ARB_WING_POINTS = 100
_ARB_VERTEX_SIGMAS = 200.0
_ARB_VERTEX_POINTS = 801


def _arbitrage_grid(k: FloatArray, params: SVIParams) -> FloatArray:
    """Butterfly-check grid: wide wings, observed range, and sigma-scaled vertex.

    Fixed length (wings + vertex, no dedup) so it can seed the fit penalty's
    residual vector, which ``least_squares`` requires to stay constant-length.
    """
    lo = min(float(k.min()), -_ARB_GRID_HALFWIDTH)
    hi = max(float(k.max()), _ARB_GRID_HALFWIDTH)
    wings = np.linspace(lo, hi, _ARB_WING_POINTS)
    half = _ARB_VERTEX_SIGMAS * params.sigma
    vertex = np.linspace(params.m - half, params.m + half, _ARB_VERTEX_POINTS)
    return np.concatenate([wings, vertex])


def _minimum_svi_g(params: SVIParams, k: FloatArray) -> float:
    """Minimize g across the traded wings and sigma-scaled vertex region."""
    half = _ARB_VERTEX_SIGMAS * params.sigma
    lo = min(float(k.min()), -_ARB_GRID_HALFWIDTH, params.m - half)
    hi = max(float(k.max()), _ARB_GRID_HALFWIDTH, params.m + half)
    edges = np.linspace(lo, hi, _ARB_WING_POINTS)
    min_g = float(np.min(svi_g(params, edges)))
    if min_g < 0.0:
        return min_g

    def objective(x: float) -> float:
        return float(svi_g(params, x))

    # The fit penalty needs a fixed residual grid, but final certification can
    # afford bounded minimization. Searching every interval catches thin
    # negative-g pockets even when the combined wing/vertex span makes the
    # initial 100-point sampling coarse, without slowing each fit step.
    xatol = min(1e-8, max(1e-12, params.sigma * 1e-3))
    for left, right in pairwise(edges):
        result = minimize_scalar(
            objective,
            bounds=(float(left), float(right)),
            method="bounded",
            options={"xatol": xatol},
        )
        if not result.success or not np.isfinite(result.fun):
            return -np.inf
        min_g = min(min_g, float(result.fun))
        if min_g < 0.0:
            return min_g
    return min_g


@dataclass(frozen=True, slots=True)
class SVIFitResult:
    """Outcome of an SVI slice fit."""

    params: SVIParams
    rmse: float  # RMSE in implied-vol points on the fitted grid
    n_obs: int
    vega_sum: float
    butterfly_ok: bool
    success: bool


def _initial_guess(k: FloatArray, w: FloatArray) -> np.ndarray:
    """Heuristic starting point for the optimizer from the observed smile."""
    w_min = float(np.min(w))
    a0 = max(w_min * 0.5, 1e-4)
    b0 = 0.1
    rho0 = -0.1
    m0 = float(k[np.argmin(w)])
    sigma0 = 0.1
    return np.array([a0, b0, rho0, m0, sigma0])


def _unpack(theta: np.ndarray) -> SVIParams:
    """Map an unconstrained optimizer vector to valid SVI params.

    b -> softplus (>=0), rho -> tanh (in (-1,1)), sigma -> softplus (>0).
    ``a`` is shifted so the wing constraint ``a + b*sigma*sqrt(1-rho^2) >= 0``
    holds by construction.
    """
    a_raw, b_raw, rho_raw, m, sigma_raw = theta
    b = np.log1p(np.exp(-abs(b_raw))) + max(b_raw, 0.0)  # numerically stable softplus
    sigma = np.log1p(np.exp(-abs(sigma_raw))) + max(sigma_raw, 0.0) + 1e-6
    rho = np.tanh(rho_raw)
    wing = b * sigma * np.sqrt(1.0 - rho * rho)
    # a_raw parameterizes the slack above the wing lower bound.
    slack = np.log1p(np.exp(-abs(a_raw))) + max(a_raw, 0.0)
    a = slack - wing
    return SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma))


def fit_svi_slice(
    k: FloatArray,
    iv: FloatArray,
    tau: float,
    *,
    vega: FloatArray | None = None,
    butterfly_penalty: float = 10.0,
    max_nfev: int = 2000,
) -> SVIFitResult:
    """Fit a raw-SVI slice to observed IVs by vega-weighted least squares.

    Args:
        k: log-moneyness observations, shape (n,).
        iv: implied vols at ``k``, shape (n,).
        tau: expiry in years.
        vega: optional weights (typically Black-76 vega); defaults to equal.
        butterfly_penalty: weight on soft g(k) < 0 residuals.
        max_nfev: optimizer iteration cap.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    if k.shape != iv.shape:
        raise ValueError("k and iv must have the same shape")
    if k.size < _MIN_OBS_FOR_FIT:
        raise ValueError("need at least 5 observations to fit 5 SVI params")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")

    w_obs = iv * iv * tau
    weights = np.ones_like(k) if vega is None else np.asarray(vega, dtype=float)
    weights = np.sqrt(np.maximum(weights, 1e-8))

    def residuals(theta: np.ndarray) -> np.ndarray:
        params = _unpack(theta)
        w_model = svi_total_variance(params, k)
        fit_res = weights * (w_model - w_obs)
        if butterfly_penalty > 0.0:
            # Recenter the check grid on the current smile so the penalty sees
            # narrow vertex pockets; fixed length keeps least_squares happy.
            g = svi_g(params, _arbitrage_grid(k, params))
            pen = butterfly_penalty * np.minimum(g, 0.0)
            return np.concatenate([fit_res, pen])
        return fit_res

    theta0 = _initial_guess(k, w_obs)
    sol = least_squares(residuals, theta0, method="trf", max_nfev=max_nfev)
    params = _unpack(sol.x)

    w_model = svi_total_variance(params, k)
    iv_model = np.sqrt(np.maximum(w_model, 0.0) / tau)
    rmse = float(np.sqrt(np.mean((iv_model - iv) ** 2)))
    butterfly_ok = _minimum_svi_g(params, k) >= 0.0
    return SVIFitResult(
        params=params,
        rmse=rmse,
        n_obs=int(k.size),
        vega_sum=float(np.sum(weights**2)),
        butterfly_ok=butterfly_ok,
        success=bool(sol.success),
    )
