"""Standardized-moneyness grid tensor sampler (M4).

Sample the canonical per-tenor SVI parameters into the output grid tensor
``w[tenor, moneyness]`` on *standardized* moneyness ``d`` and record per-cell
provenance (``n_obs``, ``interp_flag``). This is the derived-view half of the
project's deliberate coordinate-separation decision: fits and arbitrage checks
live in fixed-k coordinates ``k = ln(K/F)``, while the output grid is sampled in
standardized moneyness ``d = k / (sigma_atm * sqrt(tau))`` so tenors are
comparable across the term structure. Each ``d`` is mapped back to a fixed ``k``
per tenor via that tenor's ATM vol before ``svi_total_variance`` is evaluated
(design "Coordinate model" and Component 4 "surface/grid.py").

The M1 math core (``surface/svi.py``) and the M4 fit output
(``surface/calendar_fit.py``) are reused, never reimplemented.

ATM term-structure handling (Requirement 5.4): output tenors that coincide with
a fitted expiry (within a small tolerance) use that slice's own params, its own
ATM total variance ``w(0)``, and its ``n_obs`` with ``interp_flag = False``.
Output tenors that do not coincide with any fitted expiry are flagged
``interp_flag = True``; their ATM total variance ``theta_j`` is obtained by
linear interpolation of the fitted ``(tau_s, theta_s)`` term structure via
``numpy.interp`` (which clamps flat beyond the fitted range, i.e. flat
extrapolation at the ends), and the *nearest* fitted slice supplies both the
smile shape (its params) and the provenance ``n_obs``. In all cases the ATM vol
used for the ``d -> k`` mapping is ``sigma_atm(tau_j) = sqrt(theta_source / tau_j)``
where ``theta_source`` is the interpolated ``theta_j`` for interpolated tenors
and the slice's own ``w(0)`` for exact tenors, so the sampled ATM term structure
is honored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volguard.config import SurfaceConfig
from volguard.surface.calendar_fit import SurfaceFit
from volguard.surface.svi import SVIParams, svi_total_variance
from volguard.surface.types import FloatArray

# Relative tolerance for deciding a requested output tenor coincides with a
# fitted expiry's ``tau`` (both are in years; e.g. 30/365 vs 30/365).
_TENOR_MATCH_RTOL = 1e-9
_TENOR_MATCH_ATOL = 1e-12

_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class SurfaceGrid:
    """Derived standardized-moneyness grid tensor with per-cell provenance.

    Shapes: ``tenors_days`` is ``(n_tenor,)``, ``moneyness`` is ``(n_money,)``,
    and every 2-D field is ``(n_tenor, n_money)``.
    """

    tenors_days: FloatArray  # (n_tenor,) requested output tenors in days
    moneyness: FloatArray  # (n_money,) standardized d axis
    k_grid: FloatArray  # (n_tenor, n_money) fixed-k per cell
    w: FloatArray  # (n_tenor, n_money) total variance
    n_obs: NDArray[np.int64]  # (n_tenor, n_money) source-expiry observation count
    interp_flag: NDArray[np.bool_]  # (n_tenor, n_money) interpolated/extrapolated tenor


def _source_for_tenor(
    fit: SurfaceFit,
    tau: float,
    fitted_taus: FloatArray,
    fitted_thetas: FloatArray,
) -> tuple[SVIParams, float, int, bool]:
    """Resolve the smile shape, ATM total variance, provenance, and interp flag.

    Returns ``(params, theta_source, n_obs, interp)`` for output tenor ``tau``
    (years). An exact match to a fitted expiry uses that slice directly; a
    non-matching tenor interpolates ``theta`` on the fitted term structure and
    borrows the nearest slice's shape and ``n_obs``.
    """
    # Exact-tenor path: coincides with a fitted expiry within tolerance.
    for slice_fit in fit.slices:
        if math.isclose(slice_fit.tau, tau, rel_tol=_TENOR_MATCH_RTOL, abs_tol=_TENOR_MATCH_ATOL):
            theta = float(svi_total_variance(slice_fit.params, 0.0))
            return slice_fit.params, theta, slice_fit.n_obs, False

    # Interpolated/extrapolated path: theta from the fitted term structure
    # (np.interp clamps flat beyond the ends), shape/provenance from the nearest
    # fitted slice.
    theta_j = float(np.interp(tau, fitted_taus, fitted_thetas))
    nearest_idx = int(np.argmin(np.abs(fitted_taus - tau)))
    nearest = fit.slices[nearest_idx]
    return nearest.params, theta_j, nearest.n_obs, True


def sample_grid(fit: SurfaceFit, cfg: SurfaceConfig) -> SurfaceGrid:
    """Sample fitted SVI params into the standardized-moneyness grid tensor.

    Preconditions: ``fit.slices`` is non-empty and ordered by increasing
    ``tau``; every ``slice.params`` is a valid ``SVIParams``.

    Postconditions: the returned tensor has shape
    ``(len(cfg.tenor_grid_days), len(cfg.moneyness_grid))``; ``w >= 0`` in every
    cell (non-negativity floor, Requirement 5.3);
    ``k_grid[j, i] == d_i * sigma_atm(tau_j) * sqrt(tau_j)`` (fixed-k mapping,
    Requirement 5.2); ``interp_flag[j, i]`` is True iff output tenor ``j`` does
    not coincide with a fitted expiry (Requirement 5.4).
    """
    if not fit.slices:
        raise ValueError("sample_grid requires a non-empty SurfaceFit.slices")

    tenors_days = np.asarray(cfg.tenor_grid_days, dtype=float)
    moneyness = np.asarray(cfg.moneyness_grid, dtype=float)
    tenors_years = tenors_days / _DAYS_PER_YEAR

    # ATM total-variance term structure of the fitted slices: theta_s = w_s(0).
    fitted_taus = np.asarray([s.tau for s in fit.slices], dtype=float)
    fitted_thetas = np.asarray(
        [float(svi_total_variance(s.params, 0.0)) for s in fit.slices], dtype=float
    )

    n_tenor = len(tenors_years)
    n_money = len(moneyness)
    k_grid = np.zeros((n_tenor, n_money), dtype=float)
    w = np.zeros((n_tenor, n_money), dtype=float)
    n_obs = np.zeros((n_tenor, n_money), dtype=np.int64)
    interp_flag = np.zeros((n_tenor, n_money), dtype=np.bool_)

    for j, tau in enumerate(tenors_years):
        params, theta_source, source_n_obs, interp = _source_for_tenor(
            fit, tau, fitted_taus, fitted_thetas
        )
        # ATM vol at this tenor from the (possibly interpolated) ATM total
        # variance, honoring the sampled term structure.
        sig_atm = math.sqrt(theta_source / tau)
        sqrt_tau = math.sqrt(tau)
        # Rescale the carrier smile so its ATM total variance equals
        # theta_source: the shape comes from the nearest fitted slice, but the
        # level must follow the interpolated term structure, otherwise the ATM
        # cell (d=0 -> k=0) would emit the nearest slice's variance and produce
        # a stepwise ATM term structure across interpolated tenors. For an exact
        # tenor theta_source == w(0) so the scale is exactly 1.0 (no-op).
        theta_params = float(svi_total_variance(params, 0.0))
        scale = theta_source / theta_params if theta_params > 0.0 else 1.0
        for i, d in enumerate(moneyness):
            k = float(d) * sig_atm * sqrt_tau  # standardized d -> fixed k
            k_grid[j, i] = k
            # non-negativity floor; scale lifts the smile to the interpolated ATM level
            w[j, i] = max(float(svi_total_variance(params, k)) * scale, 0.0)
            n_obs[j, i] = source_n_obs
            interp_flag[j, i] = interp

    return SurfaceGrid(
        tenors_days=tenors_days,
        moneyness=moneyness,
        k_grid=k_grid,
        w=w,
        n_obs=n_obs,
        interp_flag=interp_flag,
    )
