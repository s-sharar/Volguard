"""Pre/post arbitrage metrics in fixed-k coordinates (M4).

Compute two ``ArbitrageReport``s per snap on the *identical* shared fixed-k grid
so the market's own violations and the fitted surface's residual violations are
directly comparable (design "Component 5" reuse notes and the
"Algorithm: pre/post arbitrage metrics" pseudocode; Requirement 6):

- ``arb_pre``  — the MARKET's own violations: the raw per-expiry observations
  binned onto the shared fixed-k grid (``w = iv^2 * tau`` interpolated across the
  observed ``(k, w)`` points with :func:`numpy.interp`, which clamps flat beyond
  the observed strike range).
- ``arb_post`` — the FITTED surface's violations: each fitted slice evaluated on
  the same fixed-k grid via ``svi_total_variance``.

Both grids are the shared ``k = linspace(arb_check_k_min, arb_check_k_max,
arb_check_points)`` and share the fit's tau ordering (Requirement 6.3), so the
two reports are comparable cell-for-cell.

The M1 arbitrage checker (``surface/arbitrage.py`` ``check_surface``) is reused,
never reimplemented (Requirement 6.5); this module only assembles the two
total-variance tensors it consumes.
"""

from __future__ import annotations

import numpy as np

from volguard.config import SurfaceConfig
from volguard.surface.arbitrage import ArbitrageReport, check_surface
from volguard.surface.calendar_fit import SurfaceFit
from volguard.surface.grid import SurfaceGrid
from volguard.surface.svi import svi_total_variance
from volguard.surface.types import ExpiryObs, FloatArray


def _market_w_row(o: ExpiryObs, k_grid: FloatArray) -> FloatArray:
    """Bin one expiry's raw observations onto the shared fixed-k grid.

    Builds observed total variance ``w = iv^2 * tau`` at the observed
    log-moneyness points and maps it onto ``k_grid`` with :func:`numpy.interp`
    over the sorted observed ``(k, w)`` points. ``numpy.interp`` clamps to the
    endpoint values beyond the observed strike range (flat extrapolation), which
    keeps the binned row finite and non-negative for the M1 checker.
    """
    k_obs = np.asarray(o.k, dtype=float)
    w_obs = np.asarray(o.iv, dtype=float) ** 2 * o.tau
    # Collapse duplicate log-moneyness points (e.g. call+put rows at the same
    # strike) to a single mean total variance per unique k, so the interpolation
    # input is strictly increasing. Without this, np.interp keeps only one of the
    # duplicates and the pre-fit market metrics would depend on row ordering.
    k_unique, inverse = np.unique(k_obs, return_inverse=True)
    w_sums = np.zeros_like(k_unique)
    counts = np.zeros_like(k_unique)
    np.add.at(w_sums, inverse, w_obs)
    np.add.at(counts, inverse, 1.0)
    w_mean = w_sums / counts
    return np.interp(k_grid, k_unique, w_mean)


def surface_arb_metrics(
    obs: list[ExpiryObs], fit: SurfaceFit, grid: SurfaceGrid, cfg: SurfaceConfig
) -> tuple[ArbitrageReport, ArbitrageReport]:
    """Return ``(arb_pre, arb_post)``: market vs fitted-surface violations.

    Both reports are computed on the identical shared fixed-k grid
    ``linspace(cfg.arb_check_k_min, cfg.arb_check_k_max, cfg.arb_check_points)``
    and share the fit's increasing-``tau`` ordering (Requirement 6.1-6.3):

    - ``arb_pre`` bins each fitted expiry's *raw* observations onto the grid
      (matched to ``fit.slices`` by ``expiry``) and runs the M1 ``check_surface``.
    - ``arb_post`` evaluates each fitted slice's ``svi_total_variance`` on the
      same grid and runs the M1 ``check_surface``.

    ``fit.slices`` is already ordered by increasing ``tau`` and distinct expiries
    give distinct ``tau``s, so the ``taus`` passed to ``check_surface`` are
    strictly increasing as its calendar check requires. A single-slice surface
    yields an empty calendar difference (zero calendar violations).

    The ``grid`` argument is accepted for interface symmetry with the driver
    (the post metrics are defined on the fixed-k grid, not the standardized-``d``
    output tensor, per the coordinate-separation decision) and is not resampled
    here.
    """
    _ = grid  # post metrics live in fixed-k; the standardized-d grid is not used
    k_grid = np.linspace(cfg.arb_check_k_min, cfg.arb_check_k_max, cfg.arb_check_points)
    taus = np.asarray([s.tau for s in fit.slices], dtype=float)

    # Align raw observations to the fit's slice ordering by expiry so w_market
    # and w_fit share the same tau ordering and the same k_grid.
    obs_by_expiry = {o.expiry: o for o in obs}

    w_market = np.stack([_market_w_row(obs_by_expiry[s.expiry], k_grid) for s in fit.slices])
    arb_pre = check_surface(k_grid, w_market, taus)

    w_fit = np.stack([svi_total_variance(s.params, k_grid) for s in fit.slices])
    arb_post = check_surface(k_grid, w_fit, taus)

    return arb_pre, arb_post
