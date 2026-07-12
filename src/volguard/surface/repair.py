"""Arbitrage-repair QP: project a surface onto the nearest arb-free one.

Given a (possibly arbitraging) total-variance grid ``w[tenor, k]``, find the
closest grid ``w*`` (in weighted L2) that satisfies discrete no-arbitrage:

- **calendar monotonicity**: ``w*[j+1, i] >= w*[j, i]`` for adjacent tenors
- **butterfly + vertical spread**: Black-76 call prices convex in strike with
  slope in [-1, 0] per tenor — the *same* conditions ``check_butterfly_from_w``
  enforces, so a "repaired" surface always passes the project's primary checker.

Call price is a nonlinear (monotone) function of total variance, so butterfly
convexity is not linear in ``w``. We therefore solve a **sequential QP**: at each
step we linearize the call prices around the current iterate (``C ≈ C0 + dC/dw *
(w - w0)``, with ``dC/dw = vega / (2·sigma·tau) > 0``), impose discrete
convexity of those linearized prices plus calendar monotonicity, solve the
convex QP, then relinearize. We iterate until the true checker passes or the
iterate stops moving.

This is the universal post-hoc repair layer applied to *every* model's output
(baselines and ML alike) so the memo can cleanly quantify "accuracy paid for
arbitrage consistency" by reporting each model raw AND repaired.

Solved with cvxpy + OSQP (both core dependencies).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from volguard.curate.blackiv import black76_price, black76_vega
from volguard.surface.arbitrage import check_surface

FloatArray = NDArray[np.float64]

_MIN_MONEYNESS_POINTS = 3
_MIN_TENORS_FOR_CALENDAR = 2
_W_FLOOR = 1e-10
# OSQP's default tolerances (1e-5) leave constraint residuals around 1e-7 that
# the arbitrage checker would then flag on the repaired surface; tight eps plus
# solution polishing brings residuals to ~1e-12.
_OSQP_OPTS: dict[str, object] = {
    "eps_abs": 1e-10,
    "eps_rel": 1e-10,
    "max_iter": 100_000,
    "polishing": True,
}


def _call_price_linearization(
    strikes: FloatArray, w_by_tenor: FloatArray, taus: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Black-76 call prices ``C0`` and slopes ``dC/dw`` at each grid cell.

    Forward is normalized to 1 (strikes are ``exp(k)``). ``dC/dw`` uses
    ``dC/dsigma = vega`` and ``dsigma/dw = 1/(2·sigma·tau)``.
    """
    n_tenor, nk = w_by_tenor.shape
    c0 = np.zeros((n_tenor, nk))
    dc_dw = np.zeros((n_tenor, nk))
    for j in range(n_tenor):
        tau = float(taus[j])
        for i in range(nk):
            w_ji = max(float(w_by_tenor[j, i]), _W_FLOOR)
            sigma = np.sqrt(w_ji / tau)
            c0[j, i] = black76_price(1.0, float(strikes[i]), tau, sigma, cp=1)
            vega = black76_vega(1.0, float(strikes[i]), tau, sigma)
            dc_dw[j, i] = vega / (2.0 * sigma * tau)
    return c0, dc_dw


def repair_surface(
    k_grid: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    *,
    weights: FloatArray | None = None,
    solver: str | None = None,
    max_iter: int = 10,
    move_tol: float = 1e-10,
) -> FloatArray:
    """Return the nearest arbitrage-free total-variance grid to ``w_by_tenor``.

    Args:
        k_grid: log-moneyness grid, shape (nk,), strictly increasing.
        w_by_tenor: total variance, shape (n_tenor, nk).
        taus: tenor times in years, shape (n_tenor,), strictly increasing.
        weights: optional per-cell weights (e.g. vega), shape (n_tenor, nk).
        solver: cvxpy solver name; defaults to OSQP.
        max_iter: max sequential-QP relinearization steps.
        move_tol: stop early when the iterate moves less than this (max abs).
    """
    k_grid = np.asarray(k_grid, dtype=float)
    w_by_tenor = np.asarray(w_by_tenor, dtype=float)
    taus = np.asarray(taus, dtype=float)
    n_tenor, nk = w_by_tenor.shape
    if k_grid.size != nk:
        raise ValueError("k_grid length must match w_by_tenor columns")
    if taus.size != n_tenor:
        raise ValueError("taus length must match w_by_tenor rows")
    if nk < _MIN_MONEYNESS_POINTS:
        raise ValueError("need at least 3 moneyness points for convexity")
    if not np.all(np.diff(k_grid) > 0.0):
        raise ValueError("k_grid must be strictly increasing")
    if np.any(taus <= 0.0):
        raise ValueError("taus must be positive")
    if taus.size > 1 and not np.all(np.diff(taus) > 0.0):
        raise ValueError("taus must be strictly increasing")

    weight_grid = np.ones_like(w_by_tenor) if weights is None else np.asarray(weights, dtype=float)
    if weight_grid.shape != w_by_tenor.shape:
        raise ValueError("weights shape must match w_by_tenor")
    sqrt_w = np.sqrt(np.maximum(weight_grid, 0.0))
    strikes = np.exp(k_grid)  # forward normalized to 1
    dK = np.diff(strikes)  # strike gaps for slope/convexity operators

    current = w_by_tenor.copy()
    for _ in range(max_iter):
        c0, dc_dw = _call_price_linearization(strikes, current, taus)

        x = cp.Variable((n_tenor, nk))
        objective = cp.Minimize(cp.sum_squares(cp.multiply(sqrt_w, x - w_by_tenor)))

        constraints: list = [x >= 0]
        # Calendar: total variance non-decreasing in tenor at each k.
        if n_tenor >= _MIN_TENORS_FOR_CALENDAR:
            constraints.append(x[1:, :] - x[:-1, :] >= 0)
        # Butterfly + vertical spread on linearized call prices, per tenor.
        # C_lin = c0 + dc_dw * (x - current); slopes via divided differences.
        c_lin = c0 + cp.multiply(dc_dw, x - current)
        slopes = (c_lin[:, 1:] - c_lin[:, :-1]) / dK[None, :]
        # Convexity: forward slope >= backward slope at each interior strike.
        constraints.append(slopes[:, 1:] - slopes[:, :-1] >= 0)
        # Call-spread bounds: price non-increasing in strike, slope >= -1.
        constraints.append(slopes <= 0)
        constraints.append(slopes >= -1)

        problem = cp.Problem(objective, constraints)
        chosen = solver or cp.OSQP
        solve_kwargs = _OSQP_OPTS if chosen == cp.OSQP else {}
        problem.solve(solver=chosen, **solve_kwargs)
        if x.value is None:
            raise RuntimeError(f"repair QP did not solve (status={problem.status})")

        new = np.asarray(x.value, dtype=float)
        move = float(np.max(np.abs(new - current)))
        current = new
        # Converged if the iterate settled or the true checker is satisfied.
        if move < move_tol or check_surface(k_grid, current, taus).ok:
            break

    return current
