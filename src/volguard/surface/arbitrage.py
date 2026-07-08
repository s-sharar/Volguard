"""Static no-arbitrage checks for implied-volatility surfaces.

Three conditions are enforced across the project (plan section 11):

Butterfly (per expiry)
    Call price must be convex in strike ⇔ the risk-neutral density is
    non-negative. Checked two ways: (1) the model-agnostic discrete convexity of
    Black-76 call prices computed from a total-variance grid, and (2) Gatheral's
    g-function on fitted SVI slices (see ``surface/svi.py``).

Vertical spread (per expiry)
    Undiscounted call price must be non-increasing in strike with slope no
    steeper than -1 (a call spread must cost between 0 and its width).
    Convexity alone does not imply this: a convex but increasing call-price
    curve is still a free lunch.

Calendar (across expiries)
    Total implied variance ``w(k, tau)`` must be non-decreasing in ``tau`` at
    fixed log-moneyness ``k``. Otherwise a calendar spread is a free option.

The checker is used in four places: market-data QC, SVI fit validation, the
training penalty, and forecast evaluation/repair. It returns rich diagnostics
(counts + magnitudes) rather than a bare bool so the memo can report the
market's own violation base rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from volguard.curate.blackiv import black76_price

FloatArray = NDArray[np.float64]

# Tolerance for treating a tiny negative value as numerical noise, not a
# genuine arbitrage violation.
_EPS = 1e-10
_MIN_STRIKES_FOR_CONVEXITY = 3
_SURFACE_NDIM = 2


def _validate_k_grid(k_grid: FloatArray) -> None:
    """Reject strike grids that cannot define finite price slopes."""
    if not np.all(np.isfinite(k_grid)):
        raise ValueError("k_grid values must be finite")
    if np.unique(k_grid).size != k_grid.size:
        raise ValueError("k_grid values must be unique")


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    """Diagnostics from an arbitrage check.

    ``ok`` is True iff no violations beyond tolerance were found. The magnitude
    fields sum the size of violations so callers can rank "how bad" a surface is.
    """

    ok: bool
    butterfly_violations: int = 0
    vertical_violations: int = 0
    calendar_violations: int = 0
    negative_variance_violations: int = 0
    nonfinite_violations: int = 0
    max_butterfly_magnitude: float = 0.0
    max_vertical_magnitude: float = 0.0
    max_calendar_magnitude: float = 0.0
    max_negative_variance_magnitude: float = 0.0
    integrated_butterfly_magnitude: float = 0.0
    integrated_vertical_magnitude: float = 0.0
    integrated_calendar_magnitude: float = 0.0
    detail: dict[str, object] = field(default_factory=dict)


def check_butterfly_from_w(
    k_grid: FloatArray,
    w: FloatArray,
    tau: float,
    *,
    eps: float = _EPS,
) -> ArbitrageReport:
    """Butterfly + vertical-spread check on Black-76 call prices in strike.

    Given total variance ``w`` on a log-moneyness grid ``k_grid`` for one expiry
    (forward normalized to F=1 so K=exp(k)), reconstruct undiscounted call
    prices and test (1) discrete convexity via divided differences and (2) the
    call-spread slope bound ``-1 <= dC/dK <= 0``. Convexity alone misses
    vertical-spread arbitrage: a convex but increasing call-price curve means a
    call spread with negative cost and non-negative payoff.
    """
    k_grid = np.asarray(k_grid, dtype=float)
    w = np.asarray(w, dtype=float)
    if k_grid.shape != w.shape:
        raise ValueError("k_grid and w must have the same shape")
    if k_grid.size < _MIN_STRIKES_FOR_CONVEXITY:
        raise ValueError("need at least 3 strikes for a convexity check")
    _validate_k_grid(k_grid)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")

    order = np.argsort(k_grid)
    k_sorted = k_grid[order]
    w_sorted = w[order]

    # NaN/inf must be caught before any comparison: ``NaN < -eps`` is False, so a
    # non-finite grid (missing data, a diverged forecast) would otherwise slip
    # through every violation test and be reported as arbitrage-free.
    n_nonfinite = int(np.count_nonzero(~np.isfinite(w_sorted)))
    if n_nonfinite > 0:
        return ArbitrageReport(ok=False, nonfinite_violations=n_nonfinite)

    # Total variance is non-negative by definition. Negative w is not merely a
    # numerical edge to clip away — it is an invalid/impossible surface (bad
    # market row or a model forecast gone out of range) and must be flagged, not
    # silently zeroed, or check_surface would report it as arbitrage-free.
    neg = w_sorted < -eps
    n_neg = int(np.count_nonzero(neg))
    max_neg = float(np.where(neg, -w_sorted, 0.0).max(initial=0.0))
    if n_neg > 0:
        return ArbitrageReport(
            ok=False,
            negative_variance_violations=n_neg,
            max_negative_variance_magnitude=max_neg,
        )

    iv = np.sqrt(np.maximum(w_sorted, 0.0) / tau)
    strikes = np.exp(k_sorted)  # F normalized to 1
    calls = np.array(
        [black76_price(1.0, K, tau, s, cp=1) for K, s in zip(strikes, iv, strict=True)]
    )

    # Convexity of C in strike via divided differences: strikes are exp(k) and
    # therefore non-uniformly spaced, so a plain second difference is wrong.
    # Convex ⇔ forward slope >= backward slope at each interior strike.
    dK = np.diff(strikes)
    slopes = np.diff(calls) / dK  # slope on each strike interval
    curvature = slopes[1:] - slopes[:-1]  # >= 0 where convex
    violations = curvature < -eps
    n_viol = int(np.count_nonzero(violations))
    mags = np.where(violations, -curvature, 0.0)

    # Vertical-spread bounds: undiscounted call-spread price must lie between 0
    # and the strike gap, i.e. slope in [-1, 0] on every interval.
    vert_violations = (slopes > eps) | (slopes < -1.0 - eps)
    n_vert = int(np.count_nonzero(vert_violations))
    vert_excess = np.maximum(slopes, 0.0) + np.maximum(-1.0 - slopes, 0.0)
    vert_mags = np.where(vert_violations, vert_excess, 0.0)

    return ArbitrageReport(
        ok=n_viol == 0 and n_vert == 0,
        butterfly_violations=n_viol,
        vertical_violations=n_vert,
        max_butterfly_magnitude=float(mags.max(initial=0.0)),
        max_vertical_magnitude=float(vert_mags.max(initial=0.0)),
        integrated_butterfly_magnitude=float(mags.sum()),
        integrated_vertical_magnitude=float(vert_mags.sum()),
    )


def check_calendar(
    k_grid: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    *,
    eps: float = _EPS,
) -> ArbitrageReport:
    """Calendar check: ``w(k, tau)`` non-decreasing in ``tau`` at fixed ``k``.

    Args:
        k_grid: shared log-moneyness grid, shape (nk,).
        w_by_tenor: total variance, shape (n_tenor, nk), rows ordered by tenor.
        taus: tenor times in years, shape (n_tenor,), strictly increasing.
    """
    k_grid = np.asarray(k_grid, dtype=float)
    w_by_tenor = np.asarray(w_by_tenor, dtype=float)
    taus = np.asarray(taus, dtype=float)
    if w_by_tenor.ndim != _SURFACE_NDIM:
        raise ValueError("w_by_tenor must be 2D (n_tenor, nk)")
    if k_grid.size != w_by_tenor.shape[1]:
        raise ValueError("k_grid length must match w_by_tenor columns")
    if w_by_tenor.shape[0] != taus.size:
        raise ValueError("w_by_tenor rows must match taus length")
    _validate_k_grid(k_grid)
    if not np.all(np.isfinite(taus)):
        raise ValueError("taus must be finite")
    if np.any(taus <= 0.0):
        raise ValueError("taus must be positive")
    if not np.all(np.diff(taus) > 0.0):
        raise ValueError("taus must be strictly increasing")

    n_nonfinite = int(np.count_nonzero(~np.isfinite(w_by_tenor)))
    if n_nonfinite > 0:
        return ArbitrageReport(
            ok=False, nonfinite_violations=n_nonfinite, detail={"k_grid_size": int(k_grid.size)}
        )

    # Difference between adjacent tenors; must be >= 0 everywhere.
    dw = np.diff(w_by_tenor, axis=0)  # shape (n_tenor-1, nk)
    violations = dw < -eps
    n_viol = int(np.count_nonzero(violations))
    mags = np.where(violations, -dw, 0.0)
    return ArbitrageReport(
        ok=n_viol == 0,
        calendar_violations=n_viol,
        max_calendar_magnitude=float(mags.max(initial=0.0)),
        integrated_calendar_magnitude=float(mags.sum()),
        detail={"k_grid_size": int(k_grid.size)},
    )


def check_surface(
    k_grid: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    *,
    eps: float = _EPS,
) -> ArbitrageReport:
    """Full surface check: butterfly + vertical per tenor, calendar across tenors."""
    k_grid = np.asarray(k_grid, dtype=float)
    w_by_tenor = np.asarray(w_by_tenor, dtype=float)
    taus = np.asarray(taus, dtype=float)

    bfly_viol = 0
    bfly_max = 0.0
    bfly_int = 0.0
    vert_viol = 0
    vert_max = 0.0
    vert_int = 0.0
    neg_viol = 0
    neg_max = 0.0
    nonfinite_viol = 0
    for row, tau in zip(w_by_tenor, taus, strict=True):
        rep = check_butterfly_from_w(k_grid, row, float(tau), eps=eps)
        bfly_viol += rep.butterfly_violations
        bfly_max = max(bfly_max, rep.max_butterfly_magnitude)
        bfly_int += rep.integrated_butterfly_magnitude
        vert_viol += rep.vertical_violations
        vert_max = max(vert_max, rep.max_vertical_magnitude)
        vert_int += rep.integrated_vertical_magnitude
        neg_viol += rep.negative_variance_violations
        neg_max = max(neg_max, rep.max_negative_variance_magnitude)
        nonfinite_viol += rep.nonfinite_violations

    cal = check_calendar(k_grid, w_by_tenor, taus, eps=eps)
    return ArbitrageReport(
        ok=(bfly_viol == 0 and vert_viol == 0 and neg_viol == 0 and nonfinite_viol == 0 and cal.ok),
        butterfly_violations=bfly_viol,
        vertical_violations=vert_viol,
        calendar_violations=cal.calendar_violations,
        negative_variance_violations=neg_viol,
        nonfinite_violations=nonfinite_viol,
        max_butterfly_magnitude=bfly_max,
        max_vertical_magnitude=vert_max,
        max_calendar_magnitude=cal.max_calendar_magnitude,
        max_negative_variance_magnitude=neg_max,
        integrated_butterfly_magnitude=bfly_int,
        integrated_vertical_magnitude=vert_int,
        integrated_calendar_magnitude=cal.integrated_calendar_magnitude,
    )
