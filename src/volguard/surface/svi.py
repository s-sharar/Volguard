"""Raw SVI (Stochastic Volatility Inspired) smile parameterization.

Gatheral's *raw* SVI expresses total implied variance as a function of
log-moneyness ``k = ln(K/F)``::

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

where total variance ``w = iv^2 * tau``. The five parameters are:

- ``a``     vertical level (min total variance floor-ish); w >= a + b*sigma*sqrt(1-rho^2)
- ``b >= 0`` overall slope of the wings (angle between asymptotes)
- ``rho``   in (-1, 1); skew / left-right tilt
- ``m``     horizontal shift of the smile's minimum
- ``sigma > 0`` smoothness of the vertex (ATM curvature)

This module holds the pure parameterization, its derivatives, conversion to
implied vol, and the local (per-smile) butterfly no-arbitrage condition via
Gatheral's g-function. Fitting lives in ``surface/fit.py``; cross-expiry
calendar checks live in ``surface/arbitrage.py``.

References: Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Numerical floor: treat total variance below this as zero to avoid div-by-zero.
_W_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class SVIParams:
    """Raw-SVI parameters for a single expiry slice.

    Validated on construction to the domain where SVI is well-defined:
    ``b >= 0``, ``|rho| < 1``, ``sigma > 0``, and the wing constraint
    ``a + b*sigma*sqrt(1-rho^2) >= 0`` guaranteeing ``w(k) >= 0`` everywhere.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        # Finiteness first: NaN slips past every ordered comparison below (all
        # False), and ``m`` is otherwise unchecked, so a non-finite param would
        # yield non-finite w / g and be misread as arbitrage-free downstream.
        if not all(np.isfinite(v) for v in (self.a, self.b, self.rho, self.m, self.sigma)):
            raise ValueError("SVI params must all be finite")
        if self.b < 0.0:
            raise ValueError("SVI requires b >= 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("SVI requires -1 < rho < 1")
        if self.sigma <= 0.0:
            raise ValueError("SVI requires sigma > 0")
        w_min = self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho * self.rho)
        if w_min < -_W_FLOOR:
            raise ValueError(f"SVI params imply negative total variance (w_min={w_min:.3e})")

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


def svi_total_variance(params: SVIParams, k: float | FloatArray) -> FloatArray:
    """Total implied variance ``w(k)`` for raw SVI."""
    a, b, rho, m, sigma = params.as_tuple()
    km = np.asarray(k, dtype=float) - m
    return a + b * (rho * km + np.sqrt(km * km + sigma * sigma))


def svi_implied_vol(params: SVIParams, k: float | FloatArray, tau: float) -> FloatArray:
    """Implied volatility ``sqrt(w(k)/tau)`` from an SVI slice."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    w = svi_total_variance(params, k)
    return np.sqrt(np.maximum(w, 0.0) / tau)


def svi_derivatives(params: SVIParams, k: float | FloatArray) -> tuple[FloatArray, FloatArray]:
    """First and second derivatives of ``w(k)`` w.r.t. ``k`` (analytic)."""
    _, b, rho, m, sigma = params.as_tuple()
    km = np.asarray(k, dtype=float) - m
    root = np.sqrt(km * km + sigma * sigma)
    w1 = b * (rho + km / root)
    w2 = b * sigma * sigma / (root * root * root)
    return w1, w2


def svi_g(params: SVIParams, k: float | FloatArray) -> FloatArray:
    """Gatheral's g-function; ``g(k) >= 0`` for all k ⇔ no butterfly arbitrage.

    g(k) = (1 - k w'/(2w))^2 - (w'/2)^2 (1/w + 1/4) + w''/2

    A negative g anywhere means the slice admits a butterfly (negative implied
    risk-neutral density). Used both to validate/repair fits and to report the
    market's own violation base rate.
    """
    w = svi_total_variance(params, k)
    w1, w2 = svi_derivatives(params, k)
    # Guard against division by zero at w -> 0 (deep-OTM numerical edge).
    w_safe = np.where(w > _W_FLOOR, w, _W_FLOOR)
    term1 = (1.0 - k * w1 / (2.0 * w_safe)) ** 2
    term2 = (w1 * w1 / 4.0) * (1.0 / w_safe + 0.25)
    return term1 - term2 + w2 / 2.0


def svi_has_butterfly_arbitrage(params: SVIParams, k_grid: FloatArray | None = None) -> bool:
    """True if the slice violates the butterfly condition on the sampled grid."""
    if k_grid is None:
        k_grid = np.linspace(-2.0, 2.0, 401)
    return bool(np.any(svi_g(params, k_grid) < 0.0))
