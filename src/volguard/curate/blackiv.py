"""Black-76 option pricing, implied-volatility solver, and greeks.

Deribit BTC options are European options on a *forward* (the dated future /
perp-implied forward), so Black-76 is the correct model (not Black-Scholes on
spot). Prices here are in the same numeraire as the forward and strike; the
caller handles Deribit's inverse-contract BTC quoting separately in curation.

Everything is a pure function over floats / numpy arrays — no I/O, no data
dependencies — so the module is trivially property-tested and reused as the
trusted core across curation, SVI fitting, arbitrage checks, and evaluation.

Conventions
-----------
- ``F``  forward price of the underlying for the option's expiry
- ``K``  strike
- ``tau`` time to expiry in years
- ``sigma`` Black-76 implied volatility (annualized)
- ``r``  risk-free discount rate (continuously compounded). Deribit options are
         effectively priced off the forward with r≈0 for the undiscounted
         premium; we keep ``r`` explicit and default it to 0.0.
- ``cp`` +1 for a call, -1 for a put.
"""

from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import ndtr

CallPut = Literal[-1, 1]

# Numerical floors to keep the math well-defined at the boundaries.
_MIN_TAU = 1e-12
_MIN_SIGMA = 1e-12
_SQRT_2 = math.sqrt(2.0)
_MAX_BRACKET_EXPANSIONS = 10


def _norm_cdf(x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Standard normal CDF: scalar fast path via ``math.erf``, ufunc for arrays.

    This sits in the hot path (butterfly checks and the repair linearization
    call the pricer in tight loops), so avoid ``np.vectorize`` — a Python-level
    loop — in favor of scipy's C-implemented ``ndtr`` for array inputs.
    """
    if isinstance(x, float | int):
        return 0.5 * (1.0 + math.erf(float(x) / _SQRT_2))
    return np.asarray(ndtr(np.asarray(x, dtype=float)), dtype=np.float64)


def _norm_pdf(x: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard normal PDF (vectorized)."""
    xa = np.asarray(x, dtype=float)
    return np.exp(-0.5 * xa * xa) / math.sqrt(2.0 * math.pi)


def _d1_d2(F: float, K: float, tau: float, sigma: float) -> tuple[float, float]:
    """Black-76 d1/d2. Assumes F, K, tau, sigma all strictly positive."""
    vol_sqrt_tau = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    return d1, d2


def black76_price(
    F: float, K: float, tau: float, sigma: float, cp: CallPut = 1, r: float = 0.0
) -> float:
    """Black-76 price of a European option on a forward.

    Returns the option premium in the forward's numeraire. Handles the
    degenerate ``tau -> 0`` and ``sigma -> 0`` cases by returning the discounted
    intrinsic value, which keeps callers (IV solver, tests) numerically safe.
    """
    if F <= 0.0 or K <= 0.0:
        raise ValueError("F and K must be positive")
    disc = math.exp(-r * tau)
    if tau <= _MIN_TAU or sigma <= _MIN_SIGMA:
        intrinsic = max(cp * (F - K), 0.0)
        return disc * intrinsic
    d1, d2 = _d1_d2(F, K, tau, sigma)
    nd1 = float(_norm_cdf(cp * d1))
    nd2 = float(_norm_cdf(cp * d2))
    return disc * cp * (F * nd1 - K * nd2)


def black76_vega(F: float, K: float, tau: float, sigma: float, r: float = 0.0) -> float:
    """Vega (∂price/∂sigma). Same for calls and puts. Returns 0 at boundaries."""
    if tau <= _MIN_TAU or sigma <= _MIN_SIGMA or F <= 0.0 or K <= 0.0:
        return 0.0
    disc = math.exp(-r * tau)
    d1, _ = _d1_d2(F, K, tau, sigma)
    return disc * F * float(_norm_pdf(d1)) * math.sqrt(tau)


def black76_greeks(
    F: float, K: float, tau: float, sigma: float, cp: CallPut = 1, r: float = 0.0
) -> dict[str, float]:
    """Return delta, gamma, vega, theta for a Black-76 option (w.r.t. forward).

    Delta is ∂price/∂F. Gamma is ∂²price/∂F². Theta is ∂price/∂t (per year,
    negative for long options in most regimes). Vega is per unit of sigma
    (i.e. per 1.00 = 100 vol points), the same convention as the pricer.
    """
    if tau <= _MIN_TAU or sigma <= _MIN_SIGMA:
        # At expiry greeks collapse; report a well-defined delta and zeros.
        delta = float(cp) if cp * (F - K) > 0.0 else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    disc = math.exp(-r * tau)
    d1, d2 = _d1_d2(F, K, tau, sigma)
    pdf_d1 = float(_norm_pdf(d1))
    sqrt_tau = math.sqrt(tau)

    delta = disc * cp * float(_norm_cdf(cp * d1))
    gamma = disc * pdf_d1 / (F * sigma * sqrt_tau)
    vega = disc * F * pdf_d1 * sqrt_tau
    theta = -disc * F * pdf_d1 * sigma / (2.0 * sqrt_tau) + r * disc * cp * (
        F * float(_norm_cdf(cp * d1)) - K * float(_norm_cdf(cp * d2))
    )
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(
    price: float,
    F: float,
    K: float,
    tau: float,
    cp: CallPut = 1,
    r: float = 0.0,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    sigma_hi: float = 5.0,
) -> float:
    """Invert Black-76 for implied volatility via bracketed Brent's method.

    Returns ``nan`` if the target price is outside the no-arbitrage bounds
    (below intrinsic or above the forward), which lets callers filter bad quotes
    rather than crash. Root-finding uses ``scipy.optimize.brentq`` (scipy is
    already a core dependency).
    """
    # Non-finite inputs (NaN price from a bad quote, inf F/K) must take the
    # documented "return nan" path: every ``<``/``>`` bound comparison below is
    # False for NaN, so without this guard a NaN price would fall through to the
    # bracket expansion and return a bogus finite sigma (e.g. sigma_hi).
    if not (
        math.isfinite(price)
        and math.isfinite(F)
        and math.isfinite(K)
        and math.isfinite(tau)
        and math.isfinite(r)
    ):
        return math.nan
    if F <= 0.0 or K <= 0.0 or tau <= _MIN_TAU:
        return math.nan
    disc = math.exp(-r * tau)
    intrinsic = disc * max(cp * (F - K), 0.0)
    upper = disc * (F if cp == 1 else K)  # call <= disc*F, put <= disc*K
    # Allow tiny numerical slack at the boundaries before declaring no-solution.
    if price < intrinsic - tol or price > upper + tol:
        return math.nan
    if price <= intrinsic + tol:
        return 0.0

    def f(sigma: float) -> float:
        return black76_price(F, K, tau, sigma, cp, r) - price

    lo, hi = _MIN_SIGMA, sigma_hi
    f_lo, f_hi = f(lo), f(hi)
    # Expand the upper bracket if the price implies vol above sigma_hi.
    expand = 0
    while f_lo * f_hi > 0.0 and expand < _MAX_BRACKET_EXPANSIONS:
        hi *= 2.0
        f_hi = f(hi)
        expand += 1
    if f_lo * f_hi > 0.0:
        return math.nan  # could not bracket a root

    return cast(float, brentq(f, lo, hi, xtol=tol, maxiter=max_iter))
