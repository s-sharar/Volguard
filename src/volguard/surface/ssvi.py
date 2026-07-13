"""Global SSVI surface fit for the sparse-day fallback (M4).

When an expiry is too sparse to support five free raw-SVI parameters, M4 falls
back to a global Gatheral-Jacquier SSVI (Surface SVI) fit. SSVI ties every
smile together through the ATM total variance ``theta(tau)`` and two global
shape scalars, so it stays stable with very few points per slice::

    w(k, theta) = (theta / 2) * ( 1 + rho * phi * k
                                  + sqrt( (phi * k + rho)^2 + (1 - rho^2) ) )

with the classic power-law curvature ``phi(theta) = eta * theta^(-gamma)``.

Each SSVI slice is converted back into an equivalent raw ``SVIParams`` so the
rest of M4 (grid sampling, arbitrage checks) treats ``svi`` and ``ssvi`` slices
uniformly. The M1 math core (``surface/svi.py``) is reused, never
reimplemented, and no new dependency is introduced (``scipy.optimize`` is
already used by ``surface/fit.py``).

References: Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility
surfaces".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares

from volguard.config import SurfaceConfig
from volguard.surface.svi import SVIParams
from volguard.surface.types import ExpiryObs, FloatArray

# Keep the power-law exponent in the open interval (0, 1). The classic Gatheral
# arbitrage-free condition is gamma in (0, 0.5]; we stay in (0, 1) to remain
# safe and well-inside the domain (documented simplification; the strict no-arb
# bound theta*phi*(1+|rho|) <= 4 is not hard-enforced here).
_GAMMA_MIN = 1e-3
_GAMMA_MAX = 1.0 - 1e-3
_PHI_FLOOR = 1e-8


@dataclass(frozen=True, slots=True)
class SSVIParams:
    """Fitted SSVI surface: per-expiry ATM total variance plus global shape."""

    theta: dict[datetime, float]  # ATM total variance per expiry
    rho: float
    eta: float
    gamma: float


def _phi(theta: float, eta: float, gamma: float) -> float:
    """Power-law SSVI curvature ``phi(theta) = eta * theta^(-gamma)``."""
    return eta * float(theta) ** (-gamma)


def _ssvi_total_variance(k: FloatArray, theta: float, rho: float, phi: float) -> FloatArray:
    """SSVI total variance ``w(k, theta)`` on the surface parameterization."""
    pk = phi * k
    return (theta / 2.0) * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + (1.0 - rho * rho)))


def _atm_total_variance(k: FloatArray, iv: FloatArray, tau: float) -> float:
    """Estimate ATM total variance ``theta = iv(0)^2 * tau`` from the slice.

    Uses the total variance ``w = iv^2 * tau`` evaluated at ``k = 0``:
    interpolate linearly between the two points straddling ATM when no exact
    ATM quote exists, otherwise fall back to the nearest ``|k|`` observation.
    """
    w = iv * iv * tau
    # Collapse duplicate log-moneyness points (e.g. retained call/put rows at the
    # same strike) to a single mean total variance per unique k, so the interp
    # input is strictly increasing. Otherwise np.interp's result depends on which
    # duplicate sorts last, making the ATM theta (and the whole SSVI fit) depend
    # on row order rather than the available quotes.
    k_unique, inverse = np.unique(k, return_inverse=True)
    w_sums = np.zeros_like(k_unique)
    counts = np.zeros_like(k_unique)
    np.add.at(w_sums, inverse, w)
    np.add.at(counts, inverse, 1.0)
    w_mean = w_sums / counts
    # np.interp clamps to the endpoints outside the observed range, giving the
    # nearest point when all quotes sit on one side of ATM.
    return float(np.interp(0.0, k_unique, w_mean))


def _unpack_globals(theta_raw: np.ndarray) -> tuple[float, float, float]:
    """Map an unconstrained vector to valid ``(rho, eta, gamma)``.

    rho -> tanh (in (-1, 1)); eta -> softplus (> 0); gamma -> scaled sigmoid
    into (_GAMMA_MIN, _GAMMA_MAX).
    """
    rho_raw, eta_raw, gamma_raw = theta_raw
    rho = float(np.tanh(rho_raw))
    # numerically stable softplus
    eta = float(np.log1p(np.exp(-abs(eta_raw))) + max(eta_raw, 0.0) + 1e-6)
    sig = 1.0 / (1.0 + np.exp(-gamma_raw))
    gamma = float(_GAMMA_MIN + (_GAMMA_MAX - _GAMMA_MIN) * sig)
    return rho, eta, gamma


def fit_ssvi(obs: list[ExpiryObs], cfg: SurfaceConfig) -> SSVIParams:
    """Fit a global SSVI surface across all expiries' observations.

    Estimates ATM total variance ``theta_j`` per expiry, then fits the global
    scalars ``(rho, eta, gamma)`` by least squares over every observation of
    every expiry, minimizing ``sum (w_model(k, theta_j) - w_obs)^2`` where
    ``phi_j = eta * theta_j^(-gamma)``. The optimizer works in an unconstrained
    reparameterization so ``(rho, eta, gamma)`` always stays in-domain.
    """
    if not obs:
        raise ValueError("fit_ssvi requires at least one expiry")

    theta: dict[datetime, float] = {}
    for o in obs:
        theta_j = _atm_total_variance(o.k, o.iv, o.tau)
        # Guard: ATM total variance must be strictly positive for phi and the
        # SSVI->SVI mapping to stay in the valid domain.
        theta[o.expiry] = max(theta_j, 1e-8)

    k_all = np.concatenate([o.k for o in obs])
    w_all = np.concatenate([o.iv * o.iv * o.tau for o in obs])
    # Repeat each slice's theta across its observations for a vectorized model.
    theta_all = np.concatenate([np.full(o.k.shape, theta[o.expiry]) for o in obs])

    def residuals(x: np.ndarray) -> np.ndarray:
        rho, eta, gamma = _unpack_globals(x)
        phi_all = eta * theta_all ** (-gamma)
        pk = phi_all * k_all
        w_model = (theta_all / 2.0) * (
            1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + (1.0 - rho * rho))
        )
        return w_model - w_all

    # Initial guess: mild negative skew, moderate curvature, mid-range exponent.
    x0 = np.array([np.arctanh(-0.3), np.log(np.expm1(1.0)), 0.0])
    sol = least_squares(residuals, x0, method="trf", max_nfev=2000)
    rho, eta, gamma = _unpack_globals(sol.x)
    # ``cfg`` is accepted for interface uniformity with the other fit stages;
    # the current SSVI fit does not read tunable knobs from it.
    _ = cfg
    return SSVIParams(theta=theta, rho=rho, eta=eta, gamma=gamma)


def ssvi_slice_to_svi(ssvi: SSVIParams, expiry: datetime, tau: float) -> SVIParams:
    """Convert one SSVI slice into an equivalent valid raw ``SVIParams``.

    Matching the SSVI slice ``w(k, theta)`` to raw SVI
    ``w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2))`` yields the standard
    closed-form identities (with ``phi = eta * theta^(-gamma)``):

        b       = theta * phi / 2
        sigma   = sqrt(1 - rho^2) / phi
        rho_svi = rho
        m       = -rho / phi
        a       = (theta / 2) * (1 - rho^2)

    These satisfy the SVI validity domain by construction for ``theta > 0``,
    ``|rho| < 1``, ``phi > 0`` (wing bound ``a + b*sigma*sqrt(1-rho^2) =
    theta*(1-rho^2) >= 0``) and reproduce ``w(0) == theta`` exactly.
    """
    theta = ssvi.theta[expiry]
    if theta <= 0.0:
        raise ValueError(f"SSVI theta must be positive (expiry={expiry!r}, theta={theta})")
    rho = ssvi.rho
    phi = _phi(theta, ssvi.eta, ssvi.gamma)
    if phi <= _PHI_FLOOR:
        raise ValueError(f"SSVI phi must be positive (phi={phi:.3e})")

    one_minus_rho2 = 1.0 - rho * rho
    b = theta * phi / 2.0
    sigma = np.sqrt(one_minus_rho2) / phi
    m = -rho / phi
    a = (theta / 2.0) * one_minus_rho2
    # ``tau`` is part of the documented interface (the caller identifies the
    # slice by (expiry, tau)); the SSVI->SVI mapping is expressed purely in
    # total-variance space and does not need it.
    _ = tau
    return SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma))
