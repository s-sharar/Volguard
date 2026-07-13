"""Calendar-ordered multi-expiry surface fit (M4).

Fit every expiry's smile in increasing ``tau``, choosing per slice between a
free raw-SVI fit (M1 ``fit_svi_slice``) and the global SSVI fallback for sparse
tenors, escalating the butterfly penalty when Gatheral's g-function goes
negative, and penalizing calendar-spread crossings against the previous
(shorter) tenor on a shared fixed-k grid.

Design references:
- Component 2 "surface/calendar_fit.py" and the ``fit_surface`` /
  ``fit_with_butterfly_refit`` pseudocode (pre/postconditions + loop
  invariants).
- Requirements 2 (vega-weighted SVI fit), 2b (butterfly reject-refit),
  3.2/3.5 (SSVI selection + ``fit_method``), 4 (calendar ordering on the shared
  ``arb_check_*`` grid).

Calendar-ordering rule (Requirement 4.4): a slice's ``calendar_ok`` is True iff
``w_j(k) >= w_{j-1}(k) - EPS`` holds on the shared fixed-k grid defined by
``cfg.arb_check_k_min``, ``cfg.arb_check_k_max``, and ``cfg.arb_check_points``.

The M1 math core (``surface/svi.py``, ``surface/fit.py``) and the SSVI fallback
(``surface/ssvi.py``) are reused, never reimplemented. For the calendar-penalty
refit this module needs a fitting hook that ``fit_svi_slice`` does not expose,
so it runs a small self-contained ``least_squares`` fit here, reusing
``fit.py``'s domain reparameterization (``_unpack``) and starting-point
heuristic (``_initial_guess``) so the optimizer stays in the valid SVI region
exactly as the M1 fitter does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
from scipy.optimize import least_squares

from volguard.config import SurfaceConfig

# NB: _initial_guess / _unpack are private to fit.py but reused (not
# reimplemented) here so the calendar-penalty refit stays in the valid SVI
# region exactly as fit_svi_slice does. fit.py itself is never modified.
from volguard.surface.fit import _initial_guess, _unpack, fit_svi_slice
from volguard.surface.ssvi import SSVIParams, fit_ssvi, ssvi_slice_to_svi
from volguard.surface.svi import SVIParams, svi_g, svi_total_variance
from volguard.surface.types import ExpiryObs, FloatArray

# Tolerance for the calendar-ordering comparison ``w_j(k) >= w_{j-1}(k) - EPS``.
EPS = 1e-10


class FitMethod(StrEnum):
    """How a slice was fit: a free raw-SVI fit or the global SSVI fallback."""

    SVI = "svi"
    SSVI = "ssvi"


@dataclass(frozen=True, slots=True)
class SliceFit:
    """One fitted expiry slice with its method and diagnostics."""

    expiry: datetime
    tau: float
    params: SVIParams
    method: FitMethod
    rmse: float
    n_obs: int
    vega_sum: float
    butterfly_ok: bool
    calendar_ok: bool
    refit_attempts: int


@dataclass(frozen=True, slots=True)
class SurfaceFit:
    """A fitted surface: per-tenor slices ordered by increasing ``tau``."""

    slices: list[SliceFit]  # ordered by increasing tau
    k_grid: FloatArray  # shared fixed-k grid used for calendar checks


@dataclass(frozen=True, slots=True)
class FitAttempt:
    """Result of a (possibly refit) single-slice fit."""

    params: SVIParams
    rmse: float
    vega_sum: float
    butterfly_ok: bool
    attempts: int


def _vega_weights(vega: FloatArray) -> FloatArray:
    """Vega weights, mirroring ``fit.py`` (``sqrt(max(vega, 1e-8))``).

    Kept identical to the M1 fitter so ``vega_sum`` and the weighting are
    consistent across ``svi`` and ``ssvi`` slices.
    """
    return np.sqrt(np.maximum(np.asarray(vega, dtype=float), 1e-8))


def _iv_rmse(params: SVIParams, o: ExpiryObs) -> float:
    """RMSE in implied-vol points of ``params`` against the slice observations."""
    w_model = svi_total_variance(params, o.k)
    iv_model = np.sqrt(np.maximum(w_model, 0.0) / o.tau)
    return float(np.sqrt(np.mean((iv_model - o.iv) ** 2)))


def _butterfly_ok_on_grid(params: SVIParams, k_grid: FloatArray) -> bool:
    """Certify butterfly-freeness via ``min_k svi_g(params, k) >= 0`` on the grid."""
    return bool(np.min(svi_g(params, k_grid)) >= 0.0)


def fit_with_butterfly_refit(o: ExpiryObs, cfg: SurfaceConfig) -> FitAttempt:
    """Fit a raw-SVI slice, escalating the butterfly penalty on g-negative fits.

    Follows the design pseudocode: start at ``cfg.butterfly_penalty``; while the
    best fit is not ``butterfly_ok`` and fewer than ``cfg.max_refit_attempts``
    fits have run, multiply the penalty by ``cfg.butterfly_penalty_escalation``
    (strictly increasing, so the loop terminates) and refit; keep a candidate
    when it is arb-free or lowers the RMSE; stop as soon as the best fit is
    butterfly-free (Requirements 2b.1-2b.5).

    Preconditions: ``o.n_obs >= cfg.min_obs_svi``; ``cfg.max_refit_attempts >= 1``;
    ``cfg.butterfly_penalty_escalation > 1``.
    """
    penalty = cfg.butterfly_penalty
    best = fit_svi_slice(o.k, o.iv, o.tau, vega=o.vega, butterfly_penalty=penalty)
    attempts = 1
    while not best.butterfly_ok and attempts < cfg.max_refit_attempts:
        penalty *= cfg.butterfly_penalty_escalation  # strictly increasing -> terminates
        candidate = fit_svi_slice(o.k, o.iv, o.tau, vega=o.vega, butterfly_penalty=penalty)
        attempts += 1
        # Keep the arb-free candidate if found; else keep the lower-RMSE one.
        if candidate.butterfly_ok or candidate.rmse < best.rmse:
            best = candidate
        if best.butterfly_ok:
            break
    return FitAttempt(best.params, best.rmse, best.vega_sum, best.butterfly_ok, attempts)


def fit_with_calendar_penalty(
    o: ExpiryObs, prev_w: FloatArray, k_grid: FloatArray, cfg: SurfaceConfig
) -> FitAttempt:
    """Refit a raw-SVI slice with a calendar penalty lifting it toward ``prev_w``.

    Minimizes the vega-weighted total-variance residual on the observations plus
    ``cfg.calendar_penalty * relu(prev_w - w_model)`` on the shared k-grid (so
    the slice is pushed up wherever it dips below the previous tenor) plus the
    same soft butterfly penalty ``fit.py`` uses on the k-grid. The optimizer runs
    in ``fit.py``'s domain reparameterization (``_unpack``) from its
    ``_initial_guess`` so the fitted params stay in the valid SVI region by
    construction (Requirement 4.3).
    """
    k = np.asarray(o.k, dtype=float)
    w_obs = o.iv * o.iv * o.tau
    weights = _vega_weights(o.vega)
    prev = np.asarray(prev_w, dtype=float)

    def residuals(theta: np.ndarray) -> np.ndarray:
        params = _unpack(theta)
        fit_res = weights * (svi_total_variance(params, k) - w_obs)
        w_grid = svi_total_variance(params, k_grid)
        calendar_pen = cfg.calendar_penalty * np.maximum(prev - w_grid, 0.0)
        butterfly_pen = cfg.butterfly_penalty * np.minimum(svi_g(params, k_grid), 0.0)
        return np.concatenate([fit_res, calendar_pen, butterfly_pen])

    theta0 = _initial_guess(k, w_obs)
    sol = least_squares(residuals, theta0, method="trf", max_nfev=2000)
    params = _unpack(sol.x)
    return FitAttempt(
        params=params,
        rmse=_iv_rmse(params, o),
        vega_sum=float(np.sum(weights**2)),
        butterfly_ok=_butterfly_ok_on_grid(params, k_grid),
        attempts=1,
    )


def _fit_slice(
    o: ExpiryObs, cfg: SurfaceConfig, ssvi_provider: _SSVIProvider, k_grid: FloatArray
) -> tuple[FitAttempt, FitMethod]:
    """Fit one expiry, selecting SSVI when too sparse for free raw-SVI.

    ``method == SSVI`` iff the slice has fewer than ``cfg.min_obs_svi`` *unique*
    log-moneyness points (Requirements 3.2/3.5). The five-parameter raw-SVI fit
    is determined by distinct strikes, not raw row count: duplicate ``(expiry, k)``
    rows (e.g. a call and a put at the same strike, or repeated trades) inflate
    ``o.n_obs`` without adding shape information, so gating on unique ``k`` keeps
    an underdetermined smile from bypassing the SSVI fallback.
    """
    n_unique_k = int(np.unique(o.k).size)
    if n_unique_k < cfg.min_obs_svi:
        params = ssvi_slice_to_svi(ssvi_provider.get(), o.expiry, o.tau)
        attempt = FitAttempt(
            params=params,
            rmse=_iv_rmse(params, o),
            vega_sum=float(np.sum(_vega_weights(o.vega) ** 2)),
            butterfly_ok=_butterfly_ok_on_grid(params, k_grid),
            attempts=0,
        )
        return attempt, FitMethod.SSVI
    return fit_with_butterfly_refit(o, cfg), FitMethod.SVI


class _SSVIProvider:
    """Lazily fit the global SSVI surface once, only when a slice needs it.

    Fitting a global SSVI is unnecessary when every expiry supports a free
    raw-SVI fit, so it is deferred until the first sparse slice. Any fit failure
    is surfaced to the caller (a sparse slice cannot proceed without it).
    """

    def __init__(self, obs: list[ExpiryObs], cfg: SurfaceConfig) -> None:
        self._obs = obs
        self._cfg = cfg
        self._fitted: SSVIParams | None = None

    def get(self) -> SSVIParams:
        if self._fitted is None:
            self._fitted = fit_ssvi(self._obs, self._cfg)
        return self._fitted


def fit_surface(obs: list[ExpiryObs], cfg: SurfaceConfig) -> SurfaceFit:
    """Fit all expiries in increasing ``tau`` with calendar ordering enforced.

    Preconditions: ``len(obs) >= cfg.min_expiries_per_snap``; each ``o.tau > 0``
    with ``o.n_obs >= cfg.min_obs_slice`` finite observations.

    Postconditions: one ``SliceFit`` per expiry, ordered by increasing ``tau``;
    every ``params`` is a valid ``SVIParams``; ``method == SSVI`` iff
    ``o.n_obs < cfg.min_obs_svi``; ``calendar_ok`` is honest, i.e. True iff
    ``w_j(k) >= w_{j-1}(k) - EPS`` on the shared k-grid after fitting.
    """
    if len(obs) < cfg.min_expiries_per_snap:
        raise ValueError(
            f"fit_surface needs >= {cfg.min_expiries_per_snap} expiries, got {len(obs)}"
        )

    k_grid = np.linspace(cfg.arb_check_k_min, cfg.arb_check_k_max, cfg.arb_check_points)
    ssvi_provider = _SSVIProvider(obs, cfg)

    slices: list[SliceFit] = []
    prev_w: FloatArray | None = None  # previous (shorter-tenor) slice's w on k_grid
    for o in sorted(obs, key=lambda x: x.tau):  # increasing tau
        # Loop invariant: prev_w holds the previous slice's total variance on
        # k_grid (or None for the first slice); tau strictly increases.
        attempt, method = _fit_slice(o, cfg, ssvi_provider, k_grid)
        params = attempt.params

        w_curr = svi_total_variance(params, k_grid)
        calendar_ok = prev_w is None or bool(np.all(w_curr >= prev_w - EPS))
        if not calendar_ok and method == FitMethod.SVI and prev_w is not None:
            # Soft calendar enforcement per the project plan ("penalize
            # w_j(k) < w_{j-1}(k)"): refit the raw-SVI slice with a calendar
            # penalty that lifts it toward the previous tenor. This is
            # best-effort, not a hard guarantee — any residual crossing is
            # recorded honestly in ``calendar_ok`` and surfaces in ``arb_post`` /
            # QC. Strict no-arbitrage enforcement is the job of the shared
            # downstream repair QP (design coordinate-separation decision;
            # Scenario 5), which is applied uniformly to every model's output so
            # the "accuracy cost of arbitrage repair" comparison stays honest.
            # SSVI fallback slices are flagged only (no free per-slice refit).
            attempt = fit_with_calendar_penalty(o, prev_w, k_grid, cfg)
            params = attempt.params
            w_curr = svi_total_variance(params, k_grid)
            calendar_ok = bool(np.all(w_curr >= prev_w - EPS))

        slices.append(
            SliceFit(
                expiry=o.expiry,
                tau=o.tau,
                params=params,
                method=method,
                rmse=attempt.rmse,
                n_obs=o.n_obs,
                vega_sum=attempt.vega_sum,
                butterfly_ok=attempt.butterfly_ok,
                calendar_ok=calendar_ok,
                refit_attempts=attempt.attempts,
            )
        )
        prev_w = w_curr

    return SurfaceFit(slices=slices, k_grid=k_grid)
