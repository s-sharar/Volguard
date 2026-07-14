"""B2 per-tenor SVI-parameter AR(1) baseline with validity clipping / B0 fallback."""

from __future__ import annotations

from datetime import date

import numpy as np
from numpy.typing import NDArray

from volguard.config import EvalConfig
from volguard.models.baselines.common import fit_ar1
from volguard.models.fold_runner import FoldContext, HyperParams
from volguard.models.types import FittedBaseline
from volguard.surface.fit import _RHO_BOUND, _unpack, fit_svi_slice
from volguard.surface.svi import SVIParams, svi_total_variance

FloatArray = NDArray[np.float64]
_N_SVI = 5
_SOFTPLUS_EPS = 1e-12
_MIN_AR1_POINTS = 2


def _softplus_inv(value: float) -> float:
    y = max(float(value), _SOFTPLUS_EPS)
    return float(np.log(np.expm1(y)))


def _pack(params: SVIParams) -> FloatArray:
    """Map valid SVI params to the unconstrained fit.py / `_unpack` coordinates."""
    wing = params.b * params.sigma * float(np.sqrt(1.0 - params.rho * params.rho))
    slack = params.a + wing
    a_raw = _softplus_inv(slack)
    b_raw = _softplus_inv(params.b)
    sigma_raw = _softplus_inv(max(params.sigma - 1e-6, _SOFTPLUS_EPS))
    rho_raw = float(np.arctanh(np.clip(params.rho, -_RHO_BOUND, _RHO_BOUND)))
    return np.array([a_raw, b_raw, rho_raw, params.m, sigma_raw], dtype=np.float64)


def _fit_slice_from_grid(
    k_row: FloatArray, w_row: FloatArray, tau: float
) -> tuple[SVIParams | None, bool]:
    """Fit one tenor SVI slice from a native grid row; return (params, ok)."""
    w = np.maximum(np.asarray(w_row, dtype=np.float64), 0.0)
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(k_row)):
        return None, False
    iv = np.sqrt(np.maximum(w / tau, 0.0))
    try:
        result = fit_svi_slice(np.asarray(k_row, dtype=np.float64), iv, float(tau))
    except (ValueError, RuntimeError):
        return None, False
    if not result.success:
        return None, False
    return result.params, True


def _clip_params(params: SVIParams) -> tuple[SVIParams, bool]:
    """Clip to the valid raw-SVI domain; return (params, was_clipped)."""
    clipped = False
    b = max(params.b, 0.0)
    sigma = max(params.sigma, 1e-6)
    rho = float(np.clip(params.rho, -_RHO_BOUND, _RHO_BOUND))
    wing = b * sigma * float(np.sqrt(1.0 - rho * rho))
    a = params.a
    if a + wing < 0.0:
        a = -wing
        clipped = True
    if b != params.b or sigma != params.sigma or rho != params.rho:
        clipped = True
    try:
        return SVIParams(a=a, b=b, rho=rho, m=params.m, sigma=sigma), clipped
    except ValueError:
        # Last-resort ATM flat smile.
        return SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1), True


class SVIARBaseline:
    """Canonical-tenor SVI fits → unconstrained AR(1) → reconstruct w grid."""

    model_id = "b2"
    last_diagnostics: dict[str, int]

    def __init__(self) -> None:
        self.last_diagnostics = {"clip_count": 0, "fallback_tenors": 0}

    def fit(
        self,
        ctx: FoldContext,
        *,
        cfg: EvalConfig,
        tune: bool,
        frozen_hyperparameters: HyperParams | None,
    ) -> FittedBaseline:
        del cfg, tune, frozen_hyperparameters
        n_tenor = ctx.grid_w.shape[1]
        transformed: list[list[FloatArray]] = [[] for _ in range(n_tenor)]
        fit_failures = 0
        for index in ctx.train_indices:
            for j in range(n_tenor):
                params, ok = _fit_slice_from_grid(
                    ctx.grid_k[index, j], ctx.grid_w[index, j], float(ctx.taus[j])
                )
                if not ok or params is None:
                    fit_failures += 1
                    continue
                transformed[j].append(_pack(params))

        coeffs: list[list[tuple[float, float]] | None] = []
        for series_list in transformed:
            if len(series_list) < _MIN_AR1_POINTS:
                coeffs.append(None)
                continue
            series = np.asarray(series_list, dtype=np.float64)
            param_coeffs = [fit_ar1(series[:, p]) for p in range(_N_SVI)]
            coeffs.append(param_coeffs)

        return FittedBaseline(
            model_id=self.model_id,
            fold_id=ctx.fold.fold_id,
            train_start=ctx.fold.train_start,
            train_end=ctx.fold.train_end,
            hyperparameters={},
            state={
                "ar1_coeffs": coeffs,
                "train_fit_failures": fit_failures,
            },
        )

    def predict_next(
        self,
        fitted: FittedBaseline,
        *,
        ctx: FoldContext,
        history_end: int,
        issue_date: date,
        target_date: date,
        cfg: EvalConfig,
    ) -> FloatArray:
        del target_date, cfg
        if history_end <= 0:
            raise ValueError("SVI-AR requires at least one realized surface")
        if ctx.dates[history_end - 1] != issue_date:
            raise ValueError("history must end at the issue date")

        coeffs = fitted.state["ar1_coeffs"]
        n_tenor = ctx.grid_w.shape[1]
        nk = ctx.grid_w.shape[2]
        out = np.zeros((n_tenor, nk), dtype=np.float64)
        clip_count = 0
        fallback_tenors = 0
        issue_w = ctx.grid_w[history_end - 1]

        for j in range(n_tenor):
            tenor_coeffs = coeffs[j]
            params, ok = _fit_slice_from_grid(
                ctx.grid_k[history_end - 1, j],
                issue_w[j],
                float(ctx.taus[j]),
            )
            if not ok or params is None or tenor_coeffs is None:
                out[j] = issue_w[j]
                fallback_tenors += 1
                continue
            theta = _pack(params)
            next_theta = np.zeros(_N_SVI, dtype=np.float64)
            for p, (c, phi) in enumerate(tenor_coeffs):
                next_theta[p] = c + phi * theta[p]
            try:
                next_params = _unpack(next_theta)
            except ValueError:
                next_params, was_clipped = _clip_params(
                    SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.2)
                )
                clip_count += int(was_clipped)
                out[j] = issue_w[j]
                fallback_tenors += 1
                continue
            next_params, was_clipped = _clip_params(next_params)
            clip_count += int(was_clipped)
            out[j] = np.asarray(
                svi_total_variance(next_params, ctx.grid_k[history_end - 1, j]),
                dtype=np.float64,
            )

        self.last_diagnostics = {
            "clip_count": clip_count,
            "fallback_tenors": fallback_tenors,
        }
        return out
