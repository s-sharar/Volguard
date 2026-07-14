"""Diebold-Mariano tests with HLN finite-sample correction and BH adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DMResult:
    """One Diebold-Mariano comparison on daily surface losses."""

    statistic: float
    p_value: float
    mean_loss_diff: float
    n: int
    lag: int
    hln_correction: bool


def _hln_factor(n: int, horizon: int = 1) -> float:
    """Harvey-Leybourne-Newbold finite-sample scale for DM."""
    if n <= 1:
        return 1.0
    # h=1 one-step: sqrt((T + 1 - 2h + h(h-1)/T) / T) = sqrt((T - 1) / T)
    h = horizon
    inside = (n + 1 - 2 * h + h * (h - 1) / n) / n
    return float(np.sqrt(max(inside, 0.0)))


def diebold_mariano(
    loss_model: FloatArray,
    loss_benchmark: FloatArray,
    *,
    lag: int = 0,
    hln_correction: bool = True,
    two_sided: bool = True,
) -> DMResult:
    """DM test on paired daily losses (model - benchmark).

    For the one-day horizon use ``lag=0``. Variance uses the sample variance of
    the loss differential; Newey-West is unused at lag 0.
    """
    _min_obs = 2
    model = np.asarray(loss_model, dtype=np.float64).ravel()
    bench = np.asarray(loss_benchmark, dtype=np.float64).ravel()
    if model.shape != bench.shape:
        raise ValueError("loss series must share a length")
    if model.size < _min_obs:
        raise ValueError("need at least two observations for DM")
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    if lag != 0:
        raise ValueError("M6 one-day DM uses lag=0 only")

    diff = model - bench
    n = int(diff.size)
    mean = float(np.mean(diff))
    var = float(np.var(diff, ddof=1))
    if var <= 0.0:
        statistic = 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)
        p_value = 1.0 if mean == 0.0 else 0.0
        return DMResult(
            statistic=statistic,
            p_value=p_value,
            mean_loss_diff=mean,
            n=n,
            lag=lag,
            hln_correction=hln_correction,
        )

    se = float(np.sqrt(var / n))
    dm = mean / se
    if hln_correction:
        dm = dm * _hln_factor(n, horizon=1)
    df = n - 1
    p_value = float(2.0 * stats.t.sf(abs(dm), df=df)) if two_sided else float(stats.t.sf(dm, df=df))
    return DMResult(
        statistic=float(dm),
        p_value=p_value,
        mean_loss_diff=mean,
        n=n,
        lag=lag,
        hln_correction=hln_correction,
    )


def benjamini_hochberg(p_values: FloatArray, *, alpha: float = 0.05) -> FloatArray:
    """BH-adjusted p-values (positive FDR control), same length as input."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        return p.copy()
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if np.any((p < 0.0) | (p > 1.0) | ~np.isfinite(p)):
        raise ValueError("p_values must be finite and in [0, 1]")

    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    adjusted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = min(prev, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out
