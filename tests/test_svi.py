"""Unit + property tests for the SVI parameterization and fitting."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volguard.surface.fit import _minimum_svi_g, fit_svi_slice
from volguard.surface.svi import (
    SVIParams,
    svi_derivatives,
    svi_g,
    svi_implied_vol,
    svi_total_variance,
)

# Strategy over valid, well-behaved SVI params.
svi_strategy = st.builds(
    SVIParams,
    a=st.floats(min_value=0.01, max_value=0.5),
    b=st.floats(min_value=0.05, max_value=0.8),
    rho=st.floats(min_value=-0.8, max_value=0.8),
    m=st.floats(min_value=-0.3, max_value=0.3),
    sigma=st.floats(min_value=0.05, max_value=0.5),
)


def test_params_reject_invalid() -> None:
    with pytest.raises(ValueError):
        SVIParams(a=0.1, b=-1.0, rho=0.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError):
        SVIParams(a=0.1, b=0.1, rho=1.5, m=0.0, sigma=0.1)
    with pytest.raises(ValueError):
        SVIParams(a=0.1, b=0.1, rho=0.0, m=0.0, sigma=-0.1)


def test_params_reject_nonfinite() -> None:
    # Codex P2 regression: NaN/inf slips past ordered comparisons, and m was
    # never bounds-checked, so every field needs an explicit finite check.
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite"):
            SVIParams(a=bad, b=0.1, rho=0.0, m=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="finite"):
            SVIParams(a=0.04, b=0.1, rho=0.0, m=bad, sigma=0.1)
        with pytest.raises(ValueError, match="finite"):
            SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=bad)


@settings(max_examples=200, deadline=None)
@given(params=svi_strategy)
def test_total_variance_nonnegative(params: SVIParams) -> None:
    k = np.linspace(-2.0, 2.0, 201)
    w = svi_total_variance(params, k)
    assert np.all(w >= -1e-9)


@settings(max_examples=100, deadline=None)
@given(params=svi_strategy)
def test_analytic_derivatives_match_finite_difference(params: SVIParams) -> None:
    k = np.linspace(-1.5, 1.5, 50)
    h = 1e-5
    w1, w2 = svi_derivatives(params, k)
    fd1 = (svi_total_variance(params, k + h) - svi_total_variance(params, k - h)) / (2 * h)
    fd2 = (
        svi_total_variance(params, k + h)
        - 2 * svi_total_variance(params, k)
        + svi_total_variance(params, k - h)
    ) / (h * h)
    assert np.allclose(w1, fd1, rtol=1e-4, atol=1e-6)
    assert np.allclose(w2, fd2, rtol=1e-3, atol=1e-4)


def test_implied_vol_positive() -> None:
    params = SVIParams(a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.1)
    iv = svi_implied_vol(params, np.linspace(-1, 1, 21), tau=0.5)
    assert np.all(iv > 0.0)


def test_g_function_positive_for_calm_smile() -> None:
    # A gently-sloped smile should be butterfly-arb-free.
    params = SVIParams(a=0.04, b=0.1, rho=-0.1, m=0.0, sigma=0.2)
    g = svi_g(params, np.linspace(-1.5, 1.5, 301))
    assert np.all(g >= 0.0)


def test_fit_recovers_synthetic_params() -> None:
    """Fitting IVs generated from known SVI params recovers the smile shape."""
    true = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.05, sigma=0.15)
    tau = 0.5
    k = np.linspace(-1.0, 1.0, 25)
    iv = svi_implied_vol(true, k, tau)

    result = fit_svi_slice(k, np.asarray(iv), tau, butterfly_penalty=0.0)
    assert result.success
    # Recovered smile should match on total variance, even if params trade off.
    w_true = svi_total_variance(true, k)
    w_fit = svi_total_variance(result.params, k)
    assert np.allclose(w_true, w_fit, atol=1e-3)
    assert result.rmse < 1e-2


def test_fit_requires_enough_points() -> None:
    with pytest.raises(ValueError):
        fit_svi_slice(np.array([0.0, 0.1]), np.array([0.5, 0.5]), 0.5)


def test_fit_rejects_nonfinite_tau() -> None:
    k = np.linspace(-1.0, 1.0, 5)
    with pytest.raises(ValueError, match="finite"):
        fit_svi_slice(k, np.full(5, 0.5), np.nan)


def test_butterfly_ok_detects_wing_arbitrage_outside_observed_range() -> None:
    # Codex P2 regression: this smile is butterfly-clean on the observed [-2, 2]
    # range but g(-3) < 0. Tying the arb grid to observed strikes would certify
    # it butterfly_ok=True while the stored continuous smile arbitrages in the
    # left wing. The fit's certification must inspect the wings too.
    arb_in_wing = SVIParams(a=0.0686, b=1.6679, rho=0.1458, m=-0.7368, sigma=0.7163)
    assert np.all(svi_g(arb_in_wing, np.linspace(-2.0, 2.0, 50)) >= 0.0)
    assert svi_g(arb_in_wing, np.array([-3.0]))[0] < 0.0

    tau = 0.5
    k = np.linspace(-2.0, 2.0, 25)
    iv = svi_implied_vol(arb_in_wing, k, tau)
    result = fit_svi_slice(k, np.asarray(iv), tau, butterfly_penalty=0.0)
    assert not result.butterfly_ok


def test_butterfly_ok_detects_narrow_vertex_pocket() -> None:
    # Codex P2 regression: a thin butterfly pocket (width scales with sigma) is
    # stepped over by a fixed-resolution grid. With sigma~4e-4 the pocket near
    # the vertex is ~0.035 wide, far below a 0.10 grid step, yet must be caught.
    narrow = SVIParams(
        a=0.0132510425, b=0.3377919308, rho=-0.6708407281, m=0.1470225412, sigma=0.0004035387
    )
    assert svi_g(narrow, np.array([0.1614]))[0] < 0.0

    tau = 0.5
    k = np.linspace(-2.0, 2.0, 25)
    iv = svi_implied_vol(narrow, k, tau)
    result = fit_svi_slice(k, np.asarray(iv), tau, butterfly_penalty=0.0)
    assert not result.butterfly_ok


def test_butterfly_ok_minimizes_between_grid_points() -> None:
    # A tiny-sigma fit can have a negative-g pocket outside the linearly sampled
    # vertex window. Final certification must minimize g between wing grid
    # points instead of trusting sampled values alone.
    narrow = SVIParams(
        a=0.2965837279,
        b=1.2320313814,
        rho=0.8910346796,
        m=-3.8971245281,
        sigma=1.3764e-05,
    )
    assert svi_g(narrow, np.array([-3.906]))[0] < 0.0

    tau = 0.5
    k = np.linspace(-5.0, -3.0, 25)
    assert _minimum_svi_g(narrow, k) < 0.0
    iv = svi_implied_vol(narrow, k, tau)
    result = fit_svi_slice(k, np.asarray(iv), tau, butterfly_penalty=0.0)
    assert not result.butterfly_ok


def test_butterfly_certification_includes_vertex_outside_wings() -> None:
    # The fit leaves m unconstrained, so the vertex can sit outside the traded
    # ±5 wing range. This smile is clean inside that range but has negative g
    # farther left; certification must retain the sigma-scaled vertex search.
    far_vertex = SVIParams(
        a=0.15090962065805608,
        b=0.8348716813027642,
        rho=-0.7662622215296833,
        m=-6.89752624281322,
        sigma=0.035703239065914466,
    )
    k = np.linspace(-1.0, 1.0, 25)
    assert np.all(svi_g(far_vertex, np.linspace(-5.0, 5.0, 401)) >= 0.0)
    assert _minimum_svi_g(far_vertex, k) < 0.0
