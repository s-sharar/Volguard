"""Unit + property tests for the Black-76 pricer, IV solver, and greeks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volguard.curate.blackiv import (
    CallPut,
    black76_greeks,
    black76_price,
    black76_vega,
    implied_vol,
)

# Reasonable option regime: forward, strike, expiry, vol.
forwards = st.floats(min_value=1_000.0, max_value=200_000.0)
strikes = st.floats(min_value=1_000.0, max_value=200_000.0)
taus = st.floats(min_value=1e-3, max_value=3.0)
sigmas = st.floats(min_value=0.05, max_value=3.0)
cps: st.SearchStrategy[CallPut] = st.sampled_from([1, -1])


def test_put_call_parity() -> None:
    F, K, tau, sigma = 50_000.0, 52_000.0, 0.25, 0.8
    call = black76_price(F, K, tau, sigma, cp=1)
    put = black76_price(F, K, tau, sigma, cp=-1)
    # Undiscounted (r=0): C - P = F - K.
    assert call - put == pytest.approx(F - K, abs=1e-6)


def test_intrinsic_at_expiry() -> None:
    F, K = 50_000.0, 48_000.0
    assert black76_price(F, K, 0.0, 0.5, cp=1) == pytest.approx(F - K)
    assert black76_price(F, K, 0.0, 0.5, cp=-1) == pytest.approx(0.0)


def test_price_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        black76_price(-1.0, 100.0, 0.5, 0.5)


@settings(max_examples=200, deadline=None)
@given(F=forwards, K=strikes, tau=taus, sigma=sigmas, cp=cps)
def test_iv_round_trip(F: float, K: float, tau: float, sigma: float, cp: CallPut) -> None:
    """Pricing at sigma then inverting recovers sigma for valid inputs."""
    price = black76_price(F, K, tau, sigma, cp=cp)
    intrinsic = max(cp * (F - K), 0.0)
    # Skip deep-OTM / worthless options where the premium above intrinsic is too
    # small to invert (the solver correctly returns 0 there, by design).
    if black76_vega(F, K, tau, sigma) < 1e-6 or (price - intrinsic) < 1e-4:
        return
    recovered = implied_vol(price, F, K, tau, cp=cp)
    assert not math.isnan(recovered)
    assert recovered == pytest.approx(sigma, rel=1e-3, abs=1e-3)


@settings(max_examples=100, deadline=None)
@given(F=forwards, K=strikes, tau=taus, sigma=sigmas)
def test_vega_positive(F: float, K: float, tau: float, sigma: float) -> None:
    assert black76_vega(F, K, tau, sigma) >= 0.0


def test_iv_out_of_bounds_returns_nan() -> None:
    F, K, tau = 50_000.0, 50_000.0, 0.5
    # Price above the forward upper bound is impossible.
    assert math.isnan(implied_vol(F + 1.0, F, K, tau, cp=1))
    # Price below intrinsic is impossible.
    assert math.isnan(implied_vol(-1.0, F, K, tau, cp=1))


def test_iv_nonfinite_inputs_return_nan() -> None:
    # Codex P2 regression: non-finite inputs must follow the documented "bad
    # quotes return nan" path, not fall through to a bogus finite sigma.
    F, K, tau = 50_000.0, 50_000.0, 0.5
    assert math.isnan(implied_vol(math.nan, F, K, tau, cp=1))
    assert math.isnan(implied_vol(math.inf, F, K, tau, cp=1))
    assert math.isnan(implied_vol(1_000.0, math.inf, K, tau, cp=1))
    assert math.isnan(implied_vol(1_000.0, F, math.nan, tau, cp=1))
    assert math.isnan(implied_vol(1_000.0, F, K, math.nan, cp=1))


def test_iv_at_intrinsic_is_zero() -> None:
    F, K, tau = 50_000.0, 48_000.0, 0.5
    assert implied_vol(F - K, F, K, tau, cp=1) == pytest.approx(0.0, abs=1e-6)


def test_greeks_consistency_with_finite_difference() -> None:
    F, K, tau, sigma = 50_000.0, 51_000.0, 0.5, 0.7
    g = black76_greeks(F, K, tau, sigma, cp=1)
    h = 1e-2
    # Delta vs central difference in F.
    fd_delta = (
        black76_price(F + h, K, tau, sigma, cp=1) - black76_price(F - h, K, tau, sigma, cp=1)
    ) / (2 * h)
    assert g["delta"] == pytest.approx(fd_delta, rel=1e-4)
    # Vega vs central difference in sigma.
    hs = 1e-4
    fd_vega = (
        black76_price(F, K, tau, sigma + hs, cp=1) - black76_price(F, K, tau, sigma - hs, cp=1)
    ) / (2 * hs)
    assert g["vega"] == pytest.approx(fd_vega, rel=1e-3)
    assert g["vega"] == pytest.approx(black76_vega(F, K, tau, sigma), rel=1e-9)


def test_gamma_matches_second_difference() -> None:
    F, K, tau, sigma = 50_000.0, 50_000.0, 0.5, 0.7
    g = black76_greeks(F, K, tau, sigma, cp=1)
    h = 1.0
    fd_gamma = (
        black76_price(F + h, K, tau, sigma, cp=1)
        - 2 * black76_price(F, K, tau, sigma, cp=1)
        + black76_price(F - h, K, tau, sigma, cp=1)
    ) / (h * h)
    assert g["gamma"] == pytest.approx(fd_gamma, rel=1e-3)
    assert np.isfinite(g["theta"])
