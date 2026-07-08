"""Tests for the arbitrage checker and the repair QP."""

from __future__ import annotations

import numpy as np
import pytest

from volguard.surface.arbitrage import (
    check_butterfly_from_w,
    check_calendar,
    check_surface,
)
from volguard.surface.repair import repair_surface
from volguard.surface.svi import SVIParams, svi_total_variance


def _clean_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A calm, arb-free SVI surface on a shared grid."""
    k = np.linspace(-1.0, 1.0, 21)
    taus = np.array([7, 30, 90, 180]) / 365.0
    smiles = [SVIParams(a=0.01 + 0.05 * t, b=0.1, rho=-0.1, m=0.0, sigma=0.2) for t in taus]
    w = np.vstack([svi_total_variance(p, k) for p in smiles])
    return k, w, taus


def test_clean_butterfly_passes() -> None:
    k, w, taus = _clean_surface()
    rep = check_butterfly_from_w(k, w[1], float(taus[1]))
    assert rep.ok
    assert rep.butterfly_violations == 0


def test_clean_calendar_passes() -> None:
    k, w, taus = _clean_surface()
    rep = check_calendar(k, w, taus)
    assert rep.ok
    assert rep.calendar_violations == 0


def test_calendar_violation_detected() -> None:
    k, w, taus = _clean_surface()
    # Force a calendar inversion: make a long-tenor slice smaller than a short.
    w_bad = w.copy()
    w_bad[2] = w[1] * 0.5
    rep = check_calendar(k, w_bad, taus)
    assert not rep.ok
    assert rep.calendar_violations > 0
    assert rep.max_calendar_magnitude > 0.0


def test_negative_variance_flagged_by_butterfly_check() -> None:
    # Codex P1 regression: negative total variance is impossible and must be
    # reported, not silently clipped to zero-vol intrinsic calls.
    k = np.array([-1.0, 0.0, 1.0])
    w = np.array([-0.01, -0.01, -0.01])
    rep = check_butterfly_from_w(k, w, tau=0.5)
    assert not rep.ok
    assert rep.negative_variance_violations == 3
    assert rep.max_negative_variance_magnitude > 0.0


def test_negative_variance_flagged_by_surface_check() -> None:
    k = np.linspace(-1.0, 1.0, 5)
    taus = np.array([0.25, 0.5])
    w = np.full((2, 5), -0.01)
    rep = check_surface(k, w, taus)
    assert not rep.ok
    assert rep.negative_variance_violations > 0


def test_vertical_spread_violation_detected() -> None:
    # Codex P1 regression: this smile produces call prices that are convex but
    # *increasing* in strike — a vertical spread with negative cost and
    # non-negative payoff. Convexity-only checking marks it arb-free.
    k = np.array([0.0, 0.5, 1.0])
    w = np.array([0.0044, 1.23, 18.0])
    rep = check_butterfly_from_w(k, w, tau=0.5)
    assert not rep.ok
    assert rep.vertical_violations > 0
    assert rep.max_vertical_magnitude > 0.0


def test_vertical_spread_flagged_by_surface_check() -> None:
    k = np.array([0.0, 0.5, 1.0])
    w = np.array([[0.0044, 1.23, 18.0]])
    rep = check_surface(k, w, np.array([0.5]))
    assert not rep.ok
    assert rep.vertical_violations > 0


def test_butterfly_rejects_nonpositive_tau() -> None:
    k = np.array([-0.5, 0.0, 0.5])
    w = np.array([0.04, 0.04, 0.04])
    with pytest.raises(ValueError, match="tau"):
        check_butterfly_from_w(k, w, tau=0.0)


def test_butterfly_rejects_nonfinite_tau() -> None:
    k = np.array([-0.5, 0.0, 0.5])
    w = np.array([0.04, 0.04, 0.04])
    with pytest.raises(ValueError, match="finite"):
        check_butterfly_from_w(k, w, tau=np.nan)


def test_surface_rejects_nonfinite_tau() -> None:
    k = np.array([-0.5, 0.0, 0.5])
    w = np.array([[0.04, 0.04, 0.04]])
    with pytest.raises(ValueError, match="finite"):
        check_surface(k, w, np.array([np.nan]))


def test_surface_rejects_nonfinite_strikes() -> None:
    k = np.array([-0.5, np.nan, 0.5])
    w = np.array([[0.04, 0.04, 0.04]])
    with pytest.raises(ValueError, match="finite"):
        check_surface(k, w, np.array([0.5]))


def test_surface_rejects_duplicate_strikes() -> None:
    k = np.array([-0.5, 0.0, 0.0])
    w = np.array([[0.04, 0.04, 0.04]])
    with pytest.raises(ValueError, match="unique"):
        check_surface(k, w, np.array([0.5]))


def test_calendar_rejects_mismatched_k_grid() -> None:
    with pytest.raises(ValueError, match="k_grid"):
        check_calendar(np.array([0.0, 0.1]), np.ones((2, 3)), np.array([0.1, 0.2]))


def test_calendar_rejects_nonpositive_taus() -> None:
    with pytest.raises(ValueError, match="positive"):
        check_calendar(np.array([-0.5, 0.0, 0.5]), np.ones((2, 3)), np.array([-0.1, 0.2]))


def test_nonfinite_variance_flagged_by_butterfly_check() -> None:
    # Codex P1 regression: NaN passes every `< -eps` comparison, so a non-finite
    # grid must be rejected explicitly rather than reported as arbitrage-free.
    k = np.array([-1.0, 0.0, 1.0])
    for bad in (np.nan, np.inf):
        w = np.array([bad, bad, bad])
        rep = check_butterfly_from_w(k, w, tau=0.5)
        assert not rep.ok
        assert rep.nonfinite_violations == 3


def test_nonfinite_variance_flagged_by_surface_check() -> None:
    k = np.array([-1.0, 0.0, 1.0])
    w = np.array([[np.nan, np.nan, np.nan]])
    rep = check_surface(k, w, np.array([0.5]))
    assert not rep.ok
    assert rep.nonfinite_violations > 0


def test_nonfinite_variance_flagged_by_calendar_check() -> None:
    k = np.array([-1.0, 0.0, 1.0])
    w = np.array([[0.04, 0.04, 0.04], [np.inf, 0.05, 0.05]])
    rep = check_calendar(k, w, np.array([0.25, 0.5]))
    assert not rep.ok
    assert rep.nonfinite_violations > 0


def test_calendar_requires_increasing_taus() -> None:
    k, w, _ = _clean_surface()
    with pytest.raises(ValueError):
        check_calendar(k, w, np.array([0.5, 0.1, 0.9, 1.0]))


def test_check_surface_clean() -> None:
    k, w, taus = _clean_surface()
    rep = check_surface(k, w, taus)
    assert rep.ok


def test_repair_is_identity_on_clean_surface() -> None:
    k, w, taus = _clean_surface()
    repaired = repair_surface(k, w, taus)
    assert np.allclose(repaired, w, atol=1e-4)


def test_repair_fixes_calendar_violation() -> None:
    k, w, taus = _clean_surface()
    w_bad = w.copy()
    w_bad[2] = w[1] * 0.5  # calendar inversion
    assert not check_calendar(k, w_bad, taus).ok

    repaired = repair_surface(k, w_bad, taus)
    # Repaired surface must pass the calendar check (within tolerance).
    rep = check_calendar(k, repaired, taus, eps=1e-6)
    assert rep.ok


def test_repair_fixes_call_price_butterfly_when_w_is_convex() -> None:
    # Codex P1 regression: this slice is convex in total variance w, so a
    # w-convexity-only repair leaves it unchanged, yet its Black-76 call prices
    # are NOT convex in strike -> the primary checker still flags butterfly arb.
    # The repaired surface must pass check_butterfly_from_w.
    tau = 0.5
    k = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    w = np.array([[0.5223, 0.0748, 0.4493, 1.7387, 4.0282]])
    taus = np.array([tau])

    assert not check_butterfly_from_w(k, w[0], tau).ok  # bad input, as claimed
    repaired = repair_surface(k, w, taus)
    rep = check_butterfly_from_w(k, repaired[0], tau, eps=1e-8)
    assert rep.ok


def test_repair_fixes_vertical_spread() -> None:
    # Increasing call prices (vertical-spread arb) must be repaired away.
    k = np.array([0.0, 0.5, 1.0])
    w = np.array([[0.0044, 1.23, 18.0]])
    taus = np.array([0.5])

    assert not check_butterfly_from_w(k, w[0], float(taus[0])).ok
    repaired = repair_surface(k, w, taus)
    rep = check_butterfly_from_w(k, repaired[0], float(taus[0]), eps=1e-8)
    assert rep.ok


def test_repair_rejects_bad_shapes() -> None:
    k = np.linspace(-1, 1, 5)
    w = np.ones((3, 4))  # mismatched columns
    taus = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        repair_surface(k, w, taus)


def test_repair_rejects_nonpositive_taus() -> None:
    k = np.linspace(-1, 1, 5)
    w = np.full((2, 5), 0.04)
    with pytest.raises(ValueError, match="positive"):
        repair_surface(k, w, np.array([-0.1, 0.5]))


def test_repair_rejects_bad_weights_shape() -> None:
    k = np.linspace(-1, 1, 5)
    w = np.full((2, 5), 0.04)
    taus = np.array([0.25, 0.5])
    with pytest.raises(ValueError, match="weights"):
        repair_surface(k, w, taus, weights=np.ones(5))  # 1D, silently broadcast before
