"""M6 repair certification: returned surfaces must pass the checker or raise."""

from __future__ import annotations

import numpy as np
import pytest

from volguard.features.surface_quality import check_model_grid
from volguard.surface.arbitrage import check_surface
from volguard.surface.repair import (
    RepairConvergenceError,
    RepairResult,
    forecast_geometry_k,
    repair_surface,
    repair_surface_native,
    repair_surface_result,
)
from volguard.surface.svi import SVIParams, svi_total_variance


def _clean_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.linspace(-1.0, 1.0, 21)
    taus = np.array([7, 30, 90, 180]) / 365.0
    smiles = [SVIParams(a=0.01 + 0.05 * t, b=0.1, rho=-0.1, m=0.0, sigma=0.2) for t in taus]
    w = np.vstack([svi_total_variance(p, k) for p in smiles])
    return k, w, taus


def _calendar_violation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared-grid surface with a clear calendar inversion (long below short)."""
    k, w, taus = _clean_surface()
    w_bad = w.copy()
    w_bad[2] = w[1] * 0.5
    assert not check_surface(k, w_bad, taus).ok
    return k, w_bad, taus


def test_returned_repair_always_passes_checker() -> None:
    k, w_bad, taus = _calendar_violation()
    repaired = repair_surface(k, w_bad, taus)
    assert check_surface(k, repaired, taus).ok


def test_stalled_invalid_iterate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    k, w_bad, taus = _calendar_violation()

    def _stalled_step(
        *,
        current: np.ndarray,
        target: np.ndarray,
        sqrt_w: np.ndarray,
        strikes: np.ndarray,
        taus: np.ndarray,
        solver: object,
    ) -> np.ndarray:
        del target, sqrt_w, strikes, taus, solver
        # Tiny move that stays invalid — previously treated as success via move_tol.
        return current + 1e-14

    monkeypatch.setattr("volguard.surface.repair._solve_linearized_qp_step", _stalled_step)

    with pytest.raises(RepairConvergenceError, match="stalled"):
        repair_surface(k, w_bad, taus)

    # Compatibility wrapper must not silently return an ndarray on failure.
    try:
        repair_surface(k, w_bad, taus)
    except RepairConvergenceError:
        pass
    else:
        pytest.fail("expected RepairConvergenceError, got a repaired array")


def test_exhausted_iterations_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    k, w_bad, taus = _calendar_violation()

    n_calls = {"n": 0}

    def _moving_but_invalid(
        *,
        current: np.ndarray,
        target: np.ndarray,
        sqrt_w: np.ndarray,
        strikes: np.ndarray,
        taus: np.ndarray,
        solver: object,
    ) -> np.ndarray:
        del target, sqrt_w, strikes, taus, solver
        n_calls["n"] += 1
        # Keep moving enough to avoid a stall, but never certify.
        return current + 1e-3

    monkeypatch.setattr("volguard.surface.repair._solve_linearized_qp_step", _moving_but_invalid)

    with pytest.raises(RepairConvergenceError, match="exhausted"):
        repair_surface(k, w_bad, taus, max_iter=3)
    assert n_calls["n"] == 3


def test_nonoptimal_solver_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    k, w_bad, taus = _calendar_violation()

    def _bad_status(**kwargs: object) -> np.ndarray:
        del kwargs
        raise RepairConvergenceError("repair QP did not solve (status=infeasible)")

    monkeypatch.setattr("volguard.surface.repair._solve_linearized_qp_step", _bad_status)

    with pytest.raises(RepairConvergenceError, match="did not solve"):
        repair_surface(k, w_bad, taus)


def test_clean_surface_is_identity() -> None:
    k = np.linspace(-1.0, 1.0, 7)
    taus = np.array([0.1, 0.25, 0.5])
    w = np.outer(taus, 0.04 + 0.01 * (k**2))
    assert check_surface(k, w, taus).ok
    repaired = repair_surface(k, w, taus)
    np.testing.assert_allclose(repaired, w, atol=1e-6)


def test_clean_surface_result_metadata() -> None:
    k, w, taus = _clean_surface()
    result = repair_surface_result(k, w, taus)
    assert isinstance(result, RepairResult)
    assert result.status == "optimal"
    assert result.arbitrage_report.ok
    assert result.iterations >= 0
    assert result.max_move >= 0.0
    assert result.repair_distance_l2 >= 0.0
    assert result.repair_distance_linf >= 0.0
    np.testing.assert_allclose(result.repaired_w, w, atol=1e-4)


def test_repair_fixes_butterfly_vertical_calendar() -> None:
    k, w, taus = _clean_surface()

    # Calendar.
    w_cal = w.copy()
    w_cal[2] = w[1] * 0.5
    assert not check_surface(k, w_cal, taus).ok
    assert check_surface(k, repair_surface(k, w_cal, taus), taus).ok

    # Butterfly on call prices (convex in w, not in C).
    k_b = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    w_b = np.array([[0.5223, 0.0748, 0.4493, 1.7387, 4.0282]])
    taus_b = np.array([0.5])
    assert not check_surface(k_b, w_b, taus_b).ok
    assert check_surface(k_b, repair_surface(k_b, w_b, taus_b), taus_b).ok

    # Vertical spread.
    k_v = np.array([0.0, 0.5, 1.0])
    w_v = np.array([[0.0044, 1.23, 18.0]])
    taus_v = np.array([0.5])
    assert not check_surface(k_v, w_v, taus_v).ok
    assert check_surface(k_v, repair_surface(k_v, w_v, taus_v), taus_v).ok


def test_rejects_nonfinite_inputs() -> None:
    k = np.linspace(-1.0, 1.0, 5)
    w = np.full((2, 5), 0.04)
    taus = np.array([0.25, 0.5])
    w_nan = w.copy()
    w_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        repair_surface(k, w_nan, taus)
    with pytest.raises(ValueError, match="finite"):
        repair_surface(np.array([-1.0, np.nan, 0.0, 0.5, 1.0]), w, taus)
    with pytest.raises(ValueError, match="finite"):
        repair_surface(k, w, np.array([0.25, np.inf]))


def test_rejects_invalid_weights_and_settings() -> None:
    k = np.linspace(-1.0, 1.0, 5)
    w = np.full((2, 5), 0.04)
    taus = np.array([0.25, 0.5])

    with pytest.raises(ValueError, match="weights"):
        repair_surface(k, w, taus, weights=np.full_like(w, -1.0))
    with pytest.raises(ValueError, match="weights"):
        bad_w = np.ones_like(w)
        bad_w[0, 0] = np.nan
        repair_surface(k, w, taus, weights=bad_w)
    with pytest.raises(ValueError, match="max_iter"):
        repair_surface(k, w, taus, max_iter=0)
    with pytest.raises(ValueError, match="move_tol"):
        repair_surface(k, w, taus, move_tol=0.0)
    with pytest.raises(ValueError, match="move_tol"):
        repair_surface(k, w, taus, move_tol=-1e-8)


# --- Native-grid repair (M5 6x9 tenor-specific geometry) ---


def _clean_native_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.array(
        [
            [-1.0, -0.65, -0.10, 0.35, 1.0],
            [-0.8, -0.25, 0.15, 0.70, 1.2],
            [-0.7, -0.15, 0.30, 0.85, 1.3],
        ]
    )
    w = np.array([[0.04] * 5, [0.06] * 5, [0.09] * 5], dtype=float)
    taus = np.array([0.25, 0.5, 1.0])
    assert check_model_grid(k, w, taus).ok
    return k, w, taus


def test_native_clean_surface_is_identity() -> None:
    k, w, taus = _clean_native_surface()
    result = repair_surface_native(k, w, taus)
    assert isinstance(result, RepairResult)
    assert result.status == "optimal"
    assert result.iterations == 0
    assert result.arbitrage_report.ok
    np.testing.assert_allclose(result.repaired_w, w)


def test_native_repairs_calendar_violation() -> None:
    k = np.array(
        [
            [-1.0, -0.55, -0.05, 0.45, 1.0],
            [-0.7, -0.20, 0.25, 0.75, 1.3],
        ]
    )
    w = np.array([[0.06] * 5, [0.04] * 5], dtype=float)
    taus = np.array([0.25, 0.5])
    assert not check_model_grid(k, w, taus).ok

    result = repair_surface_native(k, w, taus)
    assert result.arbitrage_report.ok
    assert check_model_grid(k, result.repaired_w, taus).ok


def test_native_repairs_butterfly_on_native_row() -> None:
    k = np.array(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [-0.8, -0.3, 0.2, 0.7, 1.2],
        ]
    )
    w = np.array(
        [
            [0.5223, 0.0748, 0.4493, 1.7387, 4.0282],
            [0.6, 0.6, 0.6, 0.6, 0.6],
        ],
        dtype=float,
    )
    taus = np.array([0.5, 1.0])
    assert not check_model_grid(k, w, taus).ok

    result = repair_surface_native(k, w, taus)
    assert result.arbitrage_report.ok
    assert check_model_grid(k, result.repaired_w, taus).ok


def test_native_stalled_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    k = np.array(
        [
            [-1.0, -0.55, -0.05, 0.45, 1.0],
            [-0.7, -0.20, 0.25, 0.75, 1.3],
        ]
    )
    w = np.array([[0.06] * 5, [0.04] * 5], dtype=float)
    taus = np.array([0.25, 0.5])

    def _stalled(**kwargs: object) -> np.ndarray:
        current = np.asarray(kwargs["current"], dtype=float)
        return current + 1e-14

    monkeypatch.setattr("volguard.surface.repair._solve_native_linearized_qp_step", _stalled)

    with pytest.raises(RepairConvergenceError, match="stalled"):
        repair_surface_native(k, w, taus)


def test_native_insufficient_overlap_fails_clearly() -> None:
    k = np.array(
        [
            [-3.0, -2.5, -2.0, -1.5, -1.0],
            [1.0, 1.5, 2.0, 2.5, 3.0],
        ]
    )
    w = np.array([[0.04] * 5, [0.06] * 5], dtype=float)
    taus = np.array([0.25, 0.5])
    with pytest.raises(ValueError, match="insufficient overlap"):
        repair_surface_native(k, w, taus)


def test_native_calendar_never_extrapolates_outside_shared_support() -> None:
    # Crossing only outside shared overlap must be ignored by the checker and
    # therefore treated as already-clean (identity) by repair.
    k = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    w = np.array([[0.30, 0.20, 0.05, 0.05, 0.05], [0.10] * 5], dtype=float)
    taus = np.array([0.25, 0.5])
    assert check_model_grid(k, w, taus).ok
    result = repair_surface_native(k, w, taus)
    np.testing.assert_allclose(result.repaired_w, w)
    assert result.iterations == 0


def test_forecast_geometry_k_from_atm_w() -> None:
    d = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    w = np.array([[0.04, 0.04, 0.04, 0.04, 0.04], [0.09, 0.09, 0.09, 0.09, 0.09]])
    k = forecast_geometry_k(d, w)
    np.testing.assert_allclose(k[0], d * 0.2)
    np.testing.assert_allclose(k[1], d * 0.3)


def test_native_rejects_malformed_axes() -> None:
    k, w, taus = _clean_native_surface()
    k_bad = k.copy()
    k_bad[0, 1] = 2.0  # breaks strict increase in row 0
    with pytest.raises(ValueError, match="strictly increasing"):
        repair_surface_native(k_bad, w, taus)

    w_neg = w.copy()
    w_neg[0, 0] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        repair_surface_native(k, w_neg, taus)
