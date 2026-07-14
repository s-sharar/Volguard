"""Arbitrage-repair QP: project a surface onto the nearest arb-free one.

Given a (possibly arbitraging) total-variance grid ``w[tenor, k]``, find the
closest grid ``w*`` (in weighted or uniform L2) that satisfies discrete
no-arbitrage. Two geometries are supported:

- **Shared grid** (:func:`repair_surface` / :func:`repair_surface_result`): one
  ``k`` axis for all tenors; calendar is cellwise monotonicity.
- **Native grid** (:func:`repair_surface_native`): one strictly increasing ``k``
  row per tenor (M5 6x9); calendar uses fixed linear interpolation over pairwise
  shared fixed-``k`` support and never extrapolates.

Call price is a nonlinear (monotone) function of total variance, so butterfly
convexity is not linear in ``w``. We therefore solve a **sequential QP**: at each
step we linearize the call prices around the current iterate, impose discrete
convexity of those linearized prices plus calendar monotonicity, solve the
convex QP, then relinearize. Success requires the corresponding checker to pass;
a tiny move alone never counts as success (stalled invalid iterates raise).

Solved with cvxpy + OSQP (both core dependencies).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from volguard.curate.blackiv import black76_price, black76_vega
from volguard.features.surface_quality import SurfaceDomainQuality, check_model_grid
from volguard.surface.arbitrage import ArbitrageReport, check_surface

FloatArray = NDArray[np.float64]
ArbReport = ArbitrageReport | SurfaceDomainQuality

_MIN_MONEYNESS_POINTS = 3
_MIN_TENORS_FOR_CALENDAR = 2
_MIN_NATIVE_TENORS = 2
_SURFACE_NDIM = 2
_K_GRID_NDIM = 1
_W_FLOOR = 1e-10
_DEFAULT_CALENDAR_POINTS = 9
# OSQP's default tolerances (1e-5) leave constraint residuals around 1e-7 that
# the arbitrage checker would then flag on the repaired surface; tight eps plus
# solution polishing brings residuals to ~1e-12.
_OSQP_OPTS: dict[str, object] = {
    "eps_abs": 1e-10,
    "eps_rel": 1e-10,
    "max_iter": 100_000,
    "polishing": True,
}
_OPTIMAL_STATUSES = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})


class RepairConvergenceError(RuntimeError):
    """Raised when sequential-QP repair cannot certify an arbitrage-free surface."""


@dataclass(frozen=True, slots=True)
class RepairResult:
    """Immutable record of a certified arbitrage repair."""

    repaired_w: FloatArray
    status: str
    iterations: int
    max_move: float
    repair_distance_l2: float
    repair_distance_linf: float
    arbitrage_report: ArbReport


@dataclass(frozen=True, slots=True)
class _CalendarPairMatrices:
    """Fixed linear-interpolation matrices for one adjacent tenor pair."""

    short_index: int
    long_index: int
    short_matrix: FloatArray  # (n_points, nk)
    long_matrix: FloatArray


def _call_price_linearization(
    strikes: FloatArray, w_by_tenor: FloatArray, taus: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Black-76 call prices ``C0`` and slopes ``dC/dw`` at each grid cell.

    ``strikes`` may be shape ``(nk,)`` (shared) or ``(n_tenor, nk)`` (native).
    Forward is normalized to 1. ``dC/dw`` uses ``vega / (2·sigma·tau)``.
    """
    n_tenor, nk = w_by_tenor.shape
    strike_grid = (
        np.broadcast_to(np.asarray(strikes, dtype=float), (n_tenor, nk))
        if np.asarray(strikes, dtype=float).ndim == _K_GRID_NDIM
        else np.asarray(strikes, dtype=float)
    )
    c0 = np.zeros((n_tenor, nk))
    dc_dw = np.zeros((n_tenor, nk))
    for j in range(n_tenor):
        tau = float(taus[j])
        for i in range(nk):
            w_ji = max(float(w_by_tenor[j, i]), _W_FLOOR)
            sigma = np.sqrt(w_ji / tau)
            strike = float(strike_grid[j, i])
            c0[j, i] = black76_price(1.0, strike, tau, sigma, cp=1)
            vega = black76_vega(1.0, strike, tau, sigma)
            dc_dw[j, i] = vega / (2.0 * sigma * tau)
    return c0, dc_dw


def _linear_interp_matrix(x_src: FloatArray, x_query: FloatArray) -> FloatArray:
    """Return ``M`` such that ``M @ f_src`` equals linear interpolation at ``x_query``.

    ``x_query`` must lie inside ``[x_src[0], x_src[-1]]`` (no extrapolation).
    """
    x_src = np.asarray(x_src, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    if x_src.ndim != _K_GRID_NDIM or x_query.ndim != _K_GRID_NDIM:
        raise ValueError("interpolation axes must be 1D")
    if not np.all(np.diff(x_src) > 0.0):
        raise ValueError("source axis must be strictly increasing")
    if np.any(x_query < x_src[0] - 1e-15) or np.any(x_query > x_src[-1] + 1e-15):
        raise ValueError("query points must lie within source support (no extrapolation)")

    n_src = x_src.size
    matrix = np.zeros((x_query.size, n_src), dtype=float)
    for row, xq in enumerate(x_query):
        if xq <= x_src[0]:
            matrix[row, 0] = 1.0
            continue
        if xq >= x_src[-1]:
            matrix[row, -1] = 1.0
            continue
        right = int(np.searchsorted(x_src, xq, side="left"))
        left = right - 1
        width = x_src[right] - x_src[left]
        weight = (xq - x_src[left]) / width
        matrix[row, left] = 1.0 - weight
        matrix[row, right] = weight
    return matrix


def _build_calendar_pair_matrices(
    k_by_tenor: FloatArray, *, calendar_points: int
) -> list[_CalendarPairMatrices]:
    """Build frozen interp matrices for every adjacent tenor pair with overlap."""
    pairs: list[_CalendarPairMatrices] = []
    for short_index in range(k_by_tenor.shape[0] - 1):
        long_index = short_index + 1
        short_k = k_by_tenor[short_index]
        long_k = k_by_tenor[long_index]
        overlap_min = max(float(short_k[0]), float(long_k[0]))
        overlap_max = min(float(short_k[-1]), float(long_k[-1]))
        overlap_width = overlap_max - overlap_min
        if overlap_width <= 0.0:
            raise ValueError(
                "insufficient overlap between adjacent tenors for calendar constraints"
            )
        shared_k = np.linspace(overlap_min, overlap_max, calendar_points)
        pairs.append(
            _CalendarPairMatrices(
                short_index=short_index,
                long_index=long_index,
                short_matrix=_linear_interp_matrix(short_k, shared_k),
                long_matrix=_linear_interp_matrix(long_k, shared_k),
            )
        )
    return pairs


def forecast_geometry_k(
    d_grid: FloatArray,
    w_by_tenor: FloatArray,
    *,
    atm_index: int | None = None,
) -> FloatArray:
    """Derive frozen native ``k`` from standardized moneyness and forecast ATM w.

    ``k[j, i] = d[i] * sqrt(raw_w[j, ATM])``. Uses only the forecast state's ATM
    total variance so target-day geometry cannot leak into repair.
    """
    d_grid = np.asarray(d_grid, dtype=float)
    w_by_tenor = np.asarray(w_by_tenor, dtype=float)
    if d_grid.ndim != _K_GRID_NDIM:
        raise ValueError("d_grid must be 1D")
    if w_by_tenor.ndim != _SURFACE_NDIM:
        raise ValueError("w_by_tenor must be 2D")
    if d_grid.size != w_by_tenor.shape[1]:
        raise ValueError("d_grid length must match w_by_tenor columns")
    if not np.all(np.isfinite(d_grid)) or not np.all(np.isfinite(w_by_tenor)):
        raise ValueError("d_grid and w_by_tenor must be finite")
    idx = (d_grid.size // 2) if atm_index is None else int(atm_index)
    if idx < 0 or idx >= d_grid.size:
        raise ValueError("atm_index out of range")
    atm_w = w_by_tenor[:, idx]
    if np.any(atm_w < 0.0):
        raise ValueError("ATM total variance must be nonnegative")
    return np.outer(np.sqrt(np.maximum(atm_w, 0.0)), d_grid)


def _validate_shared_geometry(k_grid: FloatArray, w_by_tenor: FloatArray, taus: FloatArray) -> None:
    """Validate shared-grid axes, shapes, and finiteness."""
    if k_grid.ndim != _K_GRID_NDIM:
        raise ValueError("k_grid must be 1D")
    if w_by_tenor.ndim != _SURFACE_NDIM:
        raise ValueError("w_by_tenor must be 2D (n_tenor, nk)")
    n_tenor, nk = w_by_tenor.shape
    if k_grid.size != nk:
        raise ValueError("k_grid length must match w_by_tenor columns")
    if taus.size != n_tenor:
        raise ValueError("taus length must match w_by_tenor rows")
    if nk < _MIN_MONEYNESS_POINTS:
        raise ValueError("need at least 3 moneyness points for convexity")
    if not np.all(np.isfinite(k_grid)):
        raise ValueError("k_grid values must be finite")
    if not np.all(np.isfinite(w_by_tenor)):
        raise ValueError("w_by_tenor values must be finite")
    if not np.all(np.isfinite(taus)):
        raise ValueError("taus values must be finite")
    if not np.all(np.diff(k_grid) > 0.0):
        raise ValueError("k_grid must be strictly increasing")
    if np.any(taus <= 0.0):
        raise ValueError("taus must be positive")
    if taus.size > 1 and not np.all(np.diff(taus) > 0.0):
        raise ValueError("taus must be strictly increasing")


def _validate_solver_settings(*, max_iter: int, move_tol: float) -> None:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if not np.isfinite(move_tol) or move_tol <= 0.0:
        raise ValueError("move_tol must be finite and positive")


def _validate_weights(weights: FloatArray | None, shape: tuple[int, ...]) -> FloatArray:
    if weights is None:
        weight_grid = np.ones(shape, dtype=float)
    else:
        weight_grid = np.asarray(weights, dtype=float)
    if weight_grid.shape != shape:
        raise ValueError("weights shape must match w_by_tenor")
    if not np.all(np.isfinite(weight_grid)):
        raise ValueError("weights must be finite")
    if np.any(weight_grid < 0.0):
        raise ValueError("weights must be nonnegative")
    return weight_grid


def _validate_shared_inputs(
    k_grid: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    weights: FloatArray | None,
    *,
    max_iter: int,
    move_tol: float,
) -> FloatArray:
    """Validate shared-grid repair inputs; return nonnegative finite weight grid."""
    _validate_shared_geometry(k_grid, w_by_tenor, taus)
    _validate_solver_settings(max_iter=max_iter, move_tol=move_tol)
    return _validate_weights(weights, w_by_tenor.shape)


def _validate_native_geometry(
    k_by_tenor: FloatArray, w_by_tenor: FloatArray, taus: FloatArray
) -> None:
    """Validate native-grid shapes, finiteness, and axis ordering."""
    if k_by_tenor.ndim != _SURFACE_NDIM or w_by_tenor.ndim != _SURFACE_NDIM:
        raise ValueError("k_by_tenor and w_by_tenor must be 2D")
    if k_by_tenor.shape != w_by_tenor.shape:
        raise ValueError("k_by_tenor and w_by_tenor must have the same shape")
    if k_by_tenor.shape[0] < _MIN_NATIVE_TENORS:
        raise ValueError("native repair requires at least 2 tenors")
    if k_by_tenor.shape[1] < _MIN_MONEYNESS_POINTS:
        raise ValueError("need at least 3 moneyness points for convexity")
    if taus.ndim != _K_GRID_NDIM or taus.size != k_by_tenor.shape[0]:
        raise ValueError("taus length must match the number of tenor rows")
    if not np.all(np.isfinite(k_by_tenor)) or not np.all(np.isfinite(w_by_tenor)):
        raise ValueError("k_by_tenor and w_by_tenor values must be finite")
    if not np.all(np.isfinite(taus)):
        raise ValueError("taus values must be finite")
    if np.any(w_by_tenor < 0.0):
        raise ValueError("w_by_tenor values must be nonnegative")
    if np.any(taus <= 0.0):
        raise ValueError("taus must be positive")
    if not np.all(np.diff(k_by_tenor, axis=1) > 0.0):
        raise ValueError("each native k row must be strictly increasing")
    if not np.all(np.diff(taus) > 0.0):
        raise ValueError("taus must be strictly increasing")


def _validate_native_inputs(
    k_by_tenor: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    *,
    max_iter: int,
    move_tol: float,
    calendar_points: int,
) -> int:
    """Validate native-grid inputs; return sanitized ``calendar_points``."""
    _validate_native_geometry(k_by_tenor, w_by_tenor, taus)
    if isinstance(calendar_points, bool) or not isinstance(calendar_points, (int, np.integer)):
        raise ValueError("calendar_points must be an integer")
    if calendar_points < _MIN_MONEYNESS_POINTS:
        raise ValueError("calendar_points must be at least 3")
    _validate_solver_settings(max_iter=max_iter, move_tol=move_tol)
    return int(calendar_points)


def _finalize_qp_solution(problem: cp.Problem, x: cp.Variable) -> FloatArray:
    if problem.status not in _OPTIMAL_STATUSES or x.value is None:
        raise RepairConvergenceError(f"repair QP did not solve (status={problem.status})")
    new = np.asarray(x.value, dtype=float)
    if not np.all(np.isfinite(new)):
        raise RepairConvergenceError("repair QP returned nonfinite solution")
    return new


def _solve_linearized_qp_step(
    *,
    current: FloatArray,
    target: FloatArray,
    sqrt_w: FloatArray,
    strikes: FloatArray,
    taus: FloatArray,
    solver: object,
) -> FloatArray:
    """One shared-grid sequential-QP step."""
    n_tenor, nk = current.shape
    c0, dc_dw = _call_price_linearization(strikes, current, taus)
    dK = np.diff(np.asarray(strikes, dtype=float))

    x = cp.Variable((n_tenor, nk))
    objective = cp.Minimize(cp.sum_squares(cp.multiply(sqrt_w, x - target)))

    constraints: list[cp.Constraint] = [x >= 0]
    if n_tenor >= _MIN_TENORS_FOR_CALENDAR:
        constraints.append(x[1:, :] - x[:-1, :] >= 0)
    c_lin = c0 + cp.multiply(dc_dw, x - current)
    slopes = (c_lin[:, 1:] - c_lin[:, :-1]) / dK[None, :]
    constraints.append(slopes[:, 1:] - slopes[:, :-1] >= 0)
    constraints.append(slopes <= 0)
    constraints.append(slopes >= -1)

    problem = cp.Problem(objective, constraints)
    solve_kwargs = _OSQP_OPTS if solver == cp.OSQP else {}
    problem.solve(solver=solver, **solve_kwargs)
    return _finalize_qp_solution(problem, x)


def _solve_native_linearized_qp_step(
    *,
    current: FloatArray,
    target: FloatArray,
    strikes_by_tenor: FloatArray,
    taus: FloatArray,
    calendar_pairs: list[_CalendarPairMatrices],
    solver: object,
) -> FloatArray:
    """One native-grid sequential-QP step (uniform L2 objective)."""
    n_tenor, nk = current.shape
    c0, dc_dw = _call_price_linearization(strikes_by_tenor, current, taus)

    x = cp.Variable((n_tenor, nk))
    objective = cp.Minimize(cp.sum_squares(x - target))
    constraints: list[cp.Constraint] = [x >= 0]

    for j in range(n_tenor):
        dK = np.diff(strikes_by_tenor[j])
        c_lin = c0[j] + cp.multiply(dc_dw[j], x[j] - current[j])
        slopes = (c_lin[1:] - c_lin[:-1]) / dK
        constraints.append(slopes[1:] - slopes[:-1] >= 0)
        constraints.append(slopes <= 0)
        constraints.append(slopes >= -1)

    for pair in calendar_pairs:
        short_w = pair.short_matrix @ x[pair.short_index]
        long_w = pair.long_matrix @ x[pair.long_index]
        constraints.append(long_w - short_w >= 0)

    problem = cp.Problem(objective, constraints)
    solve_kwargs = _OSQP_OPTS if solver == cp.OSQP else {}
    problem.solve(solver=solver, **solve_kwargs)
    return _finalize_qp_solution(problem, x)


def _repair_distances(raw: FloatArray, repaired: FloatArray) -> tuple[float, float]:
    delta = repaired - raw
    return float(np.linalg.norm(delta.ravel())), float(np.max(np.abs(delta)))


def _report_summary(report: ArbReport) -> str:
    return (
        f"bfly={report.butterfly_violations}, "
        f"vert={report.vertical_violations}, "
        f"cal={report.calendar_violations}"
    )


def _run_certified_sequential_qp(
    *,
    target: FloatArray,
    max_iter: int,
    move_tol: float,
    initial_report: ArbReport,
    step: Callable[[FloatArray], FloatArray],
    check: Callable[[FloatArray], ArbReport],
) -> RepairResult:
    """Shared certified loop used by shared-grid and native-grid repair."""
    if initial_report.ok:
        repaired = target.copy()
        l2, linf = _repair_distances(target, repaired)
        return RepairResult(
            repaired_w=repaired,
            status="optimal",
            iterations=0,
            max_move=0.0,
            repair_distance_l2=l2,
            repair_distance_linf=linf,
            arbitrage_report=initial_report,
        )

    current = target.copy()
    max_move = 0.0
    for it in range(1, max_iter + 1):
        new = step(current)
        move = float(np.max(np.abs(new - current)))
        max_move = max(max_move, move)
        current = new
        report = check(current)
        if report.ok:
            l2, linf = _repair_distances(target, current)
            return RepairResult(
                repaired_w=current,
                status="optimal",
                iterations=it,
                max_move=max_move,
                repair_distance_l2=l2,
                repair_distance_linf=linf,
                arbitrage_report=report,
            )
        if move < move_tol:
            raise RepairConvergenceError(
                "stalled with residual arbitrage "
                f"(iter={it}, move={move:.3e}, {_report_summary(report)})"
            )

    final = check(current)
    raise RepairConvergenceError(
        "exhausted iterations with residual arbitrage "
        f"(max_iter={max_iter}, {_report_summary(final)})"
    )


def repair_surface_result(
    k_grid: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    *,
    weights: FloatArray | None = None,
    solver: str | None = None,
    max_iter: int = 10,
    move_tol: float = 1e-10,
    eps: float = 1e-10,
) -> RepairResult:
    """Project onto the nearest arb-free shared grid; return a certified ``RepairResult``.

    Raises ``RepairConvergenceError`` if the checker still fails after the iterate
    stalls or iterations are exhausted. A tiny move alone is never success.
    """
    k_grid = np.asarray(k_grid, dtype=float)
    w_by_tenor = np.asarray(w_by_tenor, dtype=float)
    taus = np.asarray(taus, dtype=float)
    weight_grid = _validate_shared_inputs(
        k_grid, w_by_tenor, taus, weights, max_iter=max_iter, move_tol=move_tol
    )

    sqrt_w = np.sqrt(weight_grid)
    strikes = np.exp(k_grid)
    chosen = solver or cp.OSQP

    def _step(current: FloatArray) -> FloatArray:
        return _solve_linearized_qp_step(
            current=current,
            target=w_by_tenor,
            sqrt_w=sqrt_w,
            strikes=strikes,
            taus=taus,
            solver=chosen,
        )

    def _check(current: FloatArray) -> ArbReport:
        return check_surface(k_grid, current, taus, eps=eps)

    return _run_certified_sequential_qp(
        target=w_by_tenor,
        max_iter=max_iter,
        move_tol=move_tol,
        initial_report=check_surface(k_grid, w_by_tenor, taus, eps=eps),
        step=_step,
        check=_check,
    )


def repair_surface(
    k_grid: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    *,
    weights: FloatArray | None = None,
    solver: str | None = None,
    max_iter: int = 10,
    move_tol: float = 1e-10,
) -> FloatArray:
    """Return the nearest arbitrage-free total-variance grid to ``w_by_tenor``.

    Compatibility wrapper around :func:`repair_surface_result`. Returns only a
    certified grid, or raises :class:`RepairConvergenceError`.

    Args:
        k_grid: log-moneyness grid, shape (nk,), strictly increasing.
        w_by_tenor: total variance, shape (n_tenor, nk).
        taus: tenor times in years, shape (n_tenor,), strictly increasing.
        weights: optional per-cell weights (e.g. vega), shape (n_tenor, nk).
        solver: cvxpy solver name; defaults to OSQP.
        max_iter: max sequential-QP relinearization steps.
        move_tol: stall threshold (max abs move); stall with residual arb raises.
    """
    return repair_surface_result(
        k_grid,
        w_by_tenor,
        taus,
        weights=weights,
        solver=solver,
        max_iter=max_iter,
        move_tol=move_tol,
    ).repaired_w


def repair_surface_native(
    k_by_tenor: FloatArray,
    w_by_tenor: FloatArray,
    taus: FloatArray,
    *,
    solver: str | None = None,
    max_iter: int = 10,
    move_tol: float = 1e-10,
    calendar_points: int = _DEFAULT_CALENDAR_POINTS,
) -> RepairResult:
    """Project a native per-tenor grid onto the nearest arb-free surface.

    Geometry ``k_by_tenor`` is frozen for the whole projection. Butterfly and
    vertical constraints use each tenor's native strikes. Calendar constraints
    use fixed linear interpolation over pairwise shared fixed-``k`` support and
    never extrapolate. Objective is uniform L2 (vega/reliability never enter).

    Certifies with :func:`volguard.features.surface_quality.check_model_grid`.
    """
    k_by_tenor = np.asarray(k_by_tenor, dtype=float)
    w_by_tenor = np.asarray(w_by_tenor, dtype=float)
    taus = np.asarray(taus, dtype=float)
    n_calendar_points = _validate_native_inputs(
        k_by_tenor,
        w_by_tenor,
        taus,
        max_iter=max_iter,
        move_tol=move_tol,
        calendar_points=calendar_points,
    )
    calendar_pairs = _build_calendar_pair_matrices(k_by_tenor, calendar_points=n_calendar_points)
    strikes_by_tenor = np.exp(k_by_tenor)
    chosen = solver or cp.OSQP

    def _step(current: FloatArray) -> FloatArray:
        return _solve_native_linearized_qp_step(
            current=current,
            target=w_by_tenor,
            strikes_by_tenor=strikes_by_tenor,
            taus=taus,
            calendar_pairs=calendar_pairs,
            solver=chosen,
        )

    def _check(current: FloatArray) -> ArbReport:
        return check_model_grid(k_by_tenor, current, taus, calendar_points=n_calendar_points)

    return _run_certified_sequential_qp(
        target=w_by_tenor,
        max_iter=max_iter,
        move_tol=move_tol,
        initial_report=check_model_grid(
            k_by_tenor, w_by_tenor, taus, calendar_points=n_calendar_points
        ),
        step=_step,
        check=_check,
    )
