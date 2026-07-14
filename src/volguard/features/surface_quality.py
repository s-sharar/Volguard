"""Pure model-domain quality diagnostics for native surface grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volguard.surface.arbitrage import check_butterfly_from_w, check_calendar

FloatArray = NDArray[np.float64]

_ARBITRAGE_EPS = 1e-10
_MIN_CALENDAR_POINTS = 3
_MIN_NATIVE_POINTS = 3
_MIN_TENORS = 2
_SURFACE_NDIM = 2

SURFACE_QUALITY_FLAG_ORDER: tuple[str, ...] = (
    "m4_butterfly_post",
    "m4_calendar_post",
    "butterfly_certification_failed",
    "calendar_certification_failed",
    "model_butterfly_violations",
    "model_vertical_violations",
    "model_calendar_violations",
    "model_insufficient_overlap",
    "low_observation_support",
    "interpolated_grid_cells",
    "extrapolated_grid_cells",
)


@dataclass(frozen=True, slots=True)
class SurfaceDomainQuality:
    """Aggregated arbitrage diagnostics on the surface's represented domain."""

    butterfly_violations: int = 0
    vertical_violations: int = 0
    calendar_violations: int = 0
    max_butterfly_magnitude: float = 0.0
    max_vertical_magnitude: float = 0.0
    max_calendar_magnitude: float = 0.0
    integrated_butterfly_magnitude: float = 0.0
    integrated_vertical_magnitude: float = 0.0
    integrated_calendar_magnitude: float = 0.0
    max_relative_calendar_deficit: float = 0.0
    integrated_relative_calendar_deficit: float = 0.0
    minimum_overlap_width: float = 0.0
    insufficient_overlap_pairs: int = 0

    @property
    def ok(self) -> bool:
        """Whether all represented-domain checks passed."""
        return (
            self.butterfly_violations == 0
            and self.vertical_violations == 0
            and self.calendar_violations == 0
            and self.insufficient_overlap_pairs == 0
        )


def _validated_inputs(
    k_by_tenor: ArrayLike,
    w_by_tenor: ArrayLike,
    taus: ArrayLike,
    calendar_points: int,
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    """Normalize and validate model-grid inputs before any diagnostics run."""
    k = np.asarray(k_by_tenor, dtype=float)
    w = np.asarray(w_by_tenor, dtype=float)
    tau_values = np.asarray(taus, dtype=float)

    if k.ndim != _SURFACE_NDIM or w.ndim != _SURFACE_NDIM:
        raise ValueError("k_by_tenor and w_by_tenor must be 2D")
    if k.shape != w.shape:
        raise ValueError("k_by_tenor and w_by_tenor must have the same shape")
    if k.shape[0] < _MIN_TENORS:
        raise ValueError("model grid must contain at least 2 tenors")
    if k.shape[1] < _MIN_NATIVE_POINTS:
        raise ValueError("each tenor must contain at least 3 native k points")
    if tau_values.ndim != 1 or tau_values.size != k.shape[0]:
        raise ValueError("taus length must match the number of tenor rows")

    _validate_values(k, w, tau_values)

    if isinstance(calendar_points, bool) or not isinstance(calendar_points, (int, np.integer)):
        raise ValueError("calendar_points must be an integer")
    if calendar_points < _MIN_CALENDAR_POINTS:
        raise ValueError("calendar_points must be at least 3")

    return k, w, tau_values, int(calendar_points)


def _validate_values(k: FloatArray, w: FloatArray, tau_values: FloatArray) -> None:
    """Validate finite values, variance domains, and native ordering."""
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(w)):
        raise ValueError("model grid values must be finite")
    if not np.all(np.isfinite(tau_values)):
        raise ValueError("taus must be finite")
    if np.any(w < 0.0):
        raise ValueError("w_by_tenor values must be nonnegative")
    if np.any(tau_values <= 0.0):
        raise ValueError("taus must be positive")
    if not np.all(np.diff(k, axis=1) > 0.0):
        raise ValueError("each native k row must be strictly increasing")
    if not np.all(np.diff(tau_values) > 0.0):
        raise ValueError("taus must be strictly increasing")


def check_model_grid(
    k_by_tenor: ArrayLike,
    w_by_tenor: ArrayLike,
    taus: ArrayLike,
    *,
    calendar_points: int = 9,
) -> SurfaceDomainQuality:
    """Check native tenor rows and adjacent-tenor shared fixed-k support.

    Butterfly and vertical-spread diagnostics are evaluated directly on each
    native row. Calendar diagnostics use exactly ``calendar_points`` linearly
    spaced samples within each adjacent pair's shared interval, so neither row
    is ever extrapolated.
    """
    k, w, tau_values, n_calendar_points = _validated_inputs(
        k_by_tenor,
        w_by_tenor,
        taus,
        calendar_points,
    )

    butterfly_violations = 0
    vertical_violations = 0
    max_butterfly_magnitude = 0.0
    max_vertical_magnitude = 0.0
    integrated_butterfly_magnitude = 0.0
    integrated_vertical_magnitude = 0.0

    for k_row, w_row, tau in zip(k, w, tau_values, strict=True):
        report = check_butterfly_from_w(k_row, w_row, float(tau))
        butterfly_violations += report.butterfly_violations
        vertical_violations += report.vertical_violations
        max_butterfly_magnitude = max(
            max_butterfly_magnitude,
            report.max_butterfly_magnitude,
        )
        max_vertical_magnitude = max(max_vertical_magnitude, report.max_vertical_magnitude)
        integrated_butterfly_magnitude += report.integrated_butterfly_magnitude
        integrated_vertical_magnitude += report.integrated_vertical_magnitude

    calendar_violations = 0
    max_calendar_magnitude = 0.0
    integrated_calendar_magnitude = 0.0
    max_relative_calendar_deficit = 0.0
    integrated_relative_calendar_deficit = 0.0
    insufficient_overlap_pairs = 0
    overlap_widths: list[float] = []

    for pair_index in range(k.shape[0] - 1):
        short_k = k[pair_index]
        long_k = k[pair_index + 1]
        overlap_min = max(float(short_k[0]), float(long_k[0]))
        overlap_max = min(float(short_k[-1]), float(long_k[-1]))
        overlap_width = max(0.0, overlap_max - overlap_min)
        overlap_widths.append(overlap_width)

        if overlap_width <= 0.0:
            insufficient_overlap_pairs += 1
            continue

        shared_k = np.linspace(overlap_min, overlap_max, n_calendar_points)
        short_w = np.interp(shared_k, short_k, w[pair_index])
        long_w = np.interp(shared_k, long_k, w[pair_index + 1])
        pair_report = check_calendar(
            shared_k,
            np.vstack((short_w, long_w)),
            tau_values[pair_index : pair_index + 2],
        )

        calendar_violations += pair_report.calendar_violations
        max_calendar_magnitude = max(
            max_calendar_magnitude,
            pair_report.max_calendar_magnitude,
        )
        integrated_calendar_magnitude += pair_report.integrated_calendar_magnitude

        raw_deficit = short_w - long_w
        violating_deficit = np.where(raw_deficit > _ARBITRAGE_EPS, raw_deficit, 0.0)
        relative_deficit = violating_deficit / np.maximum(
            0.5 * (short_w + long_w),
            _ARBITRAGE_EPS,
        )
        max_relative_calendar_deficit = max(
            max_relative_calendar_deficit,
            float(relative_deficit.max(initial=0.0)),
        )
        integrated_relative_calendar_deficit += float(relative_deficit.sum())

    return SurfaceDomainQuality(
        butterfly_violations=butterfly_violations,
        vertical_violations=vertical_violations,
        calendar_violations=calendar_violations,
        max_butterfly_magnitude=max_butterfly_magnitude,
        max_vertical_magnitude=max_vertical_magnitude,
        max_calendar_magnitude=max_calendar_magnitude,
        integrated_butterfly_magnitude=integrated_butterfly_magnitude,
        integrated_vertical_magnitude=integrated_vertical_magnitude,
        integrated_calendar_magnitude=integrated_calendar_magnitude,
        max_relative_calendar_deficit=max_relative_calendar_deficit,
        integrated_relative_calendar_deficit=integrated_relative_calendar_deficit,
        minimum_overlap_width=min(overlap_widths),
        insufficient_overlap_pairs=insufficient_overlap_pairs,
    )
