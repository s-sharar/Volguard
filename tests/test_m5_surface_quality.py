"""Model-domain quality diagnostics on each surface's native fixed-k support."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from volguard.features.surface_quality import SurfaceDomainQuality, check_model_grid
from volguard.surface.arbitrage import check_butterfly_from_w


def _mutate_calendar_count(quality: Any) -> None:
    quality.calendar_violations = 1


def test_clean_nonuniform_native_grid_is_deterministic_and_frozen() -> None:
    k = np.array(
        [
            [-1.0, -0.65, -0.10, 0.35, 1.0],
            [-0.8, -0.25, 0.15, 0.70, 1.2],
            [-0.7, -0.15, 0.30, 0.85, 1.3],
        ]
    )
    w = np.array([[0.04] * 5, [0.06] * 5, [0.09] * 5])
    taus = np.array([0.25, 0.5, 1.0])

    first = check_model_grid(k, w, taus)
    second = check_model_grid(k, w, taus)

    assert isinstance(first, SurfaceDomainQuality)
    assert first == second
    assert first.ok
    assert first.butterfly_violations == 0
    assert first.vertical_violations == 0
    assert first.calendar_violations == 0
    assert first.minimum_overlap_width == pytest.approx(1.8)
    assert first.insufficient_overlap_pairs == 0
    with pytest.raises(FrozenInstanceError):
        _mutate_calendar_count(first)


def test_native_row_butterfly_and_vertical_results_are_aggregated() -> None:
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
        ]
    )
    taus = np.array([0.5, 1.0])
    native = check_butterfly_from_w(k[0], w[0], float(taus[0]))

    quality = check_model_grid(k, w, taus)

    assert native.butterfly_violations > 0
    assert quality.butterfly_violations == native.butterfly_violations
    assert quality.vertical_violations == native.vertical_violations
    assert quality.max_butterfly_magnitude == native.max_butterfly_magnitude
    assert quality.integrated_butterfly_magnitude == native.integrated_butterfly_magnitude
    assert not quality.ok


def test_calendar_crossing_inside_shared_overlap_is_measured_on_exact_points() -> None:
    k = np.array(
        [
            [-1.0, -0.55, -0.05, 0.45, 1.0],
            [-0.7, -0.20, 0.25, 0.75, 1.3],
        ]
    )
    w = np.array([[0.06] * 5, [0.04] * 5])

    quality = check_model_grid(k, w, np.array([0.25, 0.5]), calendar_points=9)

    assert quality.calendar_violations == 9
    assert quality.max_calendar_magnitude == pytest.approx(0.02)
    assert quality.integrated_calendar_magnitude == pytest.approx(0.18)
    assert quality.max_relative_calendar_deficit == pytest.approx(0.4)
    assert quality.integrated_relative_calendar_deficit == pytest.approx(3.6)
    assert quality.minimum_overlap_width == pytest.approx(1.7)
    assert not quality.ok


def test_calendar_crossing_only_outside_shared_overlap_is_ignored() -> None:
    k = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    # The short tenor exceeds the long tenor only where the long tenor has no
    # native support. Inside [0, 2], it remains below the long tenor.
    w = np.array([[0.30, 0.20, 0.05, 0.05, 0.05], [0.10] * 5])

    quality = check_model_grid(k, w, np.array([0.25, 0.5]))

    assert quality.calendar_violations == 0
    assert quality.max_calendar_magnitude == 0.0
    assert quality.integrated_calendar_magnitude == 0.0
    assert quality.max_relative_calendar_deficit == 0.0
    assert quality.integrated_relative_calendar_deficit == 0.0
    assert quality.minimum_overlap_width == pytest.approx(2.0)


def test_nonoverlapping_pair_is_flagged_without_extrapolation() -> None:
    k = np.array(
        [
            [-3.0, -2.5, -2.0, -1.5, -1.0],
            [1.0, 1.5, 2.0, 2.5, 3.0],
        ]
    )
    w = np.array([[0.04] * 5, [0.06] * 5])

    quality = check_model_grid(k, w, np.array([0.25, 0.5]))

    assert quality.insufficient_overlap_pairs == 1
    assert quality.minimum_overlap_width == 0.0
    assert quality.calendar_violations == 0
    assert not quality.ok


def test_zero_total_variance_is_valid() -> None:
    k = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    w = np.array([[0.0, 0.0, 0.0], [0.04, 0.04, 0.04]])

    quality = check_model_grid(k, w, np.array([0.25, 0.5]))

    assert quality.ok


@pytest.mark.parametrize(
    ("k", "w", "taus", "calendar_points", "match"),
    [
        (np.ones((2, 3)), np.ones((2, 4)), np.array([0.1, 0.2]), 9, "same shape"),
        (np.ones(3), np.ones(3), np.array([0.1, 0.2]), 9, "2D"),
        (np.ones((2, 2)), np.ones((2, 2)), np.array([0.1, 0.2]), 9, "3 native"),
        (np.ones((1, 3)), np.ones((1, 3)), np.array([0.1]), 9, "2 tenors"),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, np.nan]]),
            np.ones((2, 3)),
            np.array([0.1, 0.2]),
            9,
            "finite",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            np.array([[0.1, 0.1, 0.1], [0.1, np.inf, 0.1]]),
            np.array([0.1, 0.2]),
            9,
            "finite",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            np.array([[0.1, 0.1, 0.1], [0.1, -0.01, 0.1]]),
            np.array([0.1, 0.2]),
            9,
            "nonnegative",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]]),
            np.ones((2, 3)),
            np.array([0.1, 0.2]),
            9,
            "strictly increasing",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            np.ones((2, 3)),
            np.array([0.1]),
            9,
            "taus length",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            np.ones((2, 3)),
            np.array([0.1, 0.0]),
            9,
            "positive",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            np.ones((2, 3)),
            np.array([0.2, 0.1]),
            9,
            "strictly increasing",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            np.ones((2, 3)),
            np.array([0.1, 0.2]),
            2,
            "calendar_points",
        ),
        (
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            np.ones((2, 3)),
            np.array([0.1, 0.2]),
            3.5,
            "integer",
        ),
    ],
)
def test_invalid_model_grid_inputs_raise_clear_errors(
    k: np.ndarray,
    w: np.ndarray,
    taus: np.ndarray,
    calendar_points: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        check_model_grid(k, w, taus, calendar_points=calendar_points)
