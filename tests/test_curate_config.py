"""M3 unit tests for :class:`CurateConfig` (design Model 1, requirements 12.1-12.6)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from volguard.config import CurateConfig, load_config


def test_defaults_load_from_yaml() -> None:
    cfg = load_config("curate", CurateConfig)
    assert isinstance(cfg, CurateConfig)
    # Snap window (R12.1).
    assert cfg.snap_hour_utc == 8
    assert cfg.snap_minute_utc == 5
    assert cfg.window_minutes == 60
    assert cfg.widen_step_minutes == 60
    assert cfg.max_window_minutes == 360
    assert cfg.min_trades_per_expiry == 4
    assert cfg.recency_half_life_s == 900.0
    # Forward inference (R12.2).
    assert cfg.pcp_pair_window_s == 60.0
    assert cfg.min_pcp_pairs == 1
    # Filters (R12.3).
    assert cfg.tau_min_days == 2.0
    assert cfg.delta_min == 0.02
    assert cfg.delta_max == 0.98
    assert cfg.iv_min == 0.01
    assert cfg.iv_max == 5.0
    assert cfg.mad_multiplier == 5.0
    assert cfg.min_size_btc == 0.1
    assert cfg.iv_divergence_tol == 0.02


def test_defaults_construct_without_yaml() -> None:
    cfg = CurateConfig()
    assert cfg.window_minutes <= cfg.max_window_minutes


def test_tau_min_years_derivation() -> None:
    """R12.4: tau_min_years == tau_min_days / 365."""
    cfg = CurateConfig()
    assert cfg.tau_min_years == pytest.approx(2.0 / 365.0)
    cfg2 = CurateConfig(tau_min_days=365.0)
    assert cfg2.tau_min_years == pytest.approx(1.0)
    assert math.isfinite(cfg.tau_min_years)


# --- R12.5: band ordering + positivity ---------------------------------------


def test_delta_min_not_less_than_delta_max_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(delta_min=0.9, delta_max=0.5)


def test_delta_bands_equal_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(delta_min=0.5, delta_max=0.5)


def test_iv_min_not_less_than_iv_max_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(iv_min=5.0, iv_max=1.0)


def test_non_positive_delta_band_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(delta_min=-0.1, delta_max=0.98)


def test_non_positive_iv_band_rejected() -> None:
    # iv_min <= 0 while keeping iv_min < iv_max so positivity is what trips.
    with pytest.raises(ValidationError):
        CurateConfig(iv_min=0.0, iv_max=5.0)


# --- R12.6: window bounds, trade count, positive knobs ------------------------


def test_window_minutes_exceeding_max_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(window_minutes=400, max_window_minutes=360)


def test_min_trades_per_expiry_below_one_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(min_trades_per_expiry=0)


def test_non_positive_iv_divergence_tol_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(iv_divergence_tol=0.0)


def test_non_positive_mad_multiplier_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(mad_multiplier=-1.0)


def test_non_positive_recency_half_life_rejected() -> None:
    with pytest.raises(ValidationError):
        CurateConfig(recency_half_life_s=0.0)


def test_window_minutes_equal_to_max_allowed() -> None:
    cfg = CurateConfig(window_minutes=360, max_window_minutes=360)
    assert cfg.window_minutes == cfg.max_window_minutes
