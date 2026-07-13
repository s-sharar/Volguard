"""Unit + property tests for ``curate/filters.py`` IV cross-check (design R3).

Covers :func:`volguard.curate.filters.cross_check_iv` and the row-level branch
logic behind it:

- Property **CP2** — IV recompute round-trip vs Black-76: for valid
  ``(F, K, tau, sigma*, cp)`` inside the no-arb bounds, pricing then re-solving
  via the M1 solver recovers ``sigma*``. Deep-wing ``nan`` cases are excluded /
  flagged :attr:`QualityFlag.IV_UNSOLVABLE`, not treated as failures (R3.1).
- Unit tests for the cross-check branches: trade-primary vs recompute-fallback,
  the divergence threshold boundary, and the unsolvable flag (R3.2-R3.5).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volguard.config import CurateConfig
from volguard.curate.blackiv import CallPut, black76_price, black76_vega, implied_vol
from volguard.curate.filters import QualityFlag, cross_check_iv

_TS = pl.Datetime(time_unit="ms", time_zone="UTC")
_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_EXPIRY = datetime(2022, 4, 8, 8, 0, tzinfo=UTC)
_TAU = (_EXPIRY - _SNAP).total_seconds() / (365.0 * 24.0 * 3600.0)
_CFG = CurateConfig()


# --- frame builder --------------------------------------------------------


def _rows(
    specs: list[tuple[float, float, str, int, float, float]],
) -> pl.DataFrame:
    """Build a minimal post-``attach_forward`` frame for the cross-check.

    Each spec is ``(F, strike, cp, cp_sign, usd_premium, iv_trade)``. ``iv_trade``
    may be ``nan`` to exercise the recompute-fallback branch.
    """
    return pl.DataFrame(
        {
            "F": pl.Series([s[0] for s in specs], dtype=pl.Float64),
            "strike": pl.Series([s[1] for s in specs], dtype=pl.Float64),
            "cp": pl.Series([s[2] for s in specs], dtype=pl.String),
            "cp_sign": pl.Series([s[3] for s in specs], dtype=pl.Int64),
            "usd_premium": pl.Series([s[4] for s in specs], dtype=pl.Float64),
            "iv_trade": pl.Series([s[5] for s in specs], dtype=pl.Float64),
            "tau": pl.Series([_TAU] * len(specs), dtype=pl.Float64),
        }
    )


def _row_with_tau(
    forward: float,
    strike: float,
    cp: str,
    cp_sign: int,
    usd_premium: float,
    iv_trade: float,
    tau: float,
) -> pl.DataFrame:
    """Single-row frame with an explicit ``tau`` (CP2 draws tau, not ``_TAU``)."""
    return pl.DataFrame(
        {
            "F": pl.Series([forward], dtype=pl.Float64),
            "strike": pl.Series([strike], dtype=pl.Float64),
            "cp": pl.Series([cp], dtype=pl.String),
            "cp_sign": pl.Series([cp_sign], dtype=pl.Int64),
            "usd_premium": pl.Series([usd_premium], dtype=pl.Float64),
            "iv_trade": pl.Series([iv_trade], dtype=pl.Float64),
            "tau": pl.Series([tau], dtype=pl.Float64),
        }
    )


def _flag_set(value: int, flag: QualityFlag) -> bool:
    return bool(value & int(flag))


# --- Task 5.3: cross-check branch unit tests ------------------------------


def test_trade_primary_when_iv_trade_finite() -> None:
    """R3.2: a finite iv_trade is used as iv_obs with iv_source='trade'."""
    f, k, sigma = 45_000.0, 45_000.0, 0.7
    premium = black76_price(f, k, _TAU, sigma, cp=1)
    out = cross_check_iv(_rows([(f, k, "C", 1, premium, sigma)]), _CFG)
    assert out["iv_source"][0] == "trade"
    assert out["iv_obs"][0] == pytest.approx(sigma)
    # Recompute agrees, so no divergence flag.
    assert not _flag_set(out["quality_flags"][0], QualityFlag.IV_DIVERGENCE)


def test_recompute_fallback_when_iv_trade_missing() -> None:
    """R3.3: iv_trade not finite -> iv_obs = iv_recomputed, iv_source='mark'."""
    f, k, sigma = 45_000.0, 45_000.0, 0.7
    premium = black76_price(f, k, _TAU, sigma, cp=1)
    out = cross_check_iv(_rows([(f, k, "C", 1, premium, math.nan)]), _CFG)
    assert out["iv_source"][0] == "mark"
    assert out["iv_obs"][0] == pytest.approx(sigma, rel=1e-4)
    assert out["iv_recomputed"][0] == pytest.approx(sigma, rel=1e-4)


def test_iv_source_always_in_allowed_set() -> None:
    """R3.6: iv_source is always one of {trade, mark, mid}."""
    f, k, sigma = 45_000.0, 45_000.0, 0.7
    premium = black76_price(f, k, _TAU, sigma, cp=1)
    out = cross_check_iv(
        _rows(
            [
                (f, k, "C", 1, premium, sigma),  # trade
                (f, k, "C", 1, premium, math.nan),  # mark
                (f, k, "C", 1, -1.0, math.nan),  # unsolvable -> still 'mark'
            ]
        ),
        _CFG,
    )
    assert set(out["iv_source"].to_list()) <= {"trade", "mark", "mid"}


def test_divergence_flag_below_threshold_not_set() -> None:
    """R3.4: divergence within tol does NOT raise IV_DIVERGENCE."""
    f, k, sigma = 45_000.0, 45_000.0, 0.7
    premium = black76_price(f, k, _TAU, sigma, cp=1)
    # iv_trade offset by just under the tolerance from the (true) recompute.
    iv_trade = sigma + _CFG.iv_divergence_tol * 0.5
    out = cross_check_iv(_rows([(f, k, "C", 1, premium, iv_trade)]), _CFG)
    assert out["iv_source"][0] == "trade"
    assert out["iv_obs"][0] == pytest.approx(iv_trade)
    assert not _flag_set(out["quality_flags"][0], QualityFlag.IV_DIVERGENCE)


def test_divergence_flag_above_threshold_set_and_retained() -> None:
    """R3.4: |iv_trade - iv_recomputed| > tol sets IV_DIVERGENCE and retains the row."""
    f, k, sigma = 45_000.0, 45_000.0, 0.7
    premium = black76_price(f, k, _TAU, sigma, cp=1)
    # iv_trade well beyond the tolerance from the recompute.
    iv_trade = sigma + _CFG.iv_divergence_tol * 3.0
    out = cross_check_iv(_rows([(f, k, "C", 1, premium, iv_trade)]), _CFG)
    assert out.height == 1  # retained
    assert out["iv_source"][0] == "trade"
    assert out["iv_obs"][0] == pytest.approx(iv_trade)  # trade IV kept as-is
    assert _flag_set(out["quality_flags"][0], QualityFlag.IV_DIVERGENCE)


def test_unsolvable_flag_when_solver_returns_nan() -> None:
    """R3.5: an out-of-no-arb-bounds premium yields IV_UNSOLVABLE."""
    f, k = 45_000.0, 45_000.0
    # A premium above the forward upper bound is not invertible -> nan.
    out = cross_check_iv(_rows([(f, k, "C", 1, f + 1.0, math.nan)]), _CFG)
    assert math.isnan(out["iv_recomputed"][0])
    assert math.isnan(out["iv_obs"][0])
    assert out["iv_source"][0] == "mark"
    assert _flag_set(out["quality_flags"][0], QualityFlag.IV_UNSOLVABLE)


def test_unsolvable_but_trade_iv_present_falls_back_to_trade() -> None:
    """R3.2/R3.5: unsolvable recompute still keeps a finite iv_trade as iv_obs."""
    f, k = 45_000.0, 45_000.0
    out = cross_check_iv(_rows([(f, k, "C", 1, f + 1.0, 0.65)]), _CFG)
    assert out["iv_source"][0] == "trade"
    assert out["iv_obs"][0] == pytest.approx(0.65)
    assert _flag_set(out["quality_flags"][0], QualityFlag.IV_UNSOLVABLE)


def test_existing_quality_flags_are_preserved() -> None:
    """Prior quality_flags are ORed with the cross-check flags (composability)."""
    f, k = 45_000.0, 45_000.0
    df = _rows([(f, k, "C", 1, f + 1.0, math.nan)]).with_columns(
        pl.Series("quality_flags", [int(QualityFlag.BLOCK_TRADE)], dtype=pl.Int64)
    )
    out = cross_check_iv(df, _CFG)
    flags = out["quality_flags"][0]
    assert _flag_set(flags, QualityFlag.BLOCK_TRADE)
    assert _flag_set(flags, QualityFlag.IV_UNSOLVABLE)


def test_empty_frame_yields_typed_columns() -> None:
    """An empty input produces the added columns with correct dtypes."""
    out = cross_check_iv(_rows([]), _CFG)
    assert out.height == 0
    assert out.schema["iv_recomputed"] == pl.Float64
    assert out.schema["iv_obs"] == pl.Float64
    assert out.schema["iv_source"] == pl.String
    assert out.schema["quality_flags"] == pl.Int64


def test_put_recompute_uses_cp_sign() -> None:
    """A put (cp_sign=-1) recomputes correctly through the solver."""
    f, k, sigma = 45_000.0, 47_000.0, 0.8
    premium = black76_price(f, k, _TAU, sigma, cp=-1)
    out = cross_check_iv(_rows([(f, k, "P", -1, premium, math.nan)]), _CFG)
    assert out["iv_source"][0] == "mark"
    assert out["iv_obs"][0] == pytest.approx(sigma, rel=1e-4)


# --- Task 5.2 / Property CP2: IV recompute round-trip vs Black-76 ----------


@settings(max_examples=200, deadline=None)
@given(
    forward=st.floats(min_value=1_000.0, max_value=200_000.0),
    strike=st.floats(min_value=1_000.0, max_value=200_000.0),
    tau=st.floats(min_value=1e-3, max_value=3.0),
    sigma=st.floats(min_value=0.05, max_value=3.0),
    cp=st.sampled_from([1, -1]),
)
def test_cp2_iv_recompute_round_trip(
    forward: float, strike: float, tau: float, sigma: float, cp: CallPut
) -> None:
    """CP2: price at sigma* then cross-check recovers sigma* to tolerance.

    Deep-wing inputs where the premium above intrinsic is too small to invert
    (the solver returns ``0``/``nan`` by design) are excluded here and are
    flagged :attr:`QualityFlag.IV_UNSOLVABLE` in production, not test failures.

    **Validates: Requirements 3.1**
    """
    premium = black76_price(forward, strike, tau, sigma, cp=cp)
    intrinsic = max(cp * (forward - strike), 0.0)
    # Skip worthless / deep-OTM options with negligible extrinsic value: the
    # solver correctly cannot recover sigma from a premium at intrinsic.
    if black76_vega(forward, strike, tau, sigma) < 1e-6 or (premium - intrinsic) < 1e-4:
        return

    cp_str = "C" if cp == 1 else "P"
    out = cross_check_iv(
        _row_with_tau(forward, strike, cp_str, int(cp), premium, math.nan, tau), _CFG
    )
    recovered = out["iv_recomputed"][0]
    # Sanity: cross_check_iv agrees with a direct solver call.
    assert recovered == pytest.approx(implied_vol(premium, forward, strike, tau, cp=cp))
    assert not math.isnan(recovered)
    assert recovered == pytest.approx(sigma, rel=1e-3, abs=1e-3)
    # Fallback path: no iv_trade -> iv_obs is the recomputed mark.
    assert out["iv_source"][0] == "mark"
    assert out["iv_obs"][0] == pytest.approx(sigma, rel=1e-3, abs=1e-3)
