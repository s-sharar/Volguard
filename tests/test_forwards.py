"""Unit + property tests for ``curate/forwards.py`` (design Component 2).

Covers the three-tier forward inference and forward attachment:

- :func:`infer_forward_for_expiry` — each tier in isolation (PCP / future /
  index_carry), median robustness to one bad pair, and the fallthrough
  ordering PCP -> future -> index_carry (design R2.3, R2.4, R2.6).
- Property **CP1** — put-call parity forward inference correctness: C/P
  premiums synthesized from a known ``F*`` via the M1 Black-76 pricer recover
  ``F*`` with ``method == pcp`` and ``n_pairs == len(strikes)`` (R2.1, R2.2).
- Property **CP8** — log-moneyness consistency: ``k == ln(strike / F)`` to
  float tolerance and ``k > 0`` iff ``strike > F`` (R7.5).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volguard.config import CurateConfig
from volguard.curate.blackiv import black76_price
from volguard.curate.forwards import (
    ForwardEstimate,
    ForwardMethod,
    attach_forward,
    infer_forward_for_expiry,
    infer_forwards,
)

_TS = pl.Datetime(time_unit="ms", time_zone="UTC")
_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_EXPIRY = datetime(2022, 4, 8, 8, 0, tzinfo=UTC)
_TAU = (_EXPIRY - _SNAP).total_seconds() / (365.0 * 24.0 * 3600.0)
_CFG = CurateConfig()


# --- frame builders -------------------------------------------------------


def _options(
    rows: list[tuple[float, str, float, datetime]],
    *,
    expiry: datetime = _EXPIRY,
) -> pl.DataFrame:
    """Build a minimal canonical option frame: (strike, cp, usd_premium, ts)."""
    return pl.DataFrame(
        {
            "expiry": pl.Series([expiry] * len(rows), dtype=_TS),
            "strike": pl.Series([r[0] for r in rows], dtype=pl.Float64),
            "cp": pl.Series([r[1] for r in rows], dtype=pl.String),
            "usd_premium": pl.Series([r[2] for r in rows], dtype=pl.Float64),
            "source_ts": pl.Series([r[3] for r in rows], dtype=_TS),
            "tau": pl.Series([_TAU] * len(rows), dtype=pl.Float64),
            "index_price": pl.Series([45000.0] * len(rows), dtype=pl.Float64),
        }
    )


def _empty_options() -> pl.DataFrame:
    return _options([])


def _futures(rows: list[tuple[str, float, datetime]]) -> pl.DataFrame:
    """Build a minimal futures frame: (instrument, price, ts)."""
    return pl.DataFrame(
        {
            "instrument": pl.Series([r[0] for r in rows], dtype=pl.String),
            "price": pl.Series([r[1] for r in rows], dtype=pl.Float64),
            "ts": pl.Series([r[2] for r in rows], dtype=_TS),
        }
    )


def _empty_futures() -> pl.DataFrame:
    return _futures([])


def _funding(rows: list[tuple[float, datetime]]) -> pl.DataFrame:
    """Build a minimal funding frame: (interest_8h, ts)."""
    return pl.DataFrame(
        {
            "interest_8h": pl.Series([r[0] for r in rows], dtype=pl.Float64),
            "interest_1h": pl.Series([None] * len(rows), dtype=pl.Float64),
            "ts": pl.Series([r[1] for r in rows], dtype=_TS),
        }
    )


def _empty_funding() -> pl.DataFrame:
    return _funding([])


# --- Tier 1: put-call parity ----------------------------------------------


def _cp_premiums(f_star: float, strike: float, tau: float, sigma: float) -> tuple[float, float]:
    """Black-76 call/put USD premiums for a known forward ``f_star``."""
    call = black76_price(f_star, strike, tau, sigma, cp=1)
    put = black76_price(f_star, strike, tau, sigma, cp=-1)
    return call, put


def test_tier1_pcp_recovers_forward() -> None:
    """R2.1/R2.2: PCP median forward, method=pcp, n_pairs counted."""
    f_star = 45_500.0
    ts = datetime(2022, 4, 1, 8, 0, tzinfo=UTC)
    rows: list[tuple[float, str, float, datetime]] = []
    strikes = [40_000.0, 45_000.0, 50_000.0]
    for k in strikes:
        call, put = _cp_premiums(f_star, k, _TAU, 0.7)
        rows.append((k, "C", call, ts))
        rows.append((k, "P", put, ts))
    df = _options(rows)
    est = infer_forward_for_expiry(
        df.filter(pl.col("cp") == "C"),
        df.filter(pl.col("cp") == "P"),
        _empty_futures(),
        _empty_funding(),
        _SNAP,
        _TAU,
        45_000.0,
        _CFG,
        expiry=_EXPIRY,
    )
    assert est.method is ForwardMethod.PCP
    assert est.n_pairs == len(strikes)
    assert est.forward == pytest.approx(f_star, rel=1e-9)


def test_tier1_pcp_median_robust_to_one_bad_pair() -> None:
    """PCP median ignores a single corrupted pair (design robustness note)."""
    f_star = 45_500.0
    ts = datetime(2022, 4, 1, 8, 0, tzinfo=UTC)
    rows: list[tuple[float, str, float, datetime]] = []
    for k in (40_000.0, 45_000.0, 50_000.0):
        call, put = _cp_premiums(f_star, k, _TAU, 0.7)
        rows.append((k, "C", call, ts))
        rows.append((k, "P", put, ts))
    # Corrupt one strike's call premium so its parity forward is way off.
    rows.append((60_000.0, "C", 9_999.0, ts))
    rows.append((60_000.0, "P", 0.0, ts))
    df = _options(rows)
    est = infer_forward_for_expiry(
        df.filter(pl.col("cp") == "C"),
        df.filter(pl.col("cp") == "P"),
        _empty_futures(),
        _empty_funding(),
        _SNAP,
        _TAU,
        45_000.0,
        _CFG,
        expiry=_EXPIRY,
    )
    assert est.method is ForwardMethod.PCP
    # Median of the four parity forwards stays near f_star despite the outlier.
    assert est.forward == pytest.approx(f_star, abs=5_000.0)


def test_tier1_pair_window_excludes_wide_gap() -> None:
    """A C/P pair whose timestamps differ by > pcp_pair_window_s is dropped."""
    f_star = 45_500.0
    call, put = _cp_premiums(f_star, 45_000.0, _TAU, 0.7)
    rows = [
        (45_000.0, "C", call, datetime(2022, 4, 1, 8, 0, 0, tzinfo=UTC)),
        (45_000.0, "P", put, datetime(2022, 4, 1, 7, 0, 0, tzinfo=UTC)),  # 1h gap
    ]
    df = _options(rows)
    est = infer_forward_for_expiry(
        df.filter(pl.col("cp") == "C"),
        df.filter(pl.col("cp") == "P"),
        _empty_futures(),
        _empty_funding(),
        _SNAP,
        _TAU,
        45_000.0,
        _CFG,
        expiry=_EXPIRY,
    )
    # No usable pair -> falls through to Tier 3 (index_carry) here.
    assert est.method is ForwardMethod.INDEX_CARRY


# --- Tier 2: dated future -------------------------------------------------


def test_tier2_future_used_when_no_pcp() -> None:
    """R2.3: with no PCP pairs, the nearest same-expiry future price is used."""
    calls = _options([(45_000.0, "C", 1_000.0, datetime(2022, 4, 1, 8, 0, tzinfo=UTC))])
    puts = _empty_options()  # no puts -> no pairs
    futures = _futures(
        [
            ("BTC-8APR22", 44_800.0, datetime(2022, 4, 1, 7, 0, tzinfo=UTC)),
            ("BTC-8APR22", 45_100.0, datetime(2022, 4, 1, 8, 0, tzinfo=UTC)),  # nearest
            ("BTC-24JUN22", 46_000.0, datetime(2022, 4, 1, 8, 0, tzinfo=UTC)),  # wrong expiry
        ]
    )
    est = infer_forward_for_expiry(
        calls, puts, futures, _empty_funding(), _SNAP, _TAU, 45_000.0, _CFG, expiry=_EXPIRY
    )
    assert est.method is ForwardMethod.FUTURE
    assert est.n_pairs == 0
    assert est.forward == pytest.approx(45_100.0)


def test_tier2_future_ignores_post_snap_trades() -> None:
    """Leakage: a future trade after the snap is not used for the forward."""
    futures = _futures(
        [
            ("BTC-8APR22", 45_100.0, datetime(2022, 4, 1, 8, 0, tzinfo=UTC)),
            ("BTC-8APR22", 99_999.0, datetime(2022, 4, 1, 9, 0, tzinfo=UTC)),  # after snap
        ]
    )
    est = infer_forward_for_expiry(
        _empty_options(),
        _empty_options(),
        futures,
        _empty_funding(),
        _SNAP,
        _TAU,
        45_000.0,
        _CFG,
        expiry=_EXPIRY,
    )
    assert est.method is ForwardMethod.FUTURE
    assert est.forward == pytest.approx(45_100.0)


# --- Tier 3: index x carry ------------------------------------------------


def test_tier3_index_carry_fallthrough() -> None:
    """R2.4: no pairs and no future -> index * exp(carry * tau)."""
    interest_8h = 0.0001
    funding = _funding([(interest_8h, datetime(2022, 4, 1, 8, 0, tzinfo=UTC))])
    est = infer_forward_for_expiry(
        _empty_options(),
        _empty_options(),
        _empty_futures(),
        funding,
        _SNAP,
        _TAU,
        45_000.0,
        _CFG,
        expiry=_EXPIRY,
    )
    assert est.method is ForwardMethod.INDEX_CARRY
    assert est.n_pairs == 0
    carry = interest_8h * 3.0 * 365.0
    assert est.forward == pytest.approx(45_000.0 * math.exp(carry * _TAU))


def test_tier3_index_carry_zero_when_no_funding() -> None:
    """No funding observation -> carry 0 -> F collapses to the spot index."""
    est = infer_forward_for_expiry(
        _empty_options(),
        _empty_options(),
        _empty_futures(),
        _empty_funding(),
        _SNAP,
        _TAU,
        45_000.0,
        _CFG,
        expiry=_EXPIRY,
    )
    assert est.method is ForwardMethod.INDEX_CARRY
    assert est.forward == pytest.approx(45_000.0)


def test_forward_always_positive() -> None:
    """R2.5: every cell yields a strictly positive forward."""
    est = infer_forward_for_expiry(
        _empty_options(),
        _empty_options(),
        _empty_futures(),
        _empty_funding(),
        _SNAP,
        _TAU,
        45_000.0,
        _CFG,
        expiry=_EXPIRY,
    )
    assert est.forward > 0.0


# --- infer_forwards / attach_forward --------------------------------------


def test_infer_forwards_maps_each_expiry() -> None:
    """infer_forwards builds one estimate per distinct expiry."""
    exp2 = datetime(2022, 6, 24, 8, 0, tzinfo=UTC)
    ts = datetime(2022, 4, 1, 8, 0, tzinfo=UTC)
    df = pl.concat(
        [
            _options([(45_000.0, "C", 1.0, ts)], expiry=_EXPIRY),
            _options([(45_000.0, "C", 1.0, ts)], expiry=exp2),
        ]
    )
    fwds = infer_forwards(df, _empty_futures(), _empty_funding(), _SNAP, _CFG)
    assert set(fwds.keys()) == {_EXPIRY, exp2}


def test_attach_forward_adds_columns() -> None:
    """R2.7: attach_forward adds F, k, fwd_method columns."""
    ts = datetime(2022, 4, 1, 8, 0, tzinfo=UTC)
    df = _options([(50_000.0, "C", 1.0, ts), (40_000.0, "P", 1.0, ts)])
    fwds = infer_forwards(df, _empty_futures(), _empty_funding(), _SNAP, _CFG)
    out = attach_forward(df, fwds)
    assert {"F", "k", "fwd_method"} <= set(out.columns)
    assert out["F"].to_list() == pytest.approx([45_000.0, 45_000.0])
    assert out["fwd_method"][0] == "index_carry"


# --- Property CP1: put-call parity forward inference correctness -----------


@settings(max_examples=100, deadline=None)
@given(
    f_star=st.floats(min_value=5_000.0, max_value=150_000.0),
    n_strikes=st.integers(min_value=1, max_value=6),
    sigma=st.floats(min_value=0.2, max_value=2.0),
)
def test_cp1_pcp_recovers_known_forward(f_star: float, n_strikes: int, sigma: float) -> None:
    """CP1: parity forward from Black-76 premiums recovers F*.

    **Validates: Requirements 2.1, 2.2**
    """
    ts = datetime(2022, 4, 1, 8, 0, tzinfo=UTC)
    # Strikes bracketing the forward, all strictly positive and distinct.
    strikes = [f_star * (0.6 + 0.15 * i) for i in range(n_strikes)]
    rows: list[tuple[float, str, float, datetime]] = []
    for k in strikes:
        call, put = _cp_premiums(f_star, k, _TAU, sigma)
        rows.append((k, "C", call, ts))
        rows.append((k, "P", put, ts))
    df = _options(rows)
    est = infer_forward_for_expiry(
        df.filter(pl.col("cp") == "C"),
        df.filter(pl.col("cp") == "P"),
        _empty_futures(),
        _empty_funding(),
        _SNAP,
        _TAU,
        f_star,
        _CFG,
        expiry=_EXPIRY,
    )
    assert est.method is ForwardMethod.PCP
    assert est.n_pairs == len(strikes)
    assert est.forward == pytest.approx(f_star, rel=1e-6, abs=1e-6 * f_star)


# --- Property CP8: log-moneyness consistency -------------------------------


@settings(max_examples=200, deadline=None)
@given(
    strike=st.floats(min_value=1_000.0, max_value=200_000.0),
    forward=st.floats(min_value=1_000.0, max_value=200_000.0),
)
def test_cp8_log_moneyness_consistency(strike: float, forward: float) -> None:
    """CP8: k == ln(strike / F) and k > 0 iff strike > F.

    **Validates: Requirements 7.5**
    """
    ts = datetime(2022, 4, 1, 8, 0, tzinfo=UTC)
    df = _options([(strike, "C", 1.0, ts)])
    # Inject a synthetic forward directly to isolate the k computation.
    fwds = {_EXPIRY: ForwardEstimate(_EXPIRY, forward, ForwardMethod.FUTURE, 0)}
    out = attach_forward(df, fwds)
    k = out["k"][0]
    assert k == pytest.approx(math.log(strike / forward), rel=1e-12, abs=1e-12)
    assert (k > 0.0) == (strike > forward)
