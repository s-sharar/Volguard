"""Unit + property tests for the ``curate/filters.py`` cascade (design R4/R5).

Covers the Layer-1 quality filter cascade, MAD outlier rejection, and staleness
weighting (design Component 4), i.e. :func:`apply_filters`,
:func:`mad_outlier_mask`, and :func:`staleness_weight`:

- Property **CP3** — filter idempotence: ``apply_filters(apply_filters(rows))``
  reproduces the ``quality_flags`` / ``rejected`` columns of a single pass (R4.8).
- Property **CP4** — filter monotonicity: tightening any band never re-admits a
  row a looser band rejected, so ``survive(tight) ⊆ survive(loose)`` (R4.9).
- Property **CP5** — MAD preserves clean points: a tight per-expiry cluster
  flags nothing; a single injected extreme outlier flags exactly that value (R4.6).
- Property **CP9** — staleness weighting: ``1.0`` at zero, ``0.5`` at one
  half-life, strictly decreasing, and bounded in ``(0, 1]`` (R5.3-R5.5).
- Unit tests for the exact ``quality_flags`` value of each band case, the
  block-trade retained-but-flagged rule, and ``staleness_s`` / ``weight``
  correctness for a hand-built row (R4.1-R4.5, R4.7, R5.1-R5.4).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volguard.config import CurateConfig
from volguard.curate.filters import (
    QualityFlag,
    apply_filters,
    mad_outlier_mask,
    staleness_weight,
)

_TS = pl.Datetime(time_unit="ms", time_zone="UTC")
_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_EXPIRY = datetime(2022, 4, 8, 8, 0, tzinfo=UTC)
_CFG = CurateConfig()

# A few distinct expiries so generated frames exercise per-expiry MAD grouping.
_EXPIRIES = [
    datetime(2022, 4, 8, 8, 0, tzinfo=UTC),
    datetime(2022, 6, 24, 8, 0, tzinfo=UTC),
    datetime(2022, 9, 30, 8, 0, tzinfo=UTC),
]


# --- frame builders -------------------------------------------------------


def _one_row(
    *,
    forward: float = 45_000.0,
    strike: float = 45_000.0,
    tau: float = 0.1,
    iv_obs: float = 0.7,
    cp_sign: int = 1,
    size: float = 1.0,
    block_flag: bool = False,
    expiry: datetime = _EXPIRY,
    staleness_s: float = 0.0,
    quality_flags: int | None = None,
) -> pl.DataFrame:
    """Build a single post-cross-check row for :func:`apply_filters`.

    Defaults are an all-bands-OK ATM call; each unit test overrides exactly the
    field it means to push out of band so the resulting ``quality_flags`` is
    unambiguous.
    """
    source_ts = _SNAP - timedelta(seconds=staleness_s)
    data: dict[str, pl.Series] = {
        "F": pl.Series([forward], dtype=pl.Float64),
        "strike": pl.Series([strike], dtype=pl.Float64),
        "tau": pl.Series([tau], dtype=pl.Float64),
        "iv_obs": pl.Series([iv_obs], dtype=pl.Float64),
        "cp_sign": pl.Series([cp_sign], dtype=pl.Int64),
        "size": pl.Series([size], dtype=pl.Float64),
        "block_flag": pl.Series([block_flag], dtype=pl.Boolean),
        "expiry": pl.Series([expiry], dtype=_TS),
        "source_ts": pl.Series([source_ts], dtype=_TS),
    }
    df = pl.DataFrame(data)
    if quality_flags is not None:
        df = df.with_columns(pl.Series("quality_flags", [quality_flags], dtype=pl.Int64))
    return df


# A single generated row: (F, strike, tau, iv_obs, cp_sign, size, block_flag,
# expiry index, staleness seconds). All floats exclude nan/inf so the frame is
# well-formed; bands are still exercised (iv up to 6.0, size down to 0.0).
_row = st.tuples(
    st.floats(min_value=1_000.0, max_value=200_000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1_000.0, max_value=200_000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-3, max_value=3.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-3, max_value=6.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([1, -1]),
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.integers(min_value=0, max_value=len(_EXPIRIES) - 1),
    st.integers(min_value=0, max_value=6_000),
)


_RowTuple = tuple[float, float, float, float, int, float, bool, int, int]


def _frame(rows: list[_RowTuple]) -> pl.DataFrame:
    """Assemble a generated list of row tuples into an ``apply_filters`` frame."""
    return pl.DataFrame(
        {
            "F": pl.Series([r[0] for r in rows], dtype=pl.Float64),
            "strike": pl.Series([r[1] for r in rows], dtype=pl.Float64),
            "tau": pl.Series([r[2] for r in rows], dtype=pl.Float64),
            "iv_obs": pl.Series([r[3] for r in rows], dtype=pl.Float64),
            "cp_sign": pl.Series([r[4] for r in rows], dtype=pl.Int64),
            "size": pl.Series([r[5] for r in rows], dtype=pl.Float64),
            "block_flag": pl.Series([r[6] for r in rows], dtype=pl.Boolean),
            "expiry": pl.Series([_EXPIRIES[r[7]] for r in rows], dtype=_TS),
            "source_ts": pl.Series(
                [_SNAP - timedelta(seconds=r[8]) for r in rows], dtype=_TS
            ),
        }
    )


def _flag_set(value: int, flag: QualityFlag) -> bool:
    return bool(value & int(flag))


# --- Task 6.8: exact quality_flags per band case --------------------------


def test_ok_row_has_no_flags_and_is_kept() -> None:
    """R4.7: an all-bands-OK row carries OK flags and is not rejected."""
    out = apply_filters(_one_row(), _SNAP, _CFG)
    assert out["quality_flags"][0] == int(QualityFlag.OK)
    assert out["rejected"][0] is False


def test_tau_too_short_flag() -> None:
    """R4.1: tau < tau_min_years sets TAU_TOO_SHORT and rejects."""
    out = apply_filters(_one_row(tau=1e-3), _SNAP, _CFG)  # < 2/365 years
    assert out["quality_flags"][0] == int(QualityFlag.TAU_TOO_SHORT)
    assert out["rejected"][0] is True


def test_delta_out_of_band_flag() -> None:
    """R4.2: |delta| outside [delta_min, delta_max] sets DELTA_OUT_OF_BAND."""
    # Deep-OTM call (strike >> F): Black-76 delta ~ 0 < cfg.delta_min.
    out = apply_filters(_one_row(strike=250_000.0, tau=0.05, iv_obs=0.5), _SNAP, _CFG)
    assert out["quality_flags"][0] == int(QualityFlag.DELTA_OUT_OF_BAND)
    assert out["rejected"][0] is True


def test_iv_out_of_band_flag() -> None:
    """R4.3: iv_obs outside [iv_min, iv_max] sets IV_OUT_OF_BAND."""
    out = apply_filters(_one_row(iv_obs=0.005), _SNAP, _CFG)  # < iv_min = 0.01
    assert out["quality_flags"][0] == int(QualityFlag.IV_OUT_OF_BAND)
    assert out["rejected"][0] is True


def test_below_min_size_flag() -> None:
    """R4.4: size < min_size_btc sets BELOW_MIN_SIZE and rejects."""
    out = apply_filters(_one_row(size=0.05), _SNAP, _CFG)  # < min_size_btc = 0.1
    assert out["quality_flags"][0] == int(QualityFlag.BELOW_MIN_SIZE)
    assert out["rejected"][0] is True


def test_block_trade_retained_but_flagged() -> None:
    """R4.5: a block trade is flagged BLOCK_TRADE but retained (not rejected)."""
    out = apply_filters(_one_row(block_flag=True), _SNAP, _CFG)
    assert out["quality_flags"][0] == int(QualityFlag.BLOCK_TRADE)
    assert out["rejected"][0] is False


def test_prior_flags_preserved_and_ored() -> None:
    """R4.7: an inbound IV_DIVERGENCE flag is retained and ORed with new flags."""
    out = apply_filters(
        _one_row(size=0.05, quality_flags=int(QualityFlag.IV_DIVERGENCE)), _SNAP, _CFG
    )
    flags = out["quality_flags"][0]
    assert _flag_set(flags, QualityFlag.IV_DIVERGENCE)  # retained
    assert _flag_set(flags, QualityFlag.BELOW_MIN_SIZE)  # newly added
    assert out["rejected"][0] is True  # BELOW_MIN_SIZE is a hard reject


def test_staleness_and_weight_hand_built() -> None:
    """R5.1/R5.4: staleness_s = snap - source_ts, weight = 0.5 at one half-life."""
    out = apply_filters(_one_row(staleness_s=_CFG.recency_half_life_s), _SNAP, _CFG)
    assert out["staleness_s"][0] == pytest.approx(_CFG.recency_half_life_s)
    assert out["weight"][0] == pytest.approx(0.5)


def test_staleness_zero_weight_one() -> None:
    """R5.3: a fresh row (source_ts == snap_ts) has staleness 0 and weight 1.0."""
    out = apply_filters(_one_row(staleness_s=0.0), _SNAP, _CFG)
    assert out["staleness_s"][0] == pytest.approx(0.0)
    assert out["weight"][0] == pytest.approx(1.0)


# --- Task 6.4 / Property CP3: filter idempotence --------------------------


@settings(max_examples=100, deadline=None)
@given(rows=st.lists(_row, min_size=0, max_size=12))
def test_cp3_filter_idempotence(rows: list[_RowTuple]) -> None:
    """CP3: applying the cascade to its own output reproduces the result.

    ``apply_filters`` preserves every input row (nothing dropped), so the second
    pass sees the same rows plus the columns it will overwrite; the flags are
    pure per-row functions and the MAD statistic is computed over all rows, so
    ``quality_flags`` and ``rejected`` are unchanged by a repeat application.

    **Validates: Requirements 4.8**
    """
    once = apply_filters(_frame(rows), _SNAP, _CFG)
    twice = apply_filters(once, _SNAP, _CFG)
    assert once["quality_flags"].to_list() == twice["quality_flags"].to_list()
    assert once["rejected"].to_list() == twice["rejected"].to_list()


# --- Task 6.5 / Property CP4: filter monotonicity -------------------------


_LOOSE = CurateConfig(
    delta_min=0.005,
    delta_max=0.995,
    iv_min=0.005,
    iv_max=5.0,
    tau_min_days=1.0,
    min_size_btc=0.0,
)
_TIGHT = CurateConfig(
    delta_min=0.1,
    delta_max=0.9,
    iv_min=0.1,
    iv_max=3.0,
    tau_min_days=5.0,
    min_size_btc=0.5,
)


@settings(max_examples=100, deadline=None)
@given(rows=st.lists(_row, min_size=0, max_size=12))
def test_cp4_filter_monotonicity(rows: list[_RowTuple]) -> None:
    """CP4: tightening any band never re-admits a row the looser band rejected.

    Both configs share ``mad_multiplier`` so MAD rejection is identical; the
    tighter bands can only add band flags. Rows are preserved in order, so a row
    surviving under the tight config must also survive under the loose config:
    ``survive(tight) ⊆ survive(loose)``.

    **Validates: Requirements 4.9**
    """
    frame = _frame(rows)
    loose = apply_filters(frame, _SNAP, _LOOSE)
    tight = apply_filters(frame, _SNAP, _TIGHT)
    for loose_rejected, tight_rejected in zip(
        loose["rejected"].to_list(), tight["rejected"].to_list(), strict=True
    ):
        if not tight_rejected:  # survived the tight bands ...
            assert not loose_rejected  # ... so it must have survived the loose bands


# --- Task 6.6 / Property CP5: MAD preserves clean points ------------------


def _symmetric_cluster(base: float, spread: float) -> list[float]:
    """A tight 5-point cluster centred on ``base`` with spacing ``spread``.

    Median is ``base`` and MAD is ``spread``, so with the default multiplier of
    5 the threshold ``5*spread`` comfortably contains the max deviation ``2*spread``.
    """
    return [base - 2 * spread, base - spread, base, base + spread, base + 2 * spread]


@settings(max_examples=100, deadline=None)
@given(
    base=st.floats(min_value=0.2, max_value=2.0, allow_nan=False, allow_infinity=False),
    spread_frac=st.floats(min_value=1e-3, max_value=1e-2, allow_nan=False, allow_infinity=False),
)
def test_cp5_mad_no_outliers_flags_nothing(base: float, spread_frac: float) -> None:
    """CP5: a tight per-expiry cluster of w values flags no outliers.

    **Validates: Requirements 4.6**
    """
    spread = base * spread_frac
    w = pl.Series("w", _symmetric_cluster(base, spread), dtype=pl.Float64)
    expiry = pl.Series("expiry", [_EXPIRY] * w.len(), dtype=_TS)
    mask = mad_outlier_mask(w, expiry, _CFG.mad_multiplier)
    assert not any(mask.to_list())


@settings(max_examples=100, deadline=None)
@given(
    base=st.floats(min_value=0.2, max_value=2.0, allow_nan=False, allow_infinity=False),
    spread_frac=st.floats(min_value=1e-3, max_value=1e-2, allow_nan=False, allow_infinity=False),
)
def test_cp5_mad_single_outlier_flagged_exactly(base: float, spread_frac: float) -> None:
    """CP5: injecting one extreme value flags exactly that value, nothing else.

    **Validates: Requirements 4.6**
    """
    spread = base * spread_frac
    cluster = _symmetric_cluster(base, spread)
    outlier = base + 100.0 * spread  # far beyond multiplier * MAD
    values = [*cluster, outlier]
    w = pl.Series("w", values, dtype=pl.Float64)
    expiry = pl.Series("expiry", [_EXPIRY] * len(values), dtype=_TS)
    mask = mad_outlier_mask(w, expiry, _CFG.mad_multiplier).to_list()
    assert mask[-1] is True  # the injected outlier is flagged
    assert not any(mask[:-1])  # every clean cluster point is preserved


def test_cp5_mad_is_per_expiry() -> None:
    """CP5: MAD grouping is per expiry — an outlier in one expiry never flags another."""
    clean = _symmetric_cluster(0.5, 0.005)
    dirty = [*_symmetric_cluster(1.0, 0.01), 1.0 + 5.0]  # one extreme in expiry B
    values = clean + dirty
    expiries = [_EXPIRIES[0]] * len(clean) + [_EXPIRIES[1]] * len(dirty)
    w = pl.Series("w", values, dtype=pl.Float64)
    expiry = pl.Series("expiry", expiries, dtype=_TS)
    mask = mad_outlier_mask(w, expiry, _CFG.mad_multiplier).to_list()
    # Only the single injected value in expiry B is flagged.
    assert mask == [False] * (len(clean) + len(dirty) - 1) + [True]


# --- Task 6.7 / Property CP9: staleness weighting -------------------------


def test_cp9_weight_at_zero_is_one() -> None:
    """CP9/R5.3: staleness_weight(0, h) == 1.0."""
    assert staleness_weight(0.0, _CFG.recency_half_life_s) == pytest.approx(1.0)


def test_cp9_weight_at_one_half_life_is_half() -> None:
    """CP9/R5.4: staleness_weight(h, h) == 0.5."""
    h = _CFG.recency_half_life_s
    assert staleness_weight(h, h) == pytest.approx(0.5)


# The recency weight is ``exp(-ln2 * ratio)`` with ``ratio = staleness_s / half_life_s``.
# The operational regime keeps ``staleness_s <= max_window_minutes`` (6h) against a
# ~900s half-life, so ``ratio`` stays around 24; float64 ``exp`` only underflows to a
# literal 0.0 once ``ratio`` exceeds ~1074 (far outside any real snap window). We
# generate the ratio directly, bounded well above the operational max but within the
# range where the strict (0, 1] bound is observable in float64.
@settings(max_examples=200, deadline=None)
@given(
    ratio=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    half_life_s=st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
)
def test_cp9_weight_bounded_in_unit_interval(ratio: float, half_life_s: float) -> None:
    """CP9/R5.5: the recency weight is bounded in (0, 1] for non-negative staleness.

    **Validates: Requirements 5.5**
    """
    w = staleness_weight(ratio * half_life_s, half_life_s)
    assert 0.0 < w <= 1.0


@settings(max_examples=200, deadline=None)
@given(
    ratio1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ratio_delta=st.floats(min_value=1e-3, max_value=100.0, allow_nan=False, allow_infinity=False),
    half_life_s=st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
)
def test_cp9_weight_strictly_decreasing(
    ratio1: float, ratio_delta: float, half_life_s: float
) -> None:
    """CP9/R5.5: the weight strictly decreases as staleness_s increases.

    **Validates: Requirements 5.5**
    """
    w1 = staleness_weight(ratio1 * half_life_s, half_life_s)
    w2 = staleness_weight((ratio1 + ratio_delta) * half_life_s, half_life_s)
    assert w2 < w1


@settings(max_examples=200, deadline=None)
@given(
    half_life_s=st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False)
)
def test_cp9_weight_half_at_half_life(half_life_s: float) -> None:
    """CP9/R5.3-R5.4: weight is 1.0 at zero staleness and 0.5 at one half-life.

    **Validates: Requirements 5.3, 5.4**
    """
    assert staleness_weight(0.0, half_life_s) == pytest.approx(1.0)
    assert staleness_weight(half_life_s, half_life_s) == pytest.approx(0.5)
    # Consistency with the closed form.
    assert staleness_weight(2 * half_life_s, half_life_s) == pytest.approx(0.25)
    assert math.isclose(
        staleness_weight(half_life_s, half_life_s),
        math.exp(-math.log(2.0)),
    )
