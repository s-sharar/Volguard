"""Unit + property tests for ``curate/window.py`` (design Window_Builder).

Covers the staleness-aware snap-window builder :func:`build_snap_window`
(design "Snap-time window model" / "Algorithm: staleness-aware snap window"):

- Property **CP6** — no leakage: for randomly generated trade frames with
  ``source_ts`` straddling the snap across several expiries, every returned row
  has ``source_ts <= snap_ts`` (equivalently ``staleness_s >= 0``), ``weight``
  in ``(0, 1]``, and no post-snap row is ever included (R6.4, R8.1, R8.4).
- Unit tests for widening behaviour: a dense base window stays put, a sparse
  base window widens step-by-step until each expiry meets the threshold, and
  widening stops at ``max_window_minutes`` even when still sparse (R6.2, R6.3).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from volguard.config import CurateConfig
from volguard.curate.window import build_snap_window

_TS = pl.Datetime(time_unit="ms", time_zone="UTC")
_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_EXPIRIES = [
    datetime(2022, 4, 8, 8, 0, tzinfo=UTC),
    datetime(2022, 6, 24, 8, 0, tzinfo=UTC),
    datetime(2022, 9, 30, 8, 0, tzinfo=UTC),
]


# --- frame builders -------------------------------------------------------


def _trades(rows: list[tuple[datetime, datetime]]) -> pl.DataFrame:
    """Build a minimal canonical trade frame: (expiry, source_ts) per row."""
    return pl.DataFrame(
        {
            "expiry": pl.Series([r[0] for r in rows], dtype=_TS),
            "source_ts": pl.Series([r[1] for r in rows], dtype=_TS),
        }
    )


def _at(
    minutes_before_snap: float, *, expiry: datetime = _EXPIRIES[0]
) -> tuple[datetime, datetime]:
    """A trade row ``minutes_before_snap`` minutes before the snap for ``expiry``."""
    return (expiry, _SNAP - timedelta(minutes=minutes_before_snap))


# --- Task 8.3: widening behaviour -----------------------------------------


def test_dense_base_window_is_not_widened() -> None:
    """R6.2: a dense base window excludes rows older than the base window.

    Each expiry already has >= min_trades_per_expiry trades inside
    ``[07:05, 08:05]`` (60-minute base), so no widening happens and a trade
    from 90 minutes before the snap (before the base window) is excluded.
    """
    cfg = CurateConfig(window_minutes=60, min_trades_per_expiry=3, widen_step_minutes=60)
    rows: list[tuple[datetime, datetime]] = []
    for expiry in (_EXPIRIES[0], _EXPIRIES[1]):
        # Three fresh trades per expiry inside the base window.
        rows += [_at(m, expiry=expiry) for m in (5.0, 20.0, 40.0)]
    # One old trade per expiry, well before the base window.
    rows += [_at(90.0, expiry=_EXPIRIES[0]), _at(90.0, expiry=_EXPIRIES[1])]

    out = build_snap_window(_trades(rows), _SNAP, cfg)

    # Only the six in-window trades survive; the two 90-min-old rows are excluded.
    assert out.height == 6
    assert bool((out["staleness_s"] <= 60.0 * 60.0).all())


def test_empty_base_window_widens_to_reach_trades() -> None:
    """R6.2: an empty base window still widens to recover deeper trades.

    A snap with zero trades in the base ``[07:05, 08:05]`` window but usable
    trades further back within ``max_window_minutes`` must reach back for them
    rather than emit nothing (the empty base window is itself the sparse case
    the widening is meant to recover).
    """
    cfg = CurateConfig(
        window_minutes=60,
        widen_step_minutes=60,
        max_window_minutes=360,
        min_trades_per_expiry=3,
    )
    # Nothing in the first 60 minutes; three trades sit 90-150 min before snap.
    rows = [_at(m, expiry=_EXPIRIES[0]) for m in (90.0, 120.0, 150.0)]

    out = build_snap_window(_trades(rows), _SNAP, cfg)

    # Widened past the empty base window to admit all three deeper trades.
    assert out.height == 3
    assert bool((out["staleness_s"] <= 180.0 * 60.0).all())


def test_wholly_empty_max_window_returns_empty() -> None:
    """R6.3/R6.5: an empty max window stops at the cap and returns no rows.

    When even the widest window has no trades, widening halts at the cap and the
    builder emits an empty frame (the driver logs the coverage gap downstream)
    rather than looping forever.
    """
    cfg = CurateConfig(window_minutes=60, widen_step_minutes=60, max_window_minutes=120)
    # The only trade is older than the 120-minute cap.
    rows = [_at(200.0, expiry=_EXPIRIES[0])]

    out = build_snap_window(_trades(rows), _SNAP, cfg)

    assert out.height == 0


def test_sparse_base_window_widens_step_by_step() -> None:
    """R6.2: a sparse base window widens backward to admit older trades.

    The base 60-minute window holds only one trade per expiry (< threshold of
    3), so the builder widens in 60-minute steps and pulls in the older trades
    until each expiry meets the threshold.
    """
    cfg = CurateConfig(
        window_minutes=60,
        widen_step_minutes=60,
        max_window_minutes=360,
        min_trades_per_expiry=3,
    )
    rows: list[tuple[datetime, datetime]] = []
    for expiry in (_EXPIRIES[0], _EXPIRIES[1]):
        # One trade in the base window, two more further back (at 90 and 150 min).
        rows += [
            _at(10.0, expiry=expiry),
            _at(90.0, expiry=expiry),
            _at(150.0, expiry=expiry),
        ]

    out = build_snap_window(_trades(rows), _SNAP, cfg)

    # Widened to 180 minutes so every trade is admitted and each expiry has 3.
    assert out.height == 6
    counts = out.group_by("expiry").agg(pl.len().alias("n"))
    assert counts["n"].to_list() == [3, 3]
    assert bool((out["staleness_s"] <= 180.0 * 60.0).all())


def test_widening_stops_at_max_window() -> None:
    """R6.3: widening is capped at max_window_minutes even when still sparse.

    The threshold can never be met (only two trades per expiry exist), so the
    builder widens up to the 120-minute cap and stops; a trade older than the
    cap is excluded rather than looping forever.
    """
    cfg = CurateConfig(
        window_minutes=60,
        widen_step_minutes=60,
        max_window_minutes=120,
        min_trades_per_expiry=5,
    )
    rows = [
        _at(10.0, expiry=_EXPIRIES[0]),  # in base window
        _at(90.0, expiry=_EXPIRIES[0]),  # admitted after one widen step (<= 120)
        _at(200.0, expiry=_EXPIRIES[0]),  # older than the 120-min cap -> excluded
    ]

    out = build_snap_window(_trades(rows), _SNAP, cfg)

    assert out.height == 2
    # Nothing older than the max window survives.
    assert bool((out["staleness_s"] <= 120.0 * 60.0).all())


# --- Task 8.2 / Property CP6: no leakage -----------------------------------

# A generated trade: (expiry index, signed offset seconds from snap). Positive
# offsets are BEFORE the snap; negative offsets are AFTER the snap (must never
# be admitted). The range spans the widening horizon and beyond on both sides.
_trade = st.tuples(
    st.integers(min_value=0, max_value=len(_EXPIRIES) - 1),
    st.integers(min_value=-3_600, max_value=8 * 3_600),
)

_TradeTuple = tuple[int, int]


def _frame(rows: list[_TradeTuple]) -> pl.DataFrame:
    """Assemble generated (expiry index, offset seconds) tuples into a frame."""
    return pl.DataFrame(
        {
            "expiry": pl.Series([_EXPIRIES[r[0]] for r in rows], dtype=_TS),
            "source_ts": pl.Series([_SNAP - timedelta(seconds=r[1]) for r in rows], dtype=_TS),
        }
    )


@settings(max_examples=200, deadline=None)
@given(rows=st.lists(_trade, min_size=0, max_size=20))
def test_cp6_no_leakage(rows: list[_TradeTuple]) -> None:
    """CP6: every windowed row is leakage-safe with a weight in (0, 1].

    Every returned row has ``source_ts <= snap_ts`` (equivalently
    ``staleness_s >= 0``), a ``weight`` in ``(0, 1]``, and no post-snap row is
    ever admitted; the window only ever extends backward.

    **Validates: Requirements 6.4, 8.1, 8.4**
    """
    cfg = CurateConfig()
    out = build_snap_window(_frame(rows), _SNAP, cfg)

    snap_series = pl.Series([_SNAP] * out.height, dtype=_TS)
    # No leakage: no returned row is dated after the snap.
    assert bool((out["source_ts"] <= snap_series).all())
    # staleness_s >= 0 for every row (equivalent leakage statement, R8.4).
    assert bool((out["staleness_s"] >= 0.0).all())
    # Recency weight strictly positive and bounded by 1.0 (R5.5).
    if out.height:
        assert bool((out["weight"] > 0.0).all())
        assert bool((out["weight"] <= 1.0).all())
    # Post-snap rows are never included.
    post_snap = [r for r in rows if r[1] < 0]
    if post_snap:
        assert bool((out["source_ts"] <= snap_series).all())
