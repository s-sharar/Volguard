"""Layer-1 snap-window selection with staleness-aware sparse widening.

This module owns the Window_Builder concern (design Component / "Snap-time
window model" and the "Algorithm: staleness-aware snap window" pseudocode):
select the per-trade observations that feed a single 08:05-UTC snapshot from a
base ``(snap_ts - window_minutes, snap_ts]`` window, widening the window
*backward in time only* while any expiry is sparse, then attach a recency
weight to every surviving row.

The single public entry point is :func:`build_snap_window`. It is a pure
Polars-frame-in / frame-out function (config passed explicitly, no I/O), matching
the other stage modules (:mod:`volguard.curate.normalize`,
:mod:`volguard.curate.forwards`, :mod:`volguard.curate.filters`).

Conventions match the canonical frame emitted by
:func:`volguard.curate.normalize.normalize_trades`: ``source_ts`` and the caller
``snap_ts`` are UTC-millisecond ``Datetime`` (``_TS``), and ``expiry`` groups
the sparsity check. The recency weight is *not* reimplemented here — it reuses
:func:`volguard.curate.filters.staleness_weight`
(``exp(-ln(2) * staleness_s / half_life_s)``, design R5.2).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from volguard.config import CurateConfig
from volguard.curate.filters import staleness_weight

__all__ = ["build_snap_window"]

# UTC-millisecond datetimes, identical to the canonical / raw-layer schemas so
# ``snap_ts`` / ``source_ts`` arithmetic stays in one dtype (design Model 2).
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")


def _select(trades: pl.DataFrame, snap_ts: datetime, window_minutes: int) -> pl.DataFrame:
    """Rows in the half-open window ``(snap_ts - window_minutes, snap_ts]``.

    The lower bound is strict and the upper bound is inclusive (design R6.1),
    so a row is selected iff ``lo < source_ts <= snap_ts``. The upper bound is
    always ``snap_ts`` regardless of ``window_minutes``: widening only moves
    ``lo`` backward, never past ``snap_ts`` (the leakage rule, R6.4 / R8.1).
    """
    lo = snap_ts - timedelta(minutes=window_minutes)
    return trades.filter(
        (pl.col("source_ts") > pl.lit(lo).cast(_TS))
        & (pl.col("source_ts") <= pl.lit(snap_ts).cast(_TS))
    )


def _any_expiry_sparse(sel: pl.DataFrame, min_trades_per_expiry: int) -> bool:
    """True iff some expiry present in ``sel`` has ``< min_trades_per_expiry`` rows.

    Only expiries that actually appear in the selection are considered; an
    expiry with zero trades is not "sparse" here (it has no rows to widen
    toward), and is handled downstream as a coverage gap by the driver (R6.5).
    An empty selection therefore reports *not* sparse, which stops the widening
    loop immediately.
    """
    if sel.height == 0:
        return False
    counts = sel.group_by("expiry").agg(pl.len().alias("_n"))
    return bool((counts["_n"] < min_trades_per_expiry).any())


def build_snap_window(
    trades: pl.DataFrame,
    snap_ts: datetime,
    cfg: CurateConfig,
) -> pl.DataFrame:
    """Select the snap window with sparse widening and attach recency weights.

    Preconditions:
      - ``trades`` is a canonical per-trade frame carrying at least ``expiry``
        and ``source_ts`` (UTC-ms ``Datetime``); other columns are carried
        through untouched.
      - ``snap_ts`` is the tz-aware 08:05-UTC snap instant.
      - ``cfg`` is a validated :class:`~volguard.config.CurateConfig`
        (``window_minutes <= max_window_minutes``, ``min_trades_per_expiry >= 1``,
        ``recency_half_life_s > 0``).

    Behaviour (design "Algorithm: staleness-aware snap window", R6.1-R6.4):
      - Start from the base window ``(snap_ts - window_minutes, snap_ts]``.
      - WHILE any expiry in the selection has fewer than
        ``cfg.min_trades_per_expiry`` trades AND the window has not reached
        ``cfg.max_window_minutes``, extend the lower bound backward by
        ``cfg.widen_step_minutes`` (capped so the window never looks back
        further than ``cfg.max_window_minutes``).
      - The window only ever extends *backward*; the upper bound stays
        ``snap_ts``, so every returned row has ``source_ts <= snap_ts`` (R6.4).

    Postconditions:
      - Every returned row satisfies ``source_ts <= snap_ts`` (leakage-safe,
        R8.1) and ``0 <= staleness_s <= max_window_minutes * 60`` (R8.4).
      - ``weight`` is in ``(0, 1]`` — the reused
        :func:`~volguard.curate.filters.staleness_weight`
        (``exp(-ln(2) * staleness_s / half_life_s)``, R5.2), ``1.0`` at zero
        staleness and strictly decreasing in ``staleness_s``.
      - The returned frame is ``trades``'s columns plus ``staleness_s`` and
        ``weight``.

    Loop invariant / termination: the window strictly increases each iteration
    (by ``widen_step_minutes``, clamped to ``max_window_minutes``) and is bounded
    above by ``max_window_minutes``, so the loop always terminates. A
    non-positive ``widen_step_minutes`` cannot grow the window, so the loop
    stops rather than spinning.
    """
    window = cfg.window_minutes
    sel = _select(trades, snap_ts, window)

    # Widen backward while sparse, capped at max_window_minutes (R6.2, R6.3).
    while _any_expiry_sparse(sel, cfg.min_trades_per_expiry) and window < cfg.max_window_minutes:
        if cfg.widen_step_minutes <= 0:
            break  # cannot widen further; guarantees termination
        window = min(window + cfg.widen_step_minutes, cfg.max_window_minutes)
        sel = _select(trades, snap_ts, window)

    # staleness_s = (snap_ts - source_ts) in seconds, >= 0 by construction
    # since every selected row has source_ts <= snap_ts (design R5.1 / R8.4).
    staleness_s = (
        (pl.lit(snap_ts).cast(_TS) - pl.col("source_ts")).dt.total_microseconds().cast(pl.Float64)
        / 1_000_000.0
    )
    sel = sel.with_columns(staleness_s.alias("staleness_s"))

    # Recency weight reuses the trusted filters helper (never reimplemented).
    half_life = cfg.recency_half_life_s
    return sel.with_columns(
        pl.col("staleness_s")
        .map_elements(
            lambda s: staleness_weight(s, half_life),
            return_dtype=pl.Float64,
        )
        .alias("weight")
    )
