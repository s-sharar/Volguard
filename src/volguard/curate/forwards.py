"""Layer-1 forward inference: one forward ``F`` per ``(snap_ts, expiry)``.

This module implements design Component 2 — a three-tier fallback that infers a
single forward price for every expiry, records which method produced it, and
attaches ``F`` and log-moneyness ``k = ln(strike / F)`` back onto the canonical
per-trade rows emitted by :mod:`volguard.curate.normalize`.

Tiers (highest-trust first; design R2.1-R2.4):

1. **Put-call parity (PCP)** — for near-simultaneous same-strike call/put
   trades (``source_ts`` gap ``<= cfg.pcp_pair_window_s``) compute the
   undiscounted parity forward ``F = K + (usd_call - usd_put)`` (``r = 0`` so
   ``disc = 1``) and take the **median** across pairs for robustness. Chosen
   when at least ``cfg.min_pcp_pairs`` valid pairs exist.
2. **Dated future** — otherwise use the same-expiry dated-future price nearest
   (at or before) the snap from ``raw/trades_futures``.
3. **Index x carry** — otherwise ``F = index_price * exp(carry * tau)`` where
   ``carry`` is implied from the perp funding basis in ``raw/funding``.

The forward is guaranteed ``> 0`` for every cell (design R2.5): Tier 3 always
yields a positive forward because ``index_price > 0``, so PCP/future results are
only accepted when strictly positive and otherwise fall through.

Conventions match the trusted M1 core: premiums are USD (``usd_premium`` from
:mod:`~volguard.curate.normalize`), ``tau`` is in years, and timestamps are
UTC-aware millisecond ``Datetime`` (leakage-safe: only ``source_ts <= snap_ts``
rows reach here, and futures/funding lookups are pinned to ``ts <= snap_ts``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import polars as pl

from volguard.config import CurateConfig
from volguard.ingest.schemas import parse_instrument

__all__ = [
    "ForwardEstimate",
    "ForwardMethod",
    "attach_forward",
    "infer_forward_for_expiry",
    "infer_forwards",
]

logger = logging.getLogger(__name__)

_TS = pl.Datetime(time_unit="ms", time_zone="UTC")

# Deribit perp funding is quoted as an 8-hour interest rate; there are three
# 8-hour periods per day, so the annualized carry is ``interest_8h * 3 * 365``.
# The 1-hour rate (24 periods/day) is the fallback when the 8h rate is absent.
_EIGHT_HOUR_PERIODS_PER_YEAR = 3.0 * 365.0
_ONE_HOUR_PERIODS_PER_YEAR = 24.0 * 365.0


def _as_float(value: object) -> float:
    """Coerce a Polars scalar (``PythonLiteral | None``) to ``float``.

    ``None`` (empty aggregation) maps to ``nan`` so callers can guard on
    ``math.isfinite`` rather than juggling ``Optional`` at every call site.
    """
    if value is None:
        return math.nan
    return float(value)  # type: ignore[arg-type]


class ForwardMethod(StrEnum):
    """Provenance label for an inferred forward (design R2.6).

    The string values are exactly the ``fwd_method`` domain enforced by the
    ``QUOTES_NORM`` schema: ``{"pcp", "future", "index_carry"}``.
    """

    PCP = "pcp"
    FUTURE = "future"
    INDEX_CARRY = "index_carry"


@dataclass(frozen=True, slots=True)
class ForwardEstimate:
    """One inferred forward for a single ``(snap_ts, expiry)`` cell.

    ``n_pairs`` is the number of put-call-parity pairs used; it is ``> 0`` iff
    ``method is ForwardMethod.PCP`` (the fallbacks record ``0``).
    """

    expiry: datetime
    forward: float
    method: ForwardMethod
    n_pairs: int


def _seconds_to_snap(snap_ts: datetime) -> pl.Expr:
    """Non-negative seconds from a row's ``source_ts`` back to the snap.

    Every canonical row satisfies ``source_ts <= snap_ts`` (the leakage rule),
    so this is ``>= 0`` and orders rows by recency (smallest = nearest snap).
    """
    delta_us = (pl.lit(snap_ts).cast(_TS) - pl.col("source_ts")).dt.total_microseconds()
    return delta_us.cast(pl.Float64) / 1_000_000.0


def _nearest_by_strike(trades: pl.DataFrame, snap_ts: datetime) -> pl.DataFrame:
    """Collapse trades to the one nearest the snap per strike.

    Returns columns ``strike``, ``source_ts``, ``usd_premium`` (one row per
    strike), picking the trade with the smallest ``source_ts`` gap to the snap.
    """
    if trades.height == 0:
        return pl.DataFrame(
            schema={"strike": pl.Float64, "source_ts": _TS, "usd_premium": pl.Float64}
        )
    return (
        trades.with_columns(_seconds_to_snap(snap_ts).alias("_gap_s"))
        .sort("_gap_s")
        .group_by("strike", maintain_order=True)
        .first()
        .select("strike", "source_ts", "usd_premium")
    )


def _pcp_forward(
    calls: pl.DataFrame, puts: pl.DataFrame, snap_ts: datetime, cfg: CurateConfig
) -> tuple[float, int]:
    """Tier 1: median parity forward and the pair count (design R2.1, R2.2).

    Pairs the nearest-to-snap call and put at each shared strike, keeps only
    pairs whose ``source_ts`` differ by ``<= cfg.pcp_pair_window_s``, and
    returns ``(median(K + usd_call - usd_put), n_pairs)``. ``n_pairs == 0``
    means no usable pair was found (caller falls through).
    """
    calls_n = _nearest_by_strike(calls, snap_ts)
    puts_n = _nearest_by_strike(puts, snap_ts)
    if calls_n.height == 0 or puts_n.height == 0:
        return math.nan, 0

    paired = calls_n.join(puts_n, on="strike", suffix="_put")
    if paired.height == 0:
        return math.nan, 0

    gap_us = (pl.col("source_ts") - pl.col("source_ts_put")).dt.total_microseconds().abs().cast(
        pl.Float64
    ) / 1_000_000.0
    paired = paired.with_columns(gap_us.alias("_pair_gap_s")).filter(
        pl.col("_pair_gap_s") <= cfg.pcp_pair_window_s
    )
    n_pairs = paired.height
    if n_pairs == 0:
        return math.nan, 0

    # F = K + (usd_call - usd_put); r = 0 => discount factor is 1.
    forwards = paired.select(
        (pl.col("strike") + pl.col("usd_premium") - pl.col("usd_premium_put")).alias("_fwd")
    )["_fwd"]
    return _as_float(forwards.median()), n_pairs


def _nearest_future(futures: pl.DataFrame, expiry: datetime, snap_ts: datetime) -> float:
    """Tier 2: same-expiry dated-future price nearest (at/before) the snap.

    Parses each future's instrument name to match ``expiry``, keeps rows with
    ``ts <= snap_ts`` (leakage-safe), and returns the price of the latest such
    trade. Returns ``nan`` when no matching, positive-priced future exists.
    """
    if futures.height == 0:
        return math.nan
    names = futures["instrument"].to_list()
    match = [i for i, name in enumerate(names) if _safe_future_expiry(name) == expiry]
    if not match:
        return math.nan
    sel = (
        futures[match]
        .filter(pl.col("ts").cast(_TS) <= pl.lit(snap_ts).cast(_TS))
        .filter(pl.col("price").is_not_null() & (pl.col("price") > 0.0))
        .sort("ts")
    )
    if sel.height == 0:
        return math.nan
    return float(sel["price"][-1])


def _safe_future_expiry(name: str) -> datetime | None:
    """Parsed expiry for a dated-future instrument name, else ``None``."""
    try:
        parsed = parse_instrument(name)
    except ValueError:
        return None
    if parsed.kind != "future":
        return None
    return parsed.expiry


def _implied_carry(funding: pl.DataFrame, snap_ts: datetime) -> float:
    """Annualized carry implied from the perp funding basis (design R2.4).

    Uses the funding row nearest (at/before) the snap: the 8-hour interest rate
    annualized over three 8h periods per day, falling back to the 1-hour rate.
    Returns ``0.0`` when no funding observation is available (so ``F`` collapses
    to the spot index, still positive).
    """
    if funding.height == 0:
        return 0.0
    sel = funding.filter(pl.col("ts").cast(_TS) <= pl.lit(snap_ts).cast(_TS)).sort("ts")
    if sel.height == 0:
        return 0.0
    row = sel.row(-1, named=True)
    interest_8h = row.get("interest_8h")
    if interest_8h is not None and math.isfinite(interest_8h):
        return float(interest_8h) * _EIGHT_HOUR_PERIODS_PER_YEAR
    interest_1h = row.get("interest_1h")
    if interest_1h is not None and math.isfinite(interest_1h):
        return float(interest_1h) * _ONE_HOUR_PERIODS_PER_YEAR
    return 0.0


def infer_forward_for_expiry(
    calls: pl.DataFrame,
    puts: pl.DataFrame,
    futures: pl.DataFrame,
    funding: pl.DataFrame,
    snap_ts: datetime,
    tau: float,
    index_price: float,
    cfg: CurateConfig,
    *,
    expiry: datetime | None = None,
) -> ForwardEstimate:
    """Infer one forward for a single expiry via the three-tier fallback.

    Preconditions: ``tau > 0``, ``index_price > 0``, and all ``calls``/``puts``
    rows share one expiry (taken from the rows, or the explicit ``expiry``).

    Postconditions (design R2.5): ``forward > 0`` and
    ``method is ForwardMethod.PCP`` iff ``n_pairs >= cfg.min_pcp_pairs``.
    """
    resolved_expiry = expiry if expiry is not None else _expiry_of(calls, puts)

    # Tier 1: put-call parity on near-simultaneous same-strike pairs.
    pcp_fwd, n_pairs = _pcp_forward(calls, puts, snap_ts, cfg)
    if n_pairs >= cfg.min_pcp_pairs and math.isfinite(pcp_fwd) and pcp_fwd > 0.0:
        logger.debug(
            "forward expiry=%s method=pcp n_pairs=%d F=%.2f", resolved_expiry, n_pairs, pcp_fwd
        )
        return ForwardEstimate(resolved_expiry, pcp_fwd, ForwardMethod.PCP, n_pairs)

    # Tier 2: same-expiry dated future nearest the snap.
    fut_price = _nearest_future(futures, resolved_expiry, snap_ts)
    if math.isfinite(fut_price) and fut_price > 0.0:
        logger.debug("forward expiry=%s method=future F=%.2f", resolved_expiry, fut_price)
        return ForwardEstimate(resolved_expiry, fut_price, ForwardMethod.FUTURE, 0)

    # Tier 3: index x carry from the perp funding basis.
    carry = _implied_carry(funding, snap_ts)
    fwd = index_price * math.exp(carry * tau)
    logger.debug(
        "forward expiry=%s method=index_carry carry=%.6f F=%.2f", resolved_expiry, carry, fwd
    )
    return ForwardEstimate(resolved_expiry, fwd, ForwardMethod.INDEX_CARRY, 0)


def _expiry_of(calls: pl.DataFrame, puts: pl.DataFrame) -> datetime:
    """The single expiry shared by ``calls``/``puts`` (whichever is non-empty)."""
    for frame in (calls, puts):
        if frame.height > 0:
            return frame["expiry"][0]
    raise ValueError("infer_forward_for_expiry requires at least one call or put row")


def infer_forwards(
    canonical: pl.DataFrame,
    futures: pl.DataFrame,
    funding: pl.DataFrame,
    snap_ts: datetime,
    cfg: CurateConfig,
) -> dict[datetime, ForwardEstimate]:
    """Infer a forward for every expiry in ``canonical`` (design R2.5).

    Loops the distinct expiries, splits each into calls/puts, derives a
    representative ``tau`` and ``index_price`` (median across the expiry's rows),
    and delegates to :func:`infer_forward_for_expiry`. Returns the mapping
    ``{expiry -> ForwardEstimate}`` consumed by :func:`attach_forward`.
    """
    forwards: dict[datetime, ForwardEstimate] = {}
    if canonical.height == 0:
        return forwards

    for (expiry,) in canonical.select("expiry").unique().sort("expiry").iter_rows():
        rows = canonical.filter(pl.col("expiry") == expiry)
        calls = rows.filter(pl.col("cp") == "C")
        puts = rows.filter(pl.col("cp") == "P")
        tau = _as_float(rows["tau"].median())
        index_price = _as_float(rows["index_price"].median())
        forwards[expiry] = infer_forward_for_expiry(
            calls, puts, futures, funding, snap_ts, tau, index_price, cfg, expiry=expiry
        )
    return forwards


def attach_forward(
    canonical: pl.DataFrame,
    forwards: dict[datetime, ForwardEstimate],
) -> pl.DataFrame:
    """Attach ``F``, ``k = ln(strike / F)``, and ``fwd_method`` columns (R2.7).

    Joins each canonical row to its expiry's :class:`ForwardEstimate`; the
    resulting ``k`` is finite with ``k > 0`` iff ``strike > F`` (design CP8).
    """
    mapping = pl.DataFrame(
        {
            "expiry": pl.Series([e for e in forwards], dtype=_TS),
            "F": pl.Series([est.forward for est in forwards.values()], dtype=pl.Float64),
            "fwd_method": pl.Series(
                [str(est.method) for est in forwards.values()], dtype=pl.String
            ),
        }
    )
    return canonical.join(mapping, on="expiry", how="left").with_columns(
        (pl.col("strike") / pl.col("F")).log().alias("k")
    )
