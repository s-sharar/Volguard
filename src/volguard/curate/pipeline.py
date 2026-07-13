"""Layer-1 orchestration driver: raw Parquet -> ``curated/quotes_norm``.

This is design **Component 5** — the non-pure glue that wires the four pure
stage functions into one per-snap transform and drives it over a date range:

    normalize -> build_snap_window -> forwards -> cross_check_iv -> apply_filters
    -> leakage assert -> QUOTES_NORM validate -> date-partitioned Parquet

Everything reusable lives in the pure stage modules (:mod:`~volguard.curate.normalize`,
:mod:`~volguard.curate.window`, :mod:`~volguard.curate.forwards`,
:mod:`~volguard.curate.filters`) and the frozen contract
(:mod:`~volguard.curate.schemas`); this module never reimplements a stage or the
M1 ``blackiv`` math. The stage modules are referenced by module object (not by
imported symbol) so the per-snap composition stays easy to follow and to
instrument in tests.

Two public entry points (design interface):

- :func:`curate_one_snap` — the (pure-ish) per-snap composition. It asserts the
  no-leakage rule (``max(source_ts) <= snap_ts``, R8.1-R8.3) *before* dropping
  ``source_ts`` for the output, validates the result against ``QUOTES_NORM`` at
  the stage boundary (R10.1/R10.2, letting pandera raise loudly and name the
  offending column), logs a coverage warning for any empty ``(snap, expiry)``
  cell (R6.5/R10.3), and reports the ``index_carry`` forward-method fallback
  count (R10.4).
- :func:`run_curate` — reads the raw date-partitioned Parquet for a range, calls
  :func:`curate_one_snap` per snap day, and writes ``curated/quotes_norm`` as
  date-partitioned Parquet (R7.7, R11.1). Days with no raw data are logged and
  skipped rather than crashing.

Conventions match the raw/canonical layers: timestamps are UTC-millisecond
``Datetime`` (``_TS``); ``snap_ts`` is pinned to ``cfg.snap_hour_utc:cfg.snap_minute_utc``
UTC of each snap day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from volguard.config import CurateConfig, DataConfig
from volguard.curate import filters, forwards, normalize, window
from volguard.curate.forwards import ForwardMethod
from volguard.curate.schemas import QUOTES_NORM, quotes_norm_schema, validate

__all__ = ["RawInputs", "curate_one_snap", "run_curate"]

logger = logging.getLogger(__name__)

# UTC-millisecond datetimes, identical to the raw/canonical-layer schemas so all
# ``snap_ts`` / ``source_ts`` arithmetic stays in one dtype (design Model 2).
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")

# The exact, ordered ``curated/quotes_norm`` output columns (design Model 2);
# taken from the frozen schema so this driver and the contract never drift.
_OUTPUT_COLUMNS: tuple[str, ...] = tuple(QUOTES_NORM.columns.keys())


@dataclass(frozen=True, slots=True)
class RawInputs:
    """Driver-internal bundle of the raw Layer-0 lazy scans (design Model 3).

    Each field is a :class:`polars.LazyFrame` over a validated raw table so the
    driver can push per-day filters down to the partitioned Parquet scan before
    materializing:

    - ``options``     — ``TRADES_OPTIONS`` (Deribit option trades).
    - ``futures``     — ``TRADES_FUTURES`` (dated-future trades, Tier-2 forward).
    - ``funding``     — ``FUNDING`` (perp basis, Tier-3 carry).
    - ``instruments`` — ``INSTRUMENTS`` (``ref/instruments`` reference).
    """

    options: pl.LazyFrame
    futures: pl.LazyFrame
    funding: pl.LazyFrame
    instruments: pl.LazyFrame


def _snap_instant(day: datetime, cfg: CurateConfig) -> datetime:
    """Pin ``day`` to the configured snap time (08:05 UTC) of its own day."""
    aware = day.replace(tzinfo=UTC) if day.tzinfo is None else day.astimezone(UTC)
    return aware.replace(
        hour=cfg.snap_hour_utc, minute=cfg.snap_minute_utc, second=0, microsecond=0
    )


def curate_one_snap(snap_ts: datetime, raw: RawInputs, cfg: CurateConfig) -> pl.DataFrame:
    """Curate one daily snapshot into a ``QUOTES_NORM``-valid frame.

    Composes the four pure stages for a single 08:05-UTC snap (design sequence
    diagram): normalize the raw option trades into canonical rows, select the
    leakage-safe snap window (with sparse widening + recency weights), infer one
    forward per expiry and attach ``F`` / ``k`` / ``fwd_method``, recompute and
    cross-check IV, then run the band/MAD/size/block filter cascade and keep only
    the non-rejected rows.

    Preconditions:
      - ``raw`` bundles the per-day raw lazy scans (``options`` at least
        non-empty for a productive snap); ``snap_ts`` is the tz-aware snap
        instant (pinned to ``cfg.snap_hour_utc:cfg.snap_minute_utc`` UTC).
      - ``cfg`` is a validated :class:`~volguard.config.CurateConfig`.

    Postconditions / gates:
      - **Leakage (R8.1-R8.3)**: asserts ``max(source_ts) <= snap_ts`` across the
        kept rows *before* ``source_ts`` is dropped for output; a breach raises
        ``ValueError`` so the snap fails and nothing is written.
      - **Schema (R10.1/R10.2)**: the returned frame is validated against
        ``QUOTES_NORM`` at the stage boundary; a violating computed column makes
        pandera raise loudly and name the offending column.
      - **Coverage (R6.5/R10.3)**: any expiry present in the snap window but with
        zero surviving rows is logged as a coverage warning (and simply emits no
        rows for that cell).
      - **Provenance (R10.4)**: the count of expiries whose forward fell through
        to ``index_carry`` is logged for QC visibility.

    Returns the coerced ``QUOTES_NORM`` frame (exactly :data:`_OUTPUT_COLUMNS`).
    """
    canonical = normalize.normalize_trades(raw.options, raw.instruments, snap_ts, cfg)
    windowed = window.build_snap_window(canonical, snap_ts, cfg)

    futures = raw.futures.collect()
    funding = raw.funding.collect()
    estimates = forwards.infer_forwards(windowed, futures, funding, snap_ts, cfg)

    # R10.4: report the index_carry fallback count for QC visibility.
    index_carry = sum(1 for est in estimates.values() if est.method is ForwardMethod.INDEX_CARRY)
    if index_carry:
        logger.info(
            "snap %s: %d/%d expiries used index_carry forward fallback",
            snap_ts.isoformat(),
            index_carry,
            len(estimates),
        )

    rows = forwards.attach_forward(windowed, estimates)
    rows = filters.cross_check_iv(rows, cfg)
    rows = filters.apply_filters(rows, snap_ts, cfg)

    # Keep only non-rejected rows for the curated surface (filters retain every
    # input row + a ``rejected`` flag so the stage stays fully auditable, R4.7).
    kept = rows.filter(~pl.col("rejected"))

    # R6.5/R10.3: warn on any windowed expiry that produced zero surviving rows.
    _log_coverage_gaps(snap_ts, windowed, kept)

    # R8.1-R8.3 leakage gate: assert while source_ts is still present, before it
    # is dropped from the output. A breach fails the snap and writes nothing.
    if kept.height:
        max_source = kept["source_ts"].max()
        if isinstance(max_source, datetime) and max_source > snap_ts:
            msg = (
                f"leakage: max(source_ts)={max_source!r} > snap_ts={snap_ts!r} "
                f"for {kept.height} rows; failing snap without writing"
            )
            raise ValueError(msg)

    out = kept.select(_OUTPUT_COLUMNS)
    # R10.1/R10.2: pandera boundary validation — raises loudly, names the column.
    # Validate against the schema banded by *this* cfg (not the module-level
    # default), so loosening the IV band in configs/curate.yaml keeps the filter
    # stage and the boundary contract consistent (a row apply_filters admits
    # under cfg.iv_max must not then be rejected by a default-banded schema).
    return validate(out, quotes_norm_schema(cfg))


def _log_coverage_gaps(snap_ts: datetime, windowed: pl.DataFrame, kept: pl.DataFrame) -> None:
    """Log a coverage warning for each windowed expiry with zero kept rows (R6.5)."""
    if windowed.height == 0:
        return
    windowed_expiries = set(windowed["expiry"].unique().to_list())
    kept_expiries = set(kept["expiry"].unique().to_list()) if kept.height else set()
    for expiry in sorted(windowed_expiries - kept_expiries):
        logger.warning(
            "snap %s: coverage gap — expiry %s produced zero curated rows after filtering",
            snap_ts.isoformat(),
            expiry.isoformat() if isinstance(expiry, datetime) else expiry,
        )


# --- run_curate driver (read raw -> loop days -> write partitions) ---------


def _iter_days(start: date, end: date) -> list[date]:
    """Every calendar day in ``[start, end]`` inclusive (one snap per day)."""
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _scan_table(directory: Path) -> pl.LazyFrame | None:
    """Lazily scan a raw table directory's Parquet parts, or ``None`` if absent.

    Hive partitioning is disabled so the ``month=`` / ``instrument=`` partition
    directories in the raw layout are treated as plain file paths (their values
    are not needed downstream and would otherwise appear as extra columns).
    """
    if not directory.exists():
        return None
    parts = list(directory.rglob("*.parquet"))
    if not parts:
        return None
    return pl.scan_parquet(parts, hive_partitioning=False)


def _empty_like(lazy: pl.LazyFrame | None) -> pl.LazyFrame:
    """A raw scan or an empty ``ts``-only frame, so optional inputs never crash.

    The empty fallback carries a ``ts`` column so the per-day ``ts <= snap_ts``
    push-down in :func:`_day_inputs` stays valid when a raw table (futures /
    funding) has not landed yet; the stage functions short-circuit on the
    resulting zero-height frame.
    """
    return lazy if lazy is not None else pl.LazyFrame(schema={"ts": _TS})


def _day_inputs(raw: RawInputs, snap_ts: datetime, cfg: CurateConfig) -> RawInputs:
    """Push per-day ``ts`` filters down onto the raw lazy scans (leakage-safe).

    Options are narrowed to the widest possible snap window
    ``(snap_ts - max_window_minutes, snap_ts]`` (the window builder narrows
    further); futures and funding are narrowed to ``ts <= snap_ts`` so only
    leakage-safe context reaches the forward tiers.
    """
    lo = snap_ts - timedelta(minutes=cfg.max_window_minutes)
    options = raw.options.filter(
        (pl.col("ts") > pl.lit(lo).cast(_TS)) & (pl.col("ts") <= pl.lit(snap_ts).cast(_TS))
    )
    futures = raw.futures.filter(pl.col("ts") <= pl.lit(snap_ts).cast(_TS))
    funding = raw.funding.filter(pl.col("ts") <= pl.lit(snap_ts).cast(_TS))
    return RawInputs(options, futures, funding, raw.instruments)


def _write_partition(df: pl.DataFrame, path: Path) -> None:
    """Write one date partition as zstd Parquet (matching the ingest idiom)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")


def run_curate(
    cfg: CurateConfig,
    data_cfg: DataConfig,
    start: str | None,
    end: str | None,
) -> None:
    """Run Layer-1 curation over a date range into ``curated/quotes_norm``.

    Reads the raw date/month-partitioned Parquet tables (``trades_options``,
    ``trades_futures``, ``funding``, ``ref/instruments``), iterates one snap per
    day in ``[start, end]`` (defaulting to ``data_cfg.history_start`` .. today),
    calls :func:`curate_one_snap` per day, and writes each non-empty result to
    ``curated/quotes_norm/date=YYYY-MM-DD/part.parquet`` (R7.7, R11.1).

    Days with no raw option trades (or a snap that produces no curated rows) are
    logged and skipped rather than crashing, so an incomplete backfill degrades
    gracefully. The no-leakage and ``QUOTES_NORM`` gates live in
    :func:`curate_one_snap`; a failing snap raises there and no partition is
    written for that day.
    """
    start_date = date.fromisoformat(start) if start else date.fromisoformat(data_cfg.history_start)
    end_date = date.fromisoformat(end) if end else datetime.now(UTC).date()

    options = _scan_table(data_cfg.raw_table_dir("trades_options"))
    if options is None:
        logger.warning(
            "curate: no raw trades_options under %s; nothing to curate",
            data_cfg.raw_table_dir("trades_options"),
        )
        return

    raw = RawInputs(
        options=options,
        futures=_empty_like(_scan_table(data_cfg.raw_table_dir("trades_futures"))),
        funding=_empty_like(_scan_table(data_cfg.raw_table_dir("funding"))),
        instruments=_empty_like(_scan_table(data_cfg.ref_dir / "instruments")),
    )

    out_dir = data_cfg.curated_dir / "quotes_norm"
    written = 0
    for day in _iter_days(start_date, end_date):
        snap_ts = _snap_instant(datetime(day.year, day.month, day.day, tzinfo=UTC), cfg)
        day_raw = _day_inputs(raw, snap_ts, cfg)
        frame = curate_one_snap(snap_ts, day_raw, cfg)
        if frame.height == 0:
            logger.info("curate %s: no curated rows; skipping partition", day.isoformat())
            continue
        _write_partition(frame, out_dir / f"date={day.isoformat()}" / "part.parquet")
        written += 1
        logger.info("curate %s: wrote %d rows", day.isoformat(), frame.height)
    logger.info("curate: wrote %d daily partitions to %s", written, out_dir)
