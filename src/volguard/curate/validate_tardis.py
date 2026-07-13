"""Layer-1 Tardis free-day validation (design "Validation pass" / CP10, R9).

This module is the ``Golden_Validator`` from the M3 requirements: it compares
the curated ``quotes_norm`` implied vols against Tardis free-day *quote-based*
mark IVs, per ``(expiry, moneyness-bucket)``, within documented tolerances, and
does so **non-blockingly** — it validates only the free days that have actually
landed and skips + logs the rest, so it never depends on the in-progress full
Tardis backfill (R9.1-R9.3).

Two concerns live here (both pure apart from reading the Parquet/CSV paths they
are handed):

- :func:`curate_tardis_chain` — run a Tardis ``TARDIS_CHAIN`` frame through the
  *shared* curate code path (``normalize.canonical_from_tardis`` → forwards →
  IV cross-check → filters) to produce a ``QUOTES_NORM``-valid snapshot. This is
  the quote-based analogue of :func:`volguard.curate.pipeline.curate_one_snap`
  (which is trade/recency-window specific): the Tardis chain is a per-instant
  option-chain snapshot, so the trade-recency :mod:`~volguard.curate.window`
  stage is intentionally *not* applied; ``canonical_from_tardis`` already
  enforces the ``source_ts <= snap_ts`` leakage rule. Quote marks carry no
  traded ``size``, so a null ``size`` is coalesced to ``0.0`` to satisfy the
  frozen contract.

- :func:`compare_curated_vs_tardis` / :func:`validate_free_days` — bucket both
  the curated IVs and the Tardis mark IVs by ``(expiry, k-bucket)`` where
  ``k = ln(strike / F)``, and report a per-bucket absolute IV difference plus a
  within-tolerance pass/fail (CP10 / R9.1, R9.4). :func:`validate_free_days`
  drives this over a set of requested dates, loading whatever Tardis parts exist
  and skipping (with a logged coverage message) the ones that do not (R9.2/R9.3).

Tolerances (documented, R9.1): trade-based curated IVs and Tardis quote-based
mark IVs are two different measurements of the same surface, so they agree only
to a few vol points — tightest at the money, wider in the wings where a single
stale quote or a trade-vs-mid gap dominates. The default
:data:`DEFAULT_IV_TOLERANCE` is ``0.05`` (5 vol points), deliberately wider than
``CurateConfig.iv_divergence_tol`` (``0.02``, the *intra-source* trade-vs-recompute
tolerance) because this is a *cross-source* comparison; callers tighten it for
ATM-only checks or widen it for deep wings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from volguard.config import CurateConfig
from volguard.curate import filters, forwards, normalize
from volguard.curate.schemas import QUOTES_NORM, quotes_norm_schema, validate
from volguard.ingest.schemas import TARDIS_CHAIN
from volguard.ingest.schemas import validate as validate_raw

__all__ = [
    "DEFAULT_IV_TOLERANCE",
    "DEFAULT_K_BUCKET_WIDTH",
    "BucketComparison",
    "ValidationResult",
    "compare_curated_vs_tardis",
    "curate_tardis_chain",
    "validate_free_days",
]

logger = logging.getLogger(__name__)

# UTC-millisecond datetimes, identical to the raw/canonical-layer schemas so the
# empty futures/funding scaffolds line up with the forward-inference stage.
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")

# Cross-source IV agreement tolerance in *fraction* units (5 vol points). See the
# module docstring for why this is wider than ``cfg.iv_divergence_tol`` (0.02).
DEFAULT_IV_TOLERANCE: float = 0.05

# Log-moneyness bucket width for ``k = ln(strike / F)``: rows are grouped into
# fixed bins ``[floor(k / w) * w, +w)`` so a curated IV and a Tardis mark IV at a
# comparable moneyness are averaged together before differencing. 0.1 in ``k``
# is ~10% in ``strike/F`` — coarse enough that a bucket is rarely a single point,
# fine enough to keep ATM and wing buckets distinct.
DEFAULT_K_BUCKET_WIDTH: float = 0.1

# The exact, ordered ``quotes_norm`` output columns (from the frozen schema).
_OUTPUT_COLUMNS: tuple[str, ...] = tuple(QUOTES_NORM.columns.keys())


def _empty_futures() -> pl.DataFrame:
    """A zero-row ``TRADES_FUTURES``-shaped frame (Tardis has no dated futures)."""
    return pl.DataFrame(schema={"ts": _TS, "instrument": pl.String, "price": pl.Float64})


def _empty_funding() -> pl.DataFrame:
    """A zero-row ``FUNDING``-shaped frame (Tardis has no perp funding basis)."""
    return pl.DataFrame(schema={"ts": _TS, "interest_1h": pl.Float64, "interest_8h": pl.Float64})


def curate_tardis_chain(
    chain: pl.LazyFrame,
    snap_ts: datetime,
    cfg: CurateConfig,
) -> pl.DataFrame:
    """Curate a Tardis free-day chain into a ``QUOTES_NORM`` snapshot (CP10).

    Runs the Tardis ``TARDIS_CHAIN`` frame through the shared curate code path —
    :func:`normalize.canonical_from_tardis` (percent -> fraction IV, forward-free
    canonical rows, ``source_ts <= snap_ts`` leakage filter), then
    :func:`forwards.infer_forwards` / :func:`forwards.attach_forward` (per-expiry
    forward + ``k = ln(strike / F)``), :func:`filters.cross_check_iv`, and the
    :func:`filters.apply_filters` band/MAD/size cascade — and keeps only the
    non-rejected rows.

    Unlike :func:`volguard.curate.pipeline.curate_one_snap`, the trade-recency
    :mod:`~volguard.curate.window` stage is intentionally skipped: a Tardis chain
    is a single-instant quote snapshot, not a stream of trades, so there is no
    recency window to widen — every quote shares one ``source_ts``. Quote marks
    carry no traded ``size``, so a null ``size`` is coalesced to ``0.0`` before
    the frozen-contract validation (the ``BELOW_MIN_SIZE`` filter already treats a
    null ``size`` as "not below the minimum", so this only affects the output
    column, not filtering).

    The result is sorted by ``(expiry, strike, cp)`` so repeated runs on the same
    input are byte-stable (R9.5), then validated against ``QUOTES_NORM`` (R9.4).
    """
    canonical = normalize.canonical_from_tardis(chain, snap_ts, cfg)
    estimates = forwards.infer_forwards(canonical, _empty_futures(), _empty_funding(), snap_ts, cfg)
    rows = forwards.attach_forward(canonical, estimates)
    rows = filters.cross_check_iv(rows, cfg)
    rows = filters.apply_filters(rows, snap_ts, cfg)

    kept = rows.filter(~pl.col("rejected")).with_columns(
        pl.col("size").fill_null(0.0).alias("size")
    )
    out = kept.select(_OUTPUT_COLUMNS).sort(["expiry", "strike", "cp"])
    # Validate against the schema banded by *this* cfg (not the module-level
    # default), so a Tardis run with a loosened IV band in configs/curate.yaml
    # stays consistent between the filter stage and the boundary contract.
    return validate(out, quotes_norm_schema(cfg))


@dataclass(frozen=True, slots=True)
class BucketComparison:
    """One ``(expiry, k-bucket)`` curated-vs-Tardis IV comparison (design CP10).

    ``k_bucket`` is the bucket's lower edge (``floor(k / width) * width``);
    ``curated_iv`` / ``tardis_iv`` are the mean ``iv_obs`` / mark IV over the rows
    that fell in the bucket; ``abs_diff = |curated_iv - tardis_iv|`` and
    ``within_tol`` is ``abs_diff <= tolerance``.
    """

    expiry: datetime
    k_bucket: float
    n_curated: int
    n_tardis: int
    curated_iv: float
    tardis_iv: float
    abs_diff: float
    within_tol: bool


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Structured result of a single free-day curated-vs-Tardis comparison.

    ``passed`` is ``True`` iff every compared bucket is within tolerance; an empty
    ``buckets`` list (no overlapping ``(expiry, k-bucket)`` cells) is reported as
    ``passed`` vacuously — the caller decides whether "nothing to compare" is a
    skip (it is, in :func:`validate_free_days`).
    """

    tolerance: float
    bucket_width: float
    buckets: list[BucketComparison]

    @property
    def passed(self) -> bool:
        """True iff no compared bucket exceeds the tolerance."""
        return all(b.within_tol for b in self.buckets)

    @property
    def max_abs_diff(self) -> float:
        """Largest per-bucket absolute IV difference (``0.0`` when empty)."""
        return max((b.abs_diff for b in self.buckets), default=0.0)

    def to_frame(self) -> pl.DataFrame:
        """Render the per-bucket comparisons as a tidy Polars frame for QC."""
        return pl.DataFrame(
            {
                "expiry": pl.Series([b.expiry for b in self.buckets], dtype=_TS),
                "k_bucket": [b.k_bucket for b in self.buckets],
                "n_curated": [b.n_curated for b in self.buckets],
                "n_tardis": [b.n_tardis for b in self.buckets],
                "curated_iv": [b.curated_iv for b in self.buckets],
                "tardis_iv": [b.tardis_iv for b in self.buckets],
                "abs_diff": [b.abs_diff for b in self.buckets],
                "within_tol": [b.within_tol for b in self.buckets],
            }
        )


def _k_bucket(k: pl.Expr, width: float) -> pl.Expr:
    """Bucket log-moneyness ``k`` into fixed ``width``-wide bins (lower edge)."""
    return (k / width).floor() * width


def compare_curated_vs_tardis(
    curated: pl.DataFrame,
    tardis_chain: pl.LazyFrame,
    snap_ts: datetime,
    cfg: CurateConfig,
    *,
    tolerance: float = DEFAULT_IV_TOLERANCE,
    bucket_width: float = DEFAULT_K_BUCKET_WIDTH,
) -> ValidationResult:
    """Compare curated IVs to Tardis mark IVs per ``(expiry, k-bucket)`` (R9.1).

    ``curated`` is a ``QUOTES_NORM``-shaped frame (from the trade path *or* from
    :func:`curate_tardis_chain`); ``tardis_chain`` is the corresponding
    ``TARDIS_CHAIN`` free-day frame. The Tardis marks are normalized via
    :func:`normalize.canonical_from_tardis` (giving ``iv_trade`` = mark IV as a
    fraction, plus ``strike``/``expiry``) and assigned a log-moneyness ``k`` using
    the curated frame's per-expiry forward ``F`` (median), so both sides share one
    forward and one moneyness definition. Only expiries present in *both* frames
    are compared; a Tardis expiry with no curated forward is skipped.

    Both sides are bucketed by ``(expiry, floor(k / bucket_width) * bucket_width)``
    and the mean IV per bucket is differenced. Returns a :class:`ValidationResult`
    whose ``passed`` flag is ``True`` iff every overlapping bucket agrees within
    ``tolerance`` (design CP10 / R9.1, R9.4).
    """
    if curated.height == 0:
        return ValidationResult(tolerance, bucket_width, [])

    # One forward per expiry from the curated frame (median is robust to any
    # per-strike F noise); Tardis marks borrow it to define a comparable k.
    fwd = curated.group_by("expiry").agg(pl.col("F").median().alias("F"))

    marks = normalize.canonical_from_tardis(tardis_chain, snap_ts, cfg)
    if marks.height == 0:
        return ValidationResult(tolerance, bucket_width, [])

    marks = (
        marks.join(fwd, on="expiry", how="inner")
        .filter(pl.col("iv_trade").is_finite() & (pl.col("F") > 0.0) & (pl.col("strike") > 0.0))
        .with_columns((pl.col("strike") / pl.col("F")).log().alias("k"))
        .with_columns(_k_bucket(pl.col("k"), bucket_width).alias("k_bucket"))
    )

    cur = curated.with_columns(_k_bucket(pl.col("k"), bucket_width).alias("k_bucket"))

    cur_agg = cur.group_by(["expiry", "k_bucket"]).agg(
        pl.col("iv_obs").mean().alias("curated_iv"), pl.len().alias("n_curated")
    )
    mark_agg = marks.group_by(["expiry", "k_bucket"]).agg(
        pl.col("iv_trade").mean().alias("tardis_iv"), pl.len().alias("n_tardis")
    )

    joined = cur_agg.join(mark_agg, on=["expiry", "k_bucket"], how="inner").sort(
        ["expiry", "k_bucket"]
    )

    buckets: list[BucketComparison] = []
    for row in joined.iter_rows(named=True):
        curated_iv = float(row["curated_iv"])
        tardis_iv = float(row["tardis_iv"])
        abs_diff = abs(curated_iv - tardis_iv)
        buckets.append(
            BucketComparison(
                expiry=row["expiry"],
                k_bucket=float(row["k_bucket"]),
                n_curated=int(row["n_curated"]),
                n_tardis=int(row["n_tardis"]),
                curated_iv=curated_iv,
                tardis_iv=tardis_iv,
                abs_diff=abs_diff,
                within_tol=abs_diff <= tolerance,
            )
        )
    return ValidationResult(tolerance, bucket_width, buckets)


def _tardis_part_path(tardis_dir: Path, day: date) -> Path:
    """Location of a Tardis free day's Parquet part (matches the ingest layout)."""
    return tardis_dir / f"date={day.isoformat()}" / "part.parquet"


def _load_tardis_day(tardis_dir: Path, day: date) -> pl.LazyFrame | None:
    """Load one landed Tardis free day as a ``TARDIS_CHAIN`` frame, or ``None``.

    Returns ``None`` (rather than raising) when the day's Parquet part is absent,
    so the non-blocking driver can skip dates that have not landed yet (R9.3).
    """
    part = _tardis_part_path(tardis_dir, day)
    if not part.exists():
        return None
    frame = validate_raw(pl.read_parquet(part), TARDIS_CHAIN)
    return frame.lazy()


def validate_free_days(
    requested_dates: Iterable[date],
    tardis_dir: Path,
    curated_for_date: Callable[[date], pl.DataFrame | None],
    snap_ts_for_date: Callable[[date], datetime],
    cfg: CurateConfig,
    *,
    tolerance: float = DEFAULT_IV_TOLERANCE,
    bucket_width: float = DEFAULT_K_BUCKET_WIDTH,
) -> dict[date, ValidationResult]:
    """Validate curated IVs vs Tardis marks for whatever free days have landed.

    Non-blocking and gracefully degrading (R9.2/R9.3): for each requested date,

    - if no Tardis free-day Parquet has landed under ``tardis_dir`` for that date,
      the date is **skipped** with a logged coverage message (never raised);
    - if there is no curated ``quotes_norm`` for that date (``curated_for_date``
      returns ``None`` / an empty frame), the date is likewise skipped and logged;
    - otherwise the date is compared via :func:`compare_curated_vs_tardis` and its
      :class:`ValidationResult` is included in the returned mapping.

    The full Tardis backfill is therefore *not* a dependency — only the dates that
    have both curated output and a landed Tardis part are validated. ``curated_for_date``
    and ``snap_ts_for_date`` are injected so the caller controls where curated
    frames come from (in-memory, a Parquet scan, ...) and which snap instant a
    date maps to, keeping this driver free of hidden I/O.
    """
    results: dict[date, ValidationResult] = {}
    for day in requested_dates:
        chain = _load_tardis_day(tardis_dir, day)
        if chain is None:
            logger.info(
                "tardis validation: no free-day data landed for %s under %s; skipping",
                day.isoformat(),
                tardis_dir,
            )
            continue
        curated = curated_for_date(day)
        if curated is None or curated.height == 0:
            logger.info(
                "tardis validation: no curated quotes_norm for %s; skipping", day.isoformat()
            )
            continue
        result = compare_curated_vs_tardis(
            curated,
            chain,
            snap_ts_for_date(day),
            cfg,
            tolerance=tolerance,
            bucket_width=bucket_width,
        )
        results[day] = result
        if not result.passed:
            logger.warning(
                "tardis validation %s: %d/%d buckets exceed tol=%.3f (max abs diff %.4f)",
                day.isoformat(),
                sum(1 for b in result.buckets if not b.within_tol),
                len(result.buckets),
                tolerance,
                result.max_abs_diff,
            )
    return results
