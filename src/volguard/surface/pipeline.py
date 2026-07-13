"""Layer-2 orchestration driver: ``curated/quotes_norm`` -> ``curated/surfaces_daily``.

This is design **Component 1** — the non-pure glue that reads the canonical M3
observation table, groups each daily 08:05-UTC snap into per-expiry observation
bundles, drives the pure surface stages, and writes the date-partitioned
Layer-2 output:

    load_snap -> fit_surface -> sample_grid -> surface_arb_metrics
    -> long-format diagnostics assembly -> SURFACES_DAILY validate -> Parquet

Everything reusable lives in the pure stage modules
(:mod:`~volguard.surface.calendar_fit`, :mod:`~volguard.surface.grid`,
:mod:`~volguard.surface.arbitrage_metrics`) and the M1 math core
(:mod:`~volguard.surface.svi`, :mod:`~volguard.curate.blackiv`); this module
never reimplements a stage. Conventions mirror the M3 driver
(:mod:`~volguard.curate.pipeline`): UTC-millisecond ``Datetime`` timestamps
(``_TS``), the output column order taken from the frozen schema
(``_OUTPUT_COLUMNS``), ``rglob`` Parquet scans, ``mkdir`` + zstd partition
writes, a date range defaulting to ``data_cfg.history_start`` .. today, and
graceful skip-and-log for missing input data.

Three public entry points (design interface):

- :func:`load_snap` — read one snap's ``quotes_norm`` rows and group them by
  ``expiry`` into vega-weighted :class:`~volguard.surface.types.ExpiryObs`
  bundles, dropping expiries below ``cfg.min_obs_slice`` with a coverage warning
  and ordering by increasing ``tau`` (Requirement 1).
- :func:`build_one_surface` — wire the pure stages into a ``SURFACES_DAILY``-valid
  long frame plus a ``SURFACE_QC`` row, validating at the stage boundary and
  skipping snaps below ``cfg.min_expiries_per_snap`` (Requirements 7, 9.1-9.3).
- :func:`run_build_surfaces` — loop a date range, write one
  ``surfaces_daily/date=YYYY-MM-DD/part.parquet`` per productive snap, and skip
  days with no landed input (Requirements 7.6, 9.4, 11.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from volguard.config import DataConfig, SurfaceConfig
from volguard.curate.blackiv import black76_vega

# qc.py only imports schemas (no cycle back to this driver), so a top-level
# import is safe and keeps the QC wiring explicit (design Component 7).
from volguard.surface import qc
from volguard.surface.arbitrage import ArbitrageReport
from volguard.surface.arbitrage_metrics import surface_arb_metrics
from volguard.surface.calendar_fit import FitMethod, SurfaceFit, fit_surface
from volguard.surface.grid import SurfaceGrid, sample_grid
from volguard.surface.schemas import SURFACES_DAILY, validate
from volguard.surface.types import ExpiryObs

__all__ = ["SurfaceResult", "build_one_surface", "load_snap", "run_build_surfaces"]

logger = logging.getLogger(__name__)

# UTC-millisecond datetimes, identical to the curated-layer schemas (M2/M3), so
# ``expiry`` round-trips through Parquet in one dtype (design Model 2).
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")

_DAYS_PER_YEAR = 365.0

# The exact, ordered ``curated/surfaces_daily`` output columns (design Model 2),
# taken from the frozen schema so this driver and the contract never drift.
_OUTPUT_COLUMNS: tuple[str, ...] = tuple(SURFACES_DAILY.columns.keys())

# Explicit Polars dtypes for the long-format output frame. Kept aligned with
# ``_OUTPUT_COLUMNS`` (asserted at import) so all-null param/grid columns are
# well-typed before the pandera boundary coerces them.
_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "snap_date": pl.Date(),
    "expiry": _TS,
    "tau": pl.Float64(),
    "record_kind": pl.String(),
    "svi_a": pl.Float64(),
    "svi_b": pl.Float64(),
    "svi_rho": pl.Float64(),
    "svi_m": pl.Float64(),
    "svi_sigma": pl.Float64(),
    "fit_method": pl.String(),
    "moneyness": pl.Float64(),
    "grid_k": pl.Float64(),
    "grid_w": pl.Float64(),
    "cell_n_obs": pl.Int64(),
    "interp_flag": pl.Boolean(),
    "rmse": pl.Float64(),
    "n_obs": pl.Int64(),
    "vega_sum": pl.Float64(),
    "butterfly_ok": pl.Boolean(),
    "calendar_ok": pl.Boolean(),
    "arb_butterfly_pre": pl.Int64(),
    "arb_calendar_pre": pl.Int64(),
    "arb_butterfly_post": pl.Int64(),
    "arb_calendar_post": pl.Int64(),
}

# Guard: the explicit output schema must cover exactly the frozen contract's
# columns (so a schema change in surface/schemas.py fails loudly here, not with
# a confusing downstream coercion error).
assert set(_OUTPUT_SCHEMA) == set(_OUTPUT_COLUMNS), (
    "surface/pipeline._OUTPUT_SCHEMA drifted from SURFACES_DAILY.columns"
)


@dataclass(frozen=True, slots=True)
class SurfaceResult:
    """Driver-internal bundle for one built snap (design Model 4).

    ``rows`` is the ``SURFACES_DAILY``-valid long frame (coerced by the boundary
    validation) ready to be written as a date partition; ``qc`` is the single
    ``SURFACE_QC`` row (assembled here, written by the task-8 QC dashboard).
    """

    snap_date: date
    fit: SurfaceFit
    grid: SurfaceGrid
    arb_pre: ArbitrageReport  # market's own violations (raw obs binned to fixed-k)
    arb_post: ArbitrageReport  # fitted-surface violations (fitted slices on fixed-k)
    rows: pl.DataFrame  # SURFACES_DAILY-valid long frame
    qc: dict[str, object]  # one SURFACE_QC row


# --- Task 7.1: load + group a snap's curated observations ------------------


def _expiry_obs(expiry: datetime, group: pl.DataFrame) -> ExpiryObs:
    """Build one :class:`ExpiryObs` bundle from an expiry's grouped rows.

    ``tau`` is constant within an expiry (M3 derives it from ``snap_ts`` and the
    expiry), so the first value is taken; the vega weight per observation is the
    M1 Black-76 vega ``black76_vega(F, K, tau, iv_obs)`` (Requirement 1.2),
    reused, never reimplemented.
    """
    tau = float(group["tau"][0])
    k = np.asarray(group["k"].to_numpy(), dtype=float)
    iv = np.asarray(group["iv_obs"].to_numpy(), dtype=float)
    forwards = group["F"].to_list()
    strikes = group["strike"].to_list()
    ivs = group["iv_obs"].to_list()
    vega = np.asarray(
        [
            black76_vega(float(f), float(strike), tau, float(sigma))
            for f, strike, sigma in zip(forwards, strikes, ivs, strict=True)
        ],
        dtype=float,
    )
    return ExpiryObs(expiry=expiry, tau=tau, k=k, iv=iv, vega=vega, n_obs=group.height)


def load_snap(quotes_norm: pl.LazyFrame, snap_date: date, cfg: SurfaceConfig) -> list[ExpiryObs]:
    """Group one snap's ``quotes_norm`` rows into per-expiry observation bundles.

    Reads the snap's slice of ``curated/quotes_norm`` (the caller passes that
    day's partition scan; a defensive ``snap_ts`` date filter is applied when the
    column is present), groups by ``expiry`` into
    ``(tau, k, iv_obs, vega, n_obs)`` bundles with M1 vega weights, drops
    expiries with fewer than ``cfg.min_obs_slice`` observations (logging a
    coverage warning), and returns the bundles ordered by increasing ``tau``
    (Requirement 1.1-1.5).

    Hard-rejected rows are **already excluded** from ``quotes_norm`` by the M3
    filter cascade (only ``~rejected`` rows are written), so every row here is
    surface-eligible and no extra ``quality_flags`` filter is applied
    (Requirement 1.3): filtering on ``quality_flags == 0`` would wrongly drop
    informative-but-retained rows such as ``BLOCK_TRADE`` / ``IV_DIVERGENCE``.
    """
    df = quotes_norm.collect()

    # Defensive: if the caller handed a wider frame, keep only this snap's rows.
    if "snap_ts" in df.columns and df.height:
        df = df.filter(pl.col("snap_ts").dt.date() == snap_date)

    if df.height == 0:
        logger.warning("snap %s: no curated observations to load", snap_date.isoformat())
        return []

    bundles: list[ExpiryObs] = []
    for (expiry,), group in df.group_by("expiry", maintain_order=True):
        obs = _expiry_obs(expiry, group)  # type: ignore[arg-type]
        if obs.n_obs < cfg.min_obs_slice:
            logger.warning(
                "snap %s: coverage gap — expiry %s has %d obs (< min_obs_slice=%d); excluded",
                snap_date.isoformat(),
                obs.expiry.isoformat() if isinstance(obs.expiry, datetime) else obs.expiry,
                obs.n_obs,
                cfg.min_obs_slice,
            )
            continue
        bundles.append(obs)

    bundles.sort(key=lambda o: o.tau)  # strictly increasing tau (Requirement 1.5)
    return bundles


# --- Task 7.2: assemble one snap's SURFACES_DAILY frame + QC row -----------


def _param_records(
    snap_date: date, fit: SurfaceFit, arb: dict[str, int]
) -> list[dict[str, object]]:
    """One ``record_kind == "param"`` row per fitted slice (grid fields null)."""
    records: list[dict[str, object]] = []
    for s in fit.slices:
        a, b, rho, m, sigma = s.params.as_tuple()
        records.append(
            {
                "snap_date": snap_date,
                "expiry": s.expiry,
                "tau": s.tau,
                "record_kind": "param",
                "svi_a": a,
                "svi_b": b,
                "svi_rho": rho,
                "svi_m": m,
                "svi_sigma": sigma,
                "fit_method": s.method.value,
                "moneyness": None,
                "grid_k": None,
                "grid_w": None,
                "cell_n_obs": None,
                "interp_flag": None,
                "rmse": s.rmse,
                "n_obs": s.n_obs,
                "vega_sum": s.vega_sum,
                "butterfly_ok": s.butterfly_ok,
                "calendar_ok": s.calendar_ok,
                **arb,
            }
        )
    return records


def _grid_records(
    snap_date: date, grid: SurfaceGrid, arb: dict[str, int]
) -> list[dict[str, object]]:
    """Exactly ``n_tenor * n_money`` ``record_kind == "grid"`` rows (param fields null).

    ``expiry`` is null for grid tenors (interpolated across the term structure),
    and ``tau`` is the output tenor in years (``tenor_days / 365`` > 0, so the
    schema's ``tau > 0`` invariant holds; Requirement 7.6).
    """
    records: list[dict[str, object]] = []
    n_tenor = len(grid.tenors_days)
    n_money = len(grid.moneyness)
    for j in range(n_tenor):
        tau_years = float(grid.tenors_days[j]) / _DAYS_PER_YEAR
        for i in range(n_money):
            records.append(
                {
                    "snap_date": snap_date,
                    "expiry": None,
                    "tau": tau_years,
                    "record_kind": "grid",
                    "svi_a": None,
                    "svi_b": None,
                    "svi_rho": None,
                    "svi_m": None,
                    "svi_sigma": None,
                    "fit_method": None,
                    "moneyness": float(grid.moneyness[i]),
                    "grid_k": float(grid.k_grid[j, i]),
                    "grid_w": float(grid.w[j, i]),
                    "cell_n_obs": int(grid.n_obs[j, i]),
                    "interp_flag": bool(grid.interp_flag[j, i]),
                    "rmse": None,
                    "n_obs": None,
                    "vega_sum": None,
                    "butterfly_ok": None,
                    "calendar_ok": None,
                    **arb,
                }
            )
    return records


def _arb_columns(arb_pre: ArbitrageReport, arb_post: ArbitrageReport) -> dict[str, int]:
    """Snap-level arb metrics repeated identically across every row (Requirement 7.3)."""
    return {
        "arb_butterfly_pre": arb_pre.butterfly_violations,
        "arb_calendar_pre": arb_pre.calendar_violations,
        "arb_butterfly_post": arb_post.butterfly_violations,
        "arb_calendar_post": arb_post.calendar_violations,
    }


def _market_arb_rate(n_expiries: int, arb_pre: ArbitrageReport, cfg: SurfaceConfig) -> float:
    """Market's own arbitrage-violation base rate for the snap, in ``[0, 1]``.

    Defined as the fraction of shared fixed-k-grid no-arbitrage *checks* that the
    raw market observations violate pre-fit: butterfly is checked per tenor and
    calendar per adjacent tenor pair, so the denominator is
    ``n_expiries * arb_check_points`` (butterfly) plus
    ``(n_expiries - 1) * arb_check_points`` (calendar), and the numerator is
    ``arb_pre.butterfly_violations + arb_pre.calendar_violations`` (Requirement
    8.2). The per-slice violation counts never exceed these denominators, and the
    result is clamped to ``[0, 1]`` for safety.
    """
    points = cfg.arb_check_points
    denom = n_expiries * points + max(0, n_expiries - 1) * points
    if denom == 0:
        return 0.0
    numer = arb_pre.butterfly_violations + arb_pre.calendar_violations
    return min(1.0, numer / denom)


def _qc_row(
    snap_date: date,
    fit: SurfaceFit,
    arb_pre: ArbitrageReport,
    arb_post: ArbitrageReport,
    cfg: SurfaceConfig,
) -> dict[str, object]:
    """Assemble the single ``SURFACE_QC`` row for the snap (design Model 3)."""
    rmses = [s.rmse for s in fit.slices]
    return {
        "snap_date": snap_date,
        "n_expiries": len(fit.slices),
        "mean_rmse": float(np.mean(rmses)),
        "max_rmse": float(np.max(rmses)),
        "n_svi": sum(1 for s in fit.slices if s.method is FitMethod.SVI),
        "n_ssvi": sum(1 for s in fit.slices if s.method is FitMethod.SSVI),
        "market_arb_rate": _market_arb_rate(len(fit.slices), arb_pre, cfg),
        **_arb_columns(arb_pre, arb_post),
        "total_n_obs": sum(s.n_obs for s in fit.slices),
    }


def build_one_surface(
    snap_date: date, obs: list[ExpiryObs], cfg: SurfaceConfig
) -> SurfaceResult | None:
    """Build one snap's surface: fit -> grid -> arb -> validated long frame + QC.

    Returns ``None`` (writing nothing) when the snap has fewer than
    ``cfg.min_expiries_per_snap`` fittable expiries, logging a coverage warning
    (Requirement 9.3). Otherwise fits the calendar-ordered surface, samples the
    standardized-moneyness grid, computes pre/post arbitrage metrics on the
    shared fixed-k grid, and assembles a long-format ``SURFACES_DAILY`` frame:
    one ``param`` row per fitted slice plus exactly
    ``len(cfg.tenor_grid_days) * len(cfg.moneyness_grid)`` ``grid`` rows, every
    row carrying the identical snap-level arb metrics (Requirement 7).

    The frame is validated against ``SURFACES_DAILY`` at the stage boundary
    (Requirement 9.1/9.2): a computed column that violates the contract makes
    pandera raise loudly and name the offending column — the exception is *not*
    caught, so the snap fails and no partition is written (design Scenario 6).
    """
    if len(obs) < cfg.min_expiries_per_snap:
        logger.warning(
            "snap %s: coverage gap — %d fittable expiries (< min_expiries_per_snap=%d); "
            "skipping snap, no partition written",
            snap_date.isoformat(),
            len(obs),
            cfg.min_expiries_per_snap,
        )
        return None

    fit = fit_surface(obs, cfg)
    grid = sample_grid(fit, cfg)
    arb_pre, arb_post = surface_arb_metrics(obs, fit, grid, cfg)

    arb = _arb_columns(arb_pre, arb_post)
    records = _param_records(snap_date, fit, arb) + _grid_records(snap_date, grid, arb)
    frame = pl.DataFrame(records, schema=_OUTPUT_SCHEMA).select(_OUTPUT_COLUMNS)

    # Stage-boundary contract (Requirement 9.1/9.2): pandera raises and names the
    # offending column; the exception propagates so the snap writes nothing.
    rows = validate(frame, SURFACES_DAILY)

    qc = _qc_row(snap_date, fit, arb_pre, arb_post, cfg)
    return SurfaceResult(
        snap_date=snap_date,
        fit=fit,
        grid=grid,
        arb_pre=arb_pre,
        arb_post=arb_post,
        rows=rows,
        qc=qc,
    )


# --- Task 7.3: run_build_surfaces driver (loop days -> write partitions) ---


def _iter_days(start: date, end: date) -> list[date]:
    """Every calendar day in ``[start, end]`` inclusive (one snap per day)."""
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _write_partition(df: pl.DataFrame, path: Path) -> None:
    """Write one date partition as zstd Parquet (matching the M2/M3 idiom)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")


def _write_qc(qc_rows: list[dict[str, object]], out_dir: Path, written_parts: list[Path]) -> None:
    """Write the run-level QC dashboard alongside ``surfaces_daily`` (Requirement 8).

    Delegates to :mod:`volguard.surface.qc`: validates + writes the per-snap
    ``SURFACE_QC`` summary (``_qc/summary.parquet``; Requirement 8.1/8.2/8.4/8.5)
    and derives the ``(tenor, moneyness)`` coverage heatmap read-only from the
    partitions written by *this run* (``written_parts``; Requirement 8.3), so the
    coverage summary stays consistent with the run's ``qc_rows`` even when
    ``surfaces_daily`` already holds partitions outside the requested range.
    """
    qc.write_dashboard(qc_rows, out_dir)
    qc.write_coverage(out_dir, written_parts)


def run_build_surfaces(
    cfg: SurfaceConfig,
    data_cfg: DataConfig,
    start: str | None,
    end: str | None,
) -> None:
    """Run Layer-2 surface construction over a date range into ``surfaces_daily``.

    Iterates one snap per day in ``[start, end]`` (defaulting to
    ``data_cfg.history_start`` .. today), reads each day's
    ``curated/quotes_norm/date=YYYY-MM-DD/part.parquet``, and for each productive
    snap writes ``curated/surfaces_daily/date=YYYY-MM-DD/part.parquet`` (zstd,
    date-partitioned; Requirements 7.6/7.7, 11.1).

    Days with no landed ``quotes_norm`` partition are logged and skipped rather
    than failing (Requirement 9.4, matching M3's incomplete-backfill handling);
    snaps that fail coverage (:func:`build_one_surface` returns ``None``) are
    likewise skipped. Each written snap contributes one ``SURFACE_QC`` row; the
    collected rows are handed to :func:`_write_qc` (a task-8 stub) after the loop.
    """
    start_date = date.fromisoformat(start) if start else date.fromisoformat(data_cfg.history_start)
    end_date = date.fromisoformat(end) if end else datetime.now(UTC).date()

    qn_dir = data_cfg.curated_dir / "quotes_norm"
    if not qn_dir.exists():
        logger.warning("build-surfaces: no curated quotes_norm under %s; nothing to build", qn_dir)
        return

    out_dir = data_cfg.curated_dir / "surfaces_daily"
    qc_rows: list[dict[str, object]] = []
    written_parts: list[Path] = []
    for day in _iter_days(start_date, end_date):
        part = qn_dir / f"date={day.isoformat()}" / "part.parquet"
        if not part.exists():
            logger.info("build-surfaces %s: no quotes_norm partition; skipping", day.isoformat())
            continue

        obs = load_snap(pl.scan_parquet(part), day, cfg)
        result = build_one_surface(day, obs, cfg)
        if result is None:
            continue  # coverage skip already logged

        out_part = out_dir / f"date={day.isoformat()}" / "part.parquet"
        _write_partition(result.rows, out_part)
        written_parts.append(out_part)
        qc_rows.append(result.qc)
        logger.info("build-surfaces %s: wrote %d surface rows", day.isoformat(), result.rows.height)

    _write_qc(qc_rows, out_dir, written_parts)
    logger.info("build-surfaces: wrote %d daily partitions to %s", len(written_parts), out_dir)
