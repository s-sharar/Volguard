"""Layer-2 QC dashboard aggregation (design **Component 7**; Requirement 8).

The :func:`~volguard.surface.pipeline.run_build_surfaces` driver collects one
``SURFACE_QC`` row per productive snap (assembled by ``build_one_surface``) and
hands the list to this module after the day loop. Here we:

- assemble those rows into a single ``SURFACE_QC``-valid frame
  (:func:`build_qc_frame`), validating at the contract boundary exactly like the
  surfaces_daily write path (Requirement 8.1/8.2/8.4);
- write that frame as a Parquet summary alongside ``curated/surfaces_daily`` —
  under ``surfaces_daily/_qc/summary.parquet`` (Requirement 8.5) — and log a
  short run-level summary table (mean fit RMSE, ``svi``/``ssvi`` fallback split,
  mean market arb-rate) so method-fallback frequency is visible without a
  plotting dependency (:func:`write_dashboard`);
- derive the ``(tenor, moneyness)`` coverage summary (Requirement 8.3) by
  scanning the just-written ``surfaces_daily`` grid rows and aggregating mean
  ``cell_n_obs`` per ``(tau, moneyness)`` (:func:`coverage_summary` /
  :func:`write_coverage`).

**Coverage-summary design note (Requirement 8.3):** the per-snap ``SURFACE_QC``
row deliberately does *not* carry per-cell observation counts — those live on the
long-format ``surfaces_daily`` ``grid`` rows as ``cell_n_obs``. Rather than widen
the frozen ``SURFACE_QC`` contract, the coverage heatmap is computed read-only
from the written grid partitions: :func:`coverage_summary` ``rglob``-scans
``surfaces_daily/date=*/part.parquet``, keeps ``record_kind == "grid"`` rows, and
returns mean/min/max/total ``cell_n_obs`` grouped by ``(tau, moneyness)`` across
the whole run. This keeps the contract stable and the summary a pure derived view
of what was actually written.

Conventions mirror the M2/M3/M4 code paths: ``pandera.polars`` validation via the
shared :func:`~volguard.surface.schemas.validate` helper, explicit Polars dtypes
taken from the frozen schema, zstd Parquet writes with ``mkdir(parents=True)``,
and ``rglob`` partition scans.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from volguard.surface.schemas import SURFACE_QC, validate

__all__ = ["build_qc_frame", "coverage_summary", "write_coverage", "write_dashboard"]

logger = logging.getLogger(__name__)

# Explicit Polars dtypes for the per-snap QC frame, taken column-for-column from
# the frozen ``SURFACE_QC`` contract so this module and the schema never drift
# (the schema is strict + coerce, so an empty/typed frame validates cleanly).
_QC_SCHEMA: dict[str, pl.DataType] = {
    "snap_date": pl.Date(),
    "n_expiries": pl.Int64(),
    "mean_rmse": pl.Float64(),
    "max_rmse": pl.Float64(),
    "n_svi": pl.Int64(),
    "n_ssvi": pl.Int64(),
    "market_arb_rate": pl.Float64(),
    "arb_butterfly_pre": pl.Int64(),
    "arb_calendar_pre": pl.Int64(),
    "arb_butterfly_post": pl.Int64(),
    "arb_calendar_post": pl.Int64(),
    "total_n_obs": pl.Int64(),
}

# The exact, ordered ``SURFACE_QC`` columns, taken from the frozen schema.
_QC_COLUMNS: tuple[str, ...] = tuple(SURFACE_QC.columns.keys())

# Guard: the explicit QC schema must cover exactly the frozen contract's columns
# (a schema change in surface/schemas.py fails loudly here, not downstream).
assert set(_QC_SCHEMA) == set(_QC_COLUMNS), "surface/qc._QC_SCHEMA drifted from SURFACE_QC.columns"


def build_qc_frame(qc_rows: list[dict[str, object]]) -> pl.DataFrame:
    """Assemble + validate the per-snap ``SURFACE_QC`` frame (one row per snap).

    Builds a Polars frame from ``qc_rows`` with the frozen contract's dtypes and
    column order and validates it against ``SURFACE_QC`` (Requirement 8.1/8.2/8.4).
    An empty ``qc_rows`` yields an empty but correctly-typed, validated frame so
    callers get a uniform return type regardless of how many snaps were built.
    """
    frame = pl.DataFrame(qc_rows, schema=_QC_SCHEMA).select(_QC_COLUMNS)
    return validate(frame, SURFACE_QC)


def coverage_summary(parts: list[Path]) -> pl.DataFrame | None:
    """Aggregate mean ``cell_n_obs`` per ``(tau, moneyness)`` grid cell (R8.3).

    Read-only scan of the ``surfaces_daily`` partitions written **by this run**
    (``parts``, the exact part paths the driver just produced): keeps
    ``record_kind == "grid"`` rows and returns the per-cell coverage heatmap —
    ``mean_n_obs``, ``min_n_obs``, ``max_n_obs``, ``total_n_obs``, and the
    ``n_snaps`` contributing — grouped by ``(tau, moneyness)`` and ordered for
    stable output. Taking explicit paths (rather than globbing the whole
    directory) keeps the coverage summary consistent with the per-run
    ``_qc/summary.parquet``: a rerun over a sub-range must not fold in stale
    historical partitions outside the requested window. Returns ``None`` when
    ``parts`` is empty (nothing written), so callers can skip the write.
    """
    if not parts:
        return None

    grid = (
        pl.scan_parquet(parts)
        .filter(pl.col("record_kind") == "grid")
        .group_by("tau", "moneyness")
        .agg(
            pl.col("cell_n_obs").mean().alias("mean_n_obs"),
            pl.col("cell_n_obs").min().alias("min_n_obs"),
            pl.col("cell_n_obs").max().alias("max_n_obs"),
            pl.col("cell_n_obs").sum().alias("total_n_obs"),
            pl.len().alias("n_snaps"),
        )
        .sort("tau", "moneyness")
        .collect()
    )
    if grid.height == 0:
        return None
    return grid


def write_coverage(surfaces_dir: Path, parts: list[Path]) -> Path | None:
    """Write the ``(tenor, moneyness)`` coverage heatmap alongside surfaces (R8.3/8.5).

    Computes :func:`coverage_summary` over the partitions ``parts`` written by
    this run and, when non-empty, writes it to ``surfaces_dir/_qc/coverage.parquet``
    (zstd). Scoping to this run's partitions keeps the heatmap consistent with the
    run's ``_qc/summary.parquet`` (built only from the run's ``qc_rows``). Returns
    the written path, or ``None`` when there is nothing to summarize.
    """
    coverage = coverage_summary(parts)
    if coverage is None:
        logger.info("qc: no surfaces_daily grid rows written this run; skipping coverage summary")
        return None

    path = surfaces_dir / "_qc" / "coverage.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_parquet(path, compression="zstd")
    logger.info(
        "qc: wrote %d-cell (tenor, moneyness) coverage summary to %s", coverage.height, path
    )
    return path


def write_dashboard(qc_rows: list[dict[str, object]], out_dir: Path) -> Path | None:
    """Validate + write the QC summary Parquet alongside ``surfaces_daily`` (R8.5).

    Assembles the per-snap ``SURFACE_QC`` frame via :func:`build_qc_frame`, writes
    it to ``out_dir/_qc/summary.parquet`` (zstd), and logs a short run-level
    summary — total snaps, mean of ``mean_rmse``, the ``svi`` vs ``ssvi`` fallback
    split (Requirement 8.4), and the mean ``market_arb_rate`` (Requirement
    8.1/8.2) — as the rendered summary (a logged table suffices; no plotting
    dependency). Returns the written path, or ``None`` when ``qc_rows`` is empty
    (no productive snaps, nothing to write).
    """
    if not qc_rows:
        logger.info("qc: no SURFACE_QC rows collected; skipping QC dashboard write")
        return None

    frame = build_qc_frame(qc_rows)

    path = out_dir / "_qc" / "summary.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")

    # Rendered run-level summary (Requirement 8.1-8.4): a logged digest suffices.
    # ``.item()`` extracts a scalar (typed ``Any``) so the float/int coercions are
    # unambiguous for the type checker.
    n_snaps = frame.height
    mean_rmse = float(frame.select(pl.col("mean_rmse").mean()).item() or 0.0)
    total_svi = int(frame.select(pl.col("n_svi").sum()).item() or 0)
    total_ssvi = int(frame.select(pl.col("n_ssvi").sum()).item() or 0)
    mean_arb_rate = float(frame.select(pl.col("market_arb_rate").mean()).item() or 0.0)
    logger.info(
        "qc: %d snaps | mean fit RMSE %.6f | fits svi=%d ssvi=%d (%.1f%% SSVI fallback) | "
        "mean market_arb_rate %.4f | wrote %s",
        n_snaps,
        mean_rmse,
        total_svi,
        total_ssvi,
        100.0 * total_ssvi / (total_svi + total_ssvi) if (total_svi + total_ssvi) else 0.0,
        mean_arb_rate,
        path,
    )
    return path
