"""Layer-2 pandera contracts for ``curated/surfaces_daily`` (design Model 2/3).

These are the frozen contract between Layer 2 (surface fitting) and everything
downstream: the driver validates each snap's long-format output against
:data:`SURFACES_DAILY` before writing a date partition, and emits one
:data:`SURFACE_QC` row per snap, so consumers can trust column names, dtypes,
and the core invariants (``tau > 0``, valid SVI params where present,
``grid_w >= 0``, enumerated ``record_kind``/``fit_method``, bounded
``market_arb_rate``).

``SURFACES_DAILY`` is *long format*: per-tenor SVI params and per-cell grid
values coexist in one date-partitioned table, discriminated by
``record_kind in {param, grid}``. Param columns are populated for ``param``
rows and null for ``grid`` rows (and vice versa), so most value columns are
``nullable=True``.

Style mirrors :mod:`volguard.curate.schemas` (M3) and
:mod:`volguard.ingest.schemas` (M2): ``pandera.polars`` schemas, UTC-millisecond
``Datetime`` timestamps, ``strict=True`` + ``coerce=True``, and the shared
:func:`validate` helper (re-exported here rather than duplicated).
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

# Re-export M2's validate helper so callers use one code path (design Model 2).
from volguard.ingest.schemas import validate

__all__ = ["SURFACES_DAILY", "SURFACE_QC", "validate"]

# UTC-millisecond datetimes, identical to the raw/curated-layer schemas (M2/M3).
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")

# Long-format per-tenor SVI params + per-cell grid values + arb metrics, one
# date-partitioned table discriminated by ``record_kind`` (design Model 2).
SURFACES_DAILY = pa.DataFrameSchema(
    {
        "snap_date": pa.Column(pl.Date),
        "expiry": pa.Column(_TS, nullable=True),  # null for interpolated grid tenors
        "tau": pa.Column(pl.Float64, pa.Check.gt(0.0)),
        "record_kind": pa.Column(pl.String, pa.Check.isin(["param", "grid"])),
        # SVI params (populated when record_kind == "param")
        "svi_a": pa.Column(pl.Float64, nullable=True),
        "svi_b": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "svi_rho": pa.Column(pl.Float64, pa.Check.in_range(-1.0, 1.0), nullable=True),
        "svi_m": pa.Column(pl.Float64, nullable=True),
        "svi_sigma": pa.Column(pl.Float64, pa.Check.gt(0.0), nullable=True),
        "fit_method": pa.Column(pl.String, pa.Check.isin(["svi", "ssvi"]), nullable=True),
        # Grid cells (populated when record_kind == "grid")
        "moneyness": pa.Column(pl.Float64, nullable=True),  # standardized d
        "grid_k": pa.Column(pl.Float64, nullable=True),  # fixed-k for the cell
        "grid_w": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "cell_n_obs": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "interp_flag": pa.Column(pl.Boolean, nullable=True),
        # Per-tenor fit diagnostics
        "rmse": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "n_obs": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "vega_sum": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "butterfly_ok": pa.Column(pl.Boolean, nullable=True),
        "calendar_ok": pa.Column(pl.Boolean, nullable=True),
        # Arb violation counts (pre = market obs grid, post = fitted surface)
        "arb_butterfly_pre": pa.Column(pl.Int64, pa.Check.ge(0)),
        "arb_calendar_pre": pa.Column(pl.Int64, pa.Check.ge(0)),
        "arb_butterfly_post": pa.Column(pl.Int64, pa.Check.ge(0)),
        "arb_calendar_post": pa.Column(pl.Int64, pa.Check.ge(0)),
    },
    strict=True,
    coerce=True,
)

# One row per snap summarizing fit quality + arbitrage base rates (design Model 3).
SURFACE_QC = pa.DataFrameSchema(
    {
        "snap_date": pa.Column(pl.Date),
        "n_expiries": pa.Column(pl.Int64, pa.Check.ge(0)),
        "mean_rmse": pa.Column(pl.Float64, pa.Check.ge(0.0)),
        "max_rmse": pa.Column(pl.Float64, pa.Check.ge(0.0)),
        "n_svi": pa.Column(pl.Int64, pa.Check.ge(0)),
        "n_ssvi": pa.Column(pl.Int64, pa.Check.ge(0)),
        "market_arb_rate": pa.Column(pl.Float64, pa.Check.in_range(0.0, 1.0)),
        "arb_butterfly_pre": pa.Column(pl.Int64, pa.Check.ge(0)),
        "arb_calendar_pre": pa.Column(pl.Int64, pa.Check.ge(0)),
        "arb_butterfly_post": pa.Column(pl.Int64, pa.Check.ge(0)),
        "arb_calendar_post": pa.Column(pl.Int64, pa.Check.ge(0)),
        "total_n_obs": pa.Column(pl.Int64, pa.Check.ge(0)),
    },
    strict=True,
    coerce=True,
)
