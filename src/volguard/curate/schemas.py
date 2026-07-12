"""Layer-1 pandera contract for ``curated/quotes_norm`` (design Model 2).

This is the frozen contract between Layer 1 (curation) and everything
downstream: the per-snap output frame is validated against ``QUOTES_NORM``
before it is written to Parquet, so Layer 2+ can trust column names, dtypes,
and the core invariants (``tau > 0``, ``F > 0``, finite ``k``, banded
``iv_obs``, enumerated ``cp``/``iv_source``/``fwd_method``).

Style mirrors :mod:`volguard.ingest.schemas` (M2): ``pandera.polars`` schemas,
UTC-millisecond ``Datetime`` timestamps, ``strict=True`` + ``coerce=True``, and
the shared :func:`validate` helper (re-exported here rather than duplicated).

The ``iv_obs`` band ``[iv_min, iv_max]`` is parameterized from
:class:`~volguard.config.CurateConfig` because a pandera schema is a value, not
a class: :func:`quotes_norm_schema` closes over ``cfg.iv_min``/``cfg.iv_max``,
and the module-level :data:`QUOTES_NORM` constant is built from the
``CurateConfig`` defaults for the common case.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from volguard.config import CurateConfig

# Re-export M2's validate helper so callers use one code path (design Model 2).
from volguard.ingest.schemas import validate

__all__ = ["QUOTES_NORM", "quotes_norm_schema", "validate"]

# UTC-millisecond datetimes, identical to the raw-layer schemas (M2).
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")


def _finite(data: pa.PolarsData) -> pl.LazyFrame:
    """Element-wise finiteness check: not null, not inf, not nan.

    ``pl.Expr.is_finite`` returns ``False`` for +/-inf and null for null, so an
    explicit ``is_not_null`` guard is combined in to reject nulls too.
    """
    col = pl.col(data.key)
    return data.lazyframe.select(col.is_finite() & col.is_not_null())


def quotes_norm_schema(cfg: CurateConfig) -> pa.DataFrameSchema:
    """Build the ``QUOTES_NORM`` contract, banding ``iv_obs`` from ``cfg``.

    The exact plan columns (design Model 2) with UTC-ms datetimes, ``tau``,
    ``strike``, and ``F`` strictly positive, finite ``k``, ``iv_obs`` inside
    ``[cfg.iv_min, cfg.iv_max]``, and enumerated ``cp``/``iv_source``/
    ``fwd_method``. Runs in strict, coercing mode.
    """
    return pa.DataFrameSchema(
        {
            "snap_ts": pa.Column(_TS),
            "expiry": pa.Column(_TS),
            "tau": pa.Column(pl.Float64, pa.Check.gt(0.0)),
            "strike": pa.Column(pl.Float64, pa.Check.gt(0.0)),
            "cp": pa.Column(pl.String, pa.Check.isin(["C", "P"])),
            "F": pa.Column(pl.Float64, pa.Check.gt(0.0)),
            "k": pa.Column(pl.Float64, pa.Check(_finite, error="k must be finite")),
            "iv_obs": pa.Column(pl.Float64, pa.Check.in_range(cfg.iv_min, cfg.iv_max)),
            "iv_source": pa.Column(pl.String, pa.Check.isin(["trade", "mark", "mid"])),
            "usd_premium": pa.Column(pl.Float64, pa.Check.ge(0.0)),
            "size": pa.Column(pl.Float64, pa.Check.ge(0.0)),
            "staleness_s": pa.Column(pl.Float64, pa.Check.ge(0.0)),
            "quality_flags": pa.Column(pl.Int64, pa.Check.ge(0)),
            "fwd_method": pa.Column(
                pl.String, pa.Check.isin(["pcp", "future", "index_carry"])
            ),
        },
        strict=True,
        coerce=True,
    )


# Default contract using the CurateConfig defaults (iv_min=0.01, iv_max=5.0).
QUOTES_NORM = quotes_norm_schema(CurateConfig())
