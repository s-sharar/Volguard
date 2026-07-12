"""Tests for the frozen ``QUOTES_NORM`` Layer-1 contract (design Model 2).

A conforming frame validates and coerces; frames violating each invariant
(tau<=0, strike<=0, F<=0, non-finite k, iv_obs out of band, bad cp,
bad iv_source, bad fwd_method, missing/extra column) raise ``SchemaError``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pandera.errors as pa_errors
import polars as pl
import polars.exceptions as pl_errors
import pytest

from volguard.config import CurateConfig
from volguard.curate.schemas import QUOTES_NORM, quotes_norm_schema, validate

# Pandera coerces declared dtypes before the presence check, so a missing
# column surfaces as polars' ColumnNotFoundError rather than SchemaError; both
# are loud, fail-the-snap errors for the driver.
_SCHEMA_FAILURES = (pa_errors.SchemaError, pl_errors.ColumnNotFoundError)

_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_EXPIRY = datetime(2022, 4, 8, 8, 0, tzinfo=UTC)


def _good_frame() -> pl.DataFrame:
    """A minimal two-row frame that conforms to ``QUOTES_NORM``."""
    return pl.DataFrame(
        {
            "snap_ts": [_SNAP, _SNAP],
            "expiry": [_EXPIRY, _EXPIRY],
            "tau": [7.0 / 365.0, 7.0 / 365.0],
            "strike": [40000.0, 45000.0],
            "cp": ["C", "P"],
            "F": [42000.0, 42000.0],
            "k": [math.log(40000.0 / 42000.0), math.log(45000.0 / 42000.0)],
            "iv_obs": [0.65, 0.72],
            "iv_source": ["trade", "mark"],
            "usd_premium": [1234.5, 2345.6],
            "size": [1.5, 0.5],
            "staleness_s": [0.0, 120.0],
            "quality_flags": [0, 64],
            "fwd_method": ["pcp", "pcp"],
        }
    )


def test_conforming_frame_validates() -> None:
    out = validate(_good_frame(), QUOTES_NORM)
    assert out.height == 2
    assert set(out.columns) == set(QUOTES_NORM.columns.keys())


def test_conforming_frame_coerces_int_iv_source_dtypes() -> None:
    """coerce=True should cast compatible dtypes (e.g. int tau/size to float)."""
    df = _good_frame().with_columns(
        pl.col("quality_flags").cast(pl.Int32),  # coercible to Int64
        pl.col("size").cast(pl.Int64),  # coercible to Float64
    )
    out = validate(df, QUOTES_NORM)
    assert out.schema["quality_flags"] == pl.Int64
    assert out.schema["size"] == pl.Float64


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("tau", [0.0, 7.0 / 365.0]),  # tau <= 0
        ("strike", [-1.0, 45000.0]),  # strike <= 0
        ("F", [0.0, 42000.0]),  # F <= 0
        ("k", [float("inf"), 0.1]),  # non-finite k
        ("k", [float("nan"), 0.1]),  # non-finite k (nan)
        ("iv_obs", [10.0, 0.72]),  # above iv_max
        ("iv_obs", [0.0001, 0.72]),  # below iv_min
        ("cp", ["X", "P"]),  # bad cp
        ("iv_source", ["bogus", "mark"]),  # bad iv_source
        ("fwd_method", ["spot", "pcp"]),  # bad fwd_method
    ],
)
def test_invariant_violations_raise(column: str, bad_value: list) -> None:
    df = _good_frame().with_columns(pl.Series(column, bad_value))
    with pytest.raises(pa_errors.SchemaError):
        validate(df, QUOTES_NORM)


def test_missing_column_raises() -> None:
    df = _good_frame().drop("k")
    with pytest.raises(_SCHEMA_FAILURES):
        validate(df, QUOTES_NORM)


def test_extra_column_raises() -> None:
    """strict=True rejects unexpected columns."""
    df = _good_frame().with_columns(pl.lit(1.0).alias("unexpected"))
    with pytest.raises(pa_errors.SchemaError):
        validate(df, QUOTES_NORM)


def test_iv_band_parameterized_from_config() -> None:
    """A tighter cfg band rejects an iv_obs the default band would accept."""
    tight = quotes_norm_schema(CurateConfig(iv_min=0.5, iv_max=0.7))
    df = _good_frame()  # second row iv_obs=0.72 is outside [0.5, 0.7]
    with pytest.raises(pa_errors.SchemaError):
        validate(df, tight)
    # ...but the default band accepts it.
    assert validate(df, QUOTES_NORM).height == 2
