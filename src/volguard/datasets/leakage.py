"""Hard leakage assertions shared by feature and dataset builders."""

from __future__ import annotations

import polars as pl


def assert_no_feature_leakage(daily: pl.DataFrame) -> None:
    """Reject any row whose known source timestamp exceeds its snap timestamp."""
    required = {"snap_ts", "max_source_ts"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily features missing leakage columns: {sorted(missing)}")
    source_columns = [name for name in daily.columns if name.endswith("_source_ts")] + [
        "max_source_ts"
    ]
    for column in dict.fromkeys(source_columns):
        violations = daily.filter(
            pl.col(column).is_not_null() & (pl.col(column) > pl.col("snap_ts"))
        )
        if not violations.is_empty():
            raise ValueError(f"{column} must not exceed snap_ts")
