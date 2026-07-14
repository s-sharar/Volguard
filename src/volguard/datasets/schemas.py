"""Strict contract for persisted M5 walk-forward membership."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from volguard.ingest.schemas import validate

__all__ = ["SPLIT_MANIFEST", "validate"]

SPLIT_MANIFEST = pa.DataFrameSchema(
    {
        "fold_id": pa.Column(pl.Int64, pa.Check.ge(0)),
        "target_date": pa.Column(pl.Date),
        "split": pa.Column(pl.String, pa.Check.isin(["train", "validation", "test"])),
        "train_start": pa.Column(pl.Date),
        "train_end": pa.Column(pl.Date),
        "validation_start": pa.Column(pl.Date),
        "validation_end": pa.Column(pl.Date),
        "test_start": pa.Column(pl.Date),
        "test_end": pa.Column(pl.Date),
        "tune_hyperparameters": pa.Column(pl.Boolean),
    },
    strict=True,
    coerce=True,
)
