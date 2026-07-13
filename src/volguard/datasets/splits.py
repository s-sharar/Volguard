"""Expanding walk-forward folds and compact target-membership manifests."""

from __future__ import annotations

import calendar
import os
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from volguard.config import EvalConfig
from volguard.datasets.schemas import SPLIT_MANIFEST, validate
from volguard.datasets.types import Fold

_MANIFEST_SCHEMA = {
    "fold_id": pl.Int64,
    "target_date": pl.Date,
    "split": pl.String,
    "train_start": pl.Date,
    "train_end": pl.Date,
    "validation_start": pl.Date,
    "validation_end": pl.Date,
    "test_start": pl.Date,
    "test_end": pl.Date,
    "tune_hyperparameters": pl.Boolean,
}
_TUNING_FOLDS = 2


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def generate_walk_forward_folds(dates: Sequence[date], cfg: EvalConfig) -> tuple[Fold, ...]:
    """Generate complete expanding folds over the available target-date span."""
    unique_dates = sorted(set(dates))
    if not unique_dates:
        return ()
    start = unique_dates[0]
    data_end = unique_dates[-1] + timedelta(days=1)
    folds: list[Fold] = []
    fold_id = 0
    while True:
        train_end = _add_months(start, cfg.initial_fit_months + fold_id * cfg.step_months)
        validation_end = _add_months(train_end, cfg.val_months)
        test_end = _add_months(validation_end, cfg.test_months)
        if test_end > data_end:
            break
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=start,
                train_end=train_end,
                validation_start=train_end,
                validation_end=validation_end,
                test_start=validation_end,
                test_end=test_end,
                tune_hyperparameters=fold_id < _TUNING_FOLDS,
            )
        )
        fold_id += 1
    return tuple(folds)


def build_split_manifest(dates: Iterable[date], folds: Sequence[Fold]) -> pl.DataFrame:
    """Materialize target-date membership for each half-open fold range."""
    unique_dates = sorted(set(dates))
    rows: list[dict[str, object]] = []
    for fold in folds:
        for target_date in unique_dates:
            split = fold.split_for(target_date)
            if split is None:
                continue
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "target_date": target_date,
                    "split": split,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "validation_start": fold.validation_start,
                    "validation_end": fold.validation_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "tune_hyperparameters": fold.tune_hyperparameters,
                }
            )
    frame = (
        pl.DataFrame(rows, schema=_MANIFEST_SCHEMA)
        if rows
        else pl.DataFrame(schema=_MANIFEST_SCHEMA)
    )
    validated = validate(frame, SPLIT_MANIFEST)
    if validated.select(["fold_id", "target_date"]).unique().height != validated.height:
        raise ValueError("split manifest contains duplicate fold target membership")
    return validated


def write_split_manifest(frame: pl.DataFrame, path: Path, *, root: Path | None = None) -> None:
    """Validate and atomically replace a compact split manifest."""
    validated = validate(frame, SPLIT_MANIFEST)
    expected_root = path.parent if root is None else root
    root_resolved = expected_root.resolve()
    parent_resolved = path.parent.resolve()
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"refusing symlinked manifest path: {path}")
    if not parent_resolved.is_relative_to(root_resolved):
        raise ValueError(f"manifest path is outside expected root {expected_root}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink():
        raise ValueError(f"refusing symlinked temporary manifest path: {temporary}")
    validated.write_parquet(temporary, compression="zstd")
    os.replace(temporary, path)
