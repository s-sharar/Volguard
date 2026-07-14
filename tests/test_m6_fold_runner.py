"""Fold runner: train-only thresholds, forecast-before-target, leakage guards."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from volguard.config import EvalConfig
from volguard.datasets.splits import build_split_manifest
from volguard.datasets.types import Fold
from volguard.models.baselines import PersistenceBaseline
from volguard.models.fold_runner import apply_w_floor, build_fold_context, run_fold_forecasts
from volguard.models.inputs import load_eval_panel

_SIGNATURE = "tenors=7,14,30,60,90,180|moneyness=-2,-1.5,-1,-0.5,0,0.5,1,1.5,2"


def _write_panel(tmp_path: Path, n: int = 10, *, poison_last: bool = False) -> Path:
    features = tmp_path / "features"
    daily_dir = features / "daily"
    rows: list[dict[str, object]] = []
    start = date(2024, 1, 1)
    for i in range(n):
        snap = start + timedelta(days=i)
        ts = datetime(snap.year, snap.month, snap.day, 8, 5, tzinfo=UTC)
        level = 0.04 + 0.001 * i
        if poison_last and i == n - 1:
            level = 9.0
        w = [level] * 54
        k = [float(j) * 0.1 for j in range(54)]
        rows.append(
            {
                "snap_date": snap,
                "snap_ts": ts,
                "grid_signature": _SIGNATURE,
                "grid_w": w,
                "grid_k": k,
                "grid_quality_weight": [0.8] * 54,
                "dvol": 0.4 + 0.05 * i,
                "dvol_change_5d": 0.01 * i,
                "rv_parkinson_1d": 0.2,
                "rv_parkinson_5d": 0.21,
                "rv_parkinson_22d": 0.22,
                "max_source_ts": ts,
            }
        )
        part = daily_dir / f"date={snap.isoformat()}" / "part.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([rows[-1]]).write_parquet(part)

    fold = Fold(
        fold_id=0,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 5),
        validation_start=date(2024, 1, 5),
        validation_end=date(2024, 1, 8),
        test_start=date(2024, 1, 8),
        test_end=date(2024, 1, 11),
        tune_hyperparameters=True,
    )
    dates = [start + timedelta(days=i) for i in range(n)]
    manifest = build_split_manifest(dates, (fold,))
    splits = features / "splits" / "part.parquet"
    splits.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(splits)
    return features


def test_fold_context_dvol_threshold_uses_train_only(tmp_path: Path) -> None:
    panel = load_eval_panel(_write_panel(tmp_path))
    fold = Fold(
        fold_id=0,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 5),
        validation_start=date(2024, 1, 5),
        validation_end=date(2024, 1, 8),
        test_start=date(2024, 1, 8),
        test_end=date(2024, 1, 11),
        tune_hyperparameters=True,
    )
    cfg = EvalConfig()
    ctx = build_fold_context(panel, fold, cfg)

    train_dvol = panel.dvol[list(ctx.train_indices)]
    expected = float(np.quantile(train_dvol, cfg.regimes.stress_dvol_quantile))
    assert ctx.dvol_stress_threshold == pytest.approx(expected)
    # Validation/test DVOL levels are higher; they must not inflate the threshold.
    assert ctx.dvol_stress_threshold is not None
    assert ctx.dvol_stress_threshold < float(np.nanmax(panel.dvol[list(ctx.test_indices)]))


def test_b0_forecasts_equal_previous_day_and_ignore_future_poison(tmp_path: Path) -> None:
    clean = load_eval_panel(_write_panel(tmp_path / "clean"))
    poisoned = load_eval_panel(_write_panel(tmp_path / "poison", poison_last=True))
    fold = Fold(
        fold_id=0,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 5),
        validation_start=date(2024, 1, 5),
        validation_end=date(2024, 1, 8),
        test_start=date(2024, 1, 8),
        test_end=date(2024, 1, 11),
        tune_hyperparameters=True,
    )
    cfg = EvalConfig()
    model = PersistenceBaseline()

    clean_batch = run_fold_forecasts(model, build_fold_context(clean, fold, cfg), cfg=cfg)
    poison_batch = run_fold_forecasts(model, build_fold_context(poisoned, fold, cfg), cfg=cfg)

    assert len(clean_batch.records) == len(poison_batch.records) > 0
    assert {rec.split for rec in clean_batch.records} <= {"validation", "test"}
    assert "train" not in {rec.split for rec in clean_batch.records}
    for clean_rec, poison_rec in zip(clean_batch.records, poison_batch.records, strict=True):
        assert clean_rec.target_date == poison_rec.target_date
        assert clean_rec.issue_date < clean_rec.target_date
        # Persistence: forecast for day t+1 equals realized w at t.
        issue_idx = clean.dates.index(clean_rec.issue_date)
        np.testing.assert_allclose(clean_rec.raw_w, clean.grid_w[issue_idx])
        # Future-row poison on the last day cannot alter earlier forecasts.
        if clean_rec.target_date < clean.dates[-1]:
            np.testing.assert_allclose(clean_rec.raw_w, poison_rec.raw_w)


def test_w_floor_records_pre_floor_negatives() -> None:
    raw = -0.01 * np.ones((6, 9))
    floored, count, pre_min = apply_w_floor(raw, floor=0.0)
    assert count == 54
    assert pre_min == pytest.approx(-0.01)
    assert np.all(floored >= 0.0)
