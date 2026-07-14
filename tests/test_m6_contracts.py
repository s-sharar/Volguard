"""M6 immutable contracts, EvalConfig extensions, and input fingerprinting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from volguard.config import (
    ArtifactPathsConfig,
    BaselineConfig,
    EvalConfig,
    MetricsConfig,
    RegimeConfig,
    RepairEvalConfig,
    SignificanceConfig,
    load_config,
)
from volguard.datasets.splits import build_split_manifest
from volguard.datasets.types import Fold
from volguard.models.inputs import (
    fingerprint_eval_inputs,
    load_eval_panel,
    require_single_grid_spec,
)
from volguard.models.types import (
    FittedBaseline,
    ForecastBatch,
    ForecastRecord,
    GridSpec,
    MetricRecord,
    RunManifest,
)

_GRID = (6, 9)
_SIGNATURE = "tenors=7,14,30,60,90,180|moneyness=-2,-1.5,-1,-0.5,0,0.5,1,1.5,2"


def _daily_rows(n: int, *, start: date = date(2024, 1, 1)) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        snap = start + timedelta(days=i)
        ts = datetime(snap.year, snap.month, snap.day, 8, 5, tzinfo=UTC)
        w = [0.04 + 0.001 * i] * 54
        rows.append(
            {
                "snap_date": snap,
                "snap_ts": ts,
                "grid_signature": _SIGNATURE,
                "grid_w": w,
                "grid_k": [0.0] * 54,
                "grid_quality_weight": [0.8] * 54,
                "dvol": 0.5 + 0.01 * i,
                "dvol_change_5d": 0.01,
                "rv_parkinson_1d": 0.2,
                "rv_parkinson_5d": 0.21,
                "rv_parkinson_22d": 0.22,
                "max_source_ts": ts,
            }
        )
    return pl.DataFrame(rows)


def test_eval_config_loads_extended_defaults() -> None:
    cfg = load_config("eval", EvalConfig)
    assert cfg.initial_train_months == 18
    assert cfg.repair.max_iter == 10
    assert cfg.repair.move_tol == 1e-10
    assert cfg.baselines.ewma_lambdas[0] == 0.05
    assert cfg.baselines.pca_components == [3, 4, 5]
    assert cfg.baselines.ridge_alphas[-1] == 100.0
    assert cfg.metrics.primary_weight == "vega"
    assert cfg.significance.dm_lag == 0
    assert cfg.significance.hln_correction is True
    assert cfg.regimes.stress_dvol_quantile == 0.8
    assert cfg.w_floor == 0.0
    assert cfg.artifacts.runs_dir.name == "runs"


def test_eval_config_rejects_invalid_nested_settings() -> None:
    with pytest.raises(ValidationError):
        EvalConfig(repair=RepairEvalConfig(max_iter=0))
    with pytest.raises(ValidationError):
        EvalConfig(baselines=BaselineConfig(ewma_lambdas=[]))
    with pytest.raises(ValidationError):
        EvalConfig(regimes=RegimeConfig(stress_dvol_quantile=1.5))
    with pytest.raises(ValidationError):
        EvalConfig(significance=SignificanceConfig(dm_lag=-1))
    with pytest.raises(ValidationError):
        MetricsConfig(primary_weight="not-a-weight")


def test_grid_spec_and_forecast_types_are_immutable() -> None:
    spec = GridSpec.from_axes(
        tenors_days=(7, 14, 30, 60, 90, 180),
        moneyness=(-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2),
        signature=_SIGNATURE,
    )
    assert spec.shape == _GRID
    raw = np.ones(_GRID)
    record = ForecastRecord(
        model_id="b0",
        fold_id=0,
        split="test",
        issue_date=date(2024, 1, 1),
        target_date=date(2024, 1, 2),
        raw_w=raw,
        pre_floor_negative_count=0,
        pre_floor_min=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        record.model_id = "b1"  # type: ignore[misc]
    with pytest.raises(ValueError):
        record.raw_w[0, 0] = 99.0

    fitted = FittedBaseline(
        model_id="b0",
        fold_id=0,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 2, 1),
        hyperparameters={},
        state={},
    )
    batch = ForecastBatch(
        model_id="b0",
        fold_id=0,
        records=(record,),
        fitted=fitted,
    )
    assert len(batch.records) == 1

    metric = MetricRecord(
        model_id="b0",
        fold_id=0,
        split="test",
        variant="raw",
        scope="overall",
        metric="vw_mse_w",
        value=0.1,
        n=10,
        weight_scheme="vega",
    )
    manifest = RunManifest(
        run_id="run-test",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        model_ids=("b0",),
        config_hash="abc",
        data_fingerprint="def",
        git_commit=None,
        seed=0,
    )
    assert metric.n == 10
    assert manifest.run_id == "run-test"
    assert ArtifactPathsConfig().runs_dir.name == "runs"


def test_require_single_grid_spec_rejects_mixed_signatures() -> None:
    frame = _daily_rows(3)
    altered = frame.with_columns(
        pl.when(pl.col("snap_date") == date(2024, 1, 2))
        .then(pl.lit("other-signature"))
        .otherwise(pl.col("grid_signature"))
        .alias("grid_signature")
    )
    with pytest.raises(ValueError, match="single grid_signature"):
        require_single_grid_spec(altered)


def test_load_eval_panel_fingerprints_and_keeps_target_k(tmp_path: Path) -> None:
    daily = _daily_rows(5)
    daily_dir = tmp_path / "features" / "daily"
    for row in daily.iter_rows(named=True):
        day = row["snap_date"]
        part = daily_dir / f"date={day.isoformat()}" / "part.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([row]).write_parquet(part)

    fold = Fold(
        fold_id=0,
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 3),
        validation_start=date(2024, 1, 3),
        validation_end=date(2024, 1, 4),
        test_start=date(2024, 1, 4),
        test_end=date(2024, 1, 6),
        tune_hyperparameters=True,
    )
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    manifest = build_split_manifest(dates, (fold,))
    splits_path = tmp_path / "features" / "splits" / "part.parquet"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(splits_path)

    panel = load_eval_panel(tmp_path / "features")
    assert panel.grid_spec.signature == _SIGNATURE
    assert panel.grid_w.shape == (5, 6, 9)
    assert panel.grid_k.shape == (5, 6, 9)
    assert panel.reliability.shape == (5, 6, 9)
    fp1 = fingerprint_eval_inputs(panel)
    assert fingerprint_eval_inputs(panel) == fp1
    assert len(fp1) == 64

    last_part = daily_dir / "date=2024-01-05" / "part.parquet"
    last = pl.read_parquet(last_part)
    w = list(last["grid_w"][0])
    w[0] = float(w[0]) + 1.0
    last.with_columns(pl.Series("grid_w", [w])).write_parquet(last_part)
    panel2 = load_eval_panel(tmp_path / "features")
    assert fingerprint_eval_inputs(panel2) != fp1
