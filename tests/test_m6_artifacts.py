"""M6 artifacts, registry, and train/evaluate pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from typer.testing import CliRunner

from volguard.cli import app
from volguard.config import ArtifactPathsConfig, DataConfig, EvalConfig
from volguard.datasets.splits import build_split_manifest
from volguard.datasets.types import Fold
from volguard.experiments.pipeline import evaluate_run, train_and_evaluate, train_baselines
from volguard.experiments.registry import ExperimentRegistry
from volguard.experiments.store import RunStore

_SIGNATURE = "tenors=7,14,30,60,90,180|moneyness=-2,-1.5,-1,-0.5,0,0.5,1,1.5,2"
runner = CliRunner()


def _write_panel(tmp_path: Path, n: int = 24) -> Path:
    features = tmp_path / "features"
    daily_dir = features / "daily"
    start = date(2024, 1, 1)
    for i in range(n):
        snap = start + timedelta(days=i)
        ts = datetime(snap.year, snap.month, snap.day, 8, 5, tzinfo=UTC)
        level = 0.04 + 0.0005 * i
        row = {
            "snap_date": snap,
            "snap_ts": ts,
            "grid_signature": _SIGNATURE,
            "grid_w": [level + 0.001 * ((j % 9) - 4) * 0.01 for j in range(54)],
            "grid_k": [float((j % 9) - 4) * 0.1 for j in range(54)],
            "grid_quality_weight": [0.85] * 54,
            "dvol": 0.45 + 0.01 * (i % 5),
            "dvol_change_5d": 0.0 if i < 5 else 0.01,
            "rv_parkinson_1d": 0.2,
            "rv_parkinson_5d": 0.21,
            "rv_parkinson_22d": 0.22,
            "max_source_ts": ts,
        }
        part = daily_dir / f"date={snap.isoformat()}" / "part.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([row]).write_parquet(part)

    # Two folds so freeze + tune paths both exercise.
    folds = (
        Fold(
            fold_id=0,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 10),
            validation_start=date(2024, 1, 10),
            validation_end=date(2024, 1, 14),
            test_start=date(2024, 1, 14),
            test_end=date(2024, 1, 18),
            tune_hyperparameters=True,
        ),
        Fold(
            fold_id=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 14),
            validation_start=date(2024, 1, 14),
            validation_end=date(2024, 1, 18),
            test_start=date(2024, 1, 18),
            test_end=date(2024, 1, 22),
            tune_hyperparameters=True,
        ),
        Fold(
            fold_id=2,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 18),
            validation_start=date(2024, 1, 18),
            validation_end=date(2024, 1, 21),
            test_start=date(2024, 1, 21),
            test_end=date(2024, 1, 25),
            tune_hyperparameters=False,
        ),
    )
    dates = [start + timedelta(days=i) for i in range(n)]
    manifest = build_split_manifest(dates, folds)
    splits = features / "splits" / "part.parquet"
    splits.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(splits)
    return features


def _cfg(tmp_path: Path) -> tuple[EvalConfig, DataConfig]:
    experiments = tmp_path / "experiments"
    log_path = tmp_path / "experiment-log.md"
    log_path.write_text(
        "# Experiment Log\n\n_(empty - first entries land with M6 baselines)_\n",
        encoding="utf-8",
    )
    eval_cfg = EvalConfig(
        artifacts=ArtifactPathsConfig(
            experiments_dir=experiments,
            experiment_log_path=log_path,
        )
    )
    data_cfg = DataConfig(features_dir=tmp_path / "features")
    return eval_cfg, data_cfg


def test_registry_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "registry.duckdb"
    with ExperimentRegistry(path) as registry:
        registry.register_run(
            run_id="r1",
            model_ids=("b0",),
            config_hash="abc",
            data_fingerprint="def",
            git_commit=None,
            lockfile_hash=None,
            seed=0,
            platform={"python": "3.12"},
            dependencies={"numpy": "2"},
        )
        registry.set_status("r1", "trained")
        assert registry.latest_successful_run_id() == "r1"
        assert registry.get_run("r1")["status"] == "trained"


def test_latest_run_includes_repair_failure_status(tmp_path: Path) -> None:
    path = tmp_path / "registry.duckdb"
    with ExperimentRegistry(path) as registry:
        registry.register_run(
            run_id="r-fail",
            model_ids=("b0",),
            config_hash="abc",
            data_fingerprint="def",
            git_commit=None,
            lockfile_hash=None,
            seed=0,
            platform={"python": "3.12"},
            dependencies={"numpy": "2"},
        )
        registry.set_status("r-fail", "evaluated_with_repair_failures")
        assert registry.latest_successful_run_id() == "r-fail"


def test_new_run_id_unique_within_same_second() -> None:
    from datetime import UTC, datetime

    from volguard.experiments.provenance import new_run_id

    when = datetime(2026, 7, 13, 21, 0, 0, tzinfo=UTC)
    ids = {new_run_id(when=when) for _ in range(20)}
    assert len(ids) == 20
    assert all(rid.startswith("20260713T210000Z-") for rid in ids)


def test_train_b0_persist_and_evaluate_reload(tmp_path: Path) -> None:
    features = _write_panel(tmp_path)
    eval_cfg, data_cfg = _cfg(tmp_path)
    data_cfg = DataConfig(features_dir=features)
    train = train_baselines(
        model="b0",
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        features_dir=features,
        run_id="test-b0-run",
    )
    assert train.run_dir.exists()
    assert (train.run_dir / "config.yaml").exists()
    assert (train.run_dir / "forecasts" / "b0" / "fold_0.npz").exists()

    store = RunStore(eval_cfg.artifacts.runs_dir)
    reloaded = store.load_all_batches(train.run_id)
    assert "b0" in reloaded
    assert len(reloaded["b0"]) == 3
    np.testing.assert_allclose(
        reloaded["b0"][0].records[0].raw_w,
        train.batches["b0"][0].records[0].raw_w,
    )

    result = evaluate_run(
        run_id=train.run_id,
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        features_dir=features,
        apply_repair=False,
    )
    assert result.repair_failures == 0
    assert any(m.metric == "mse_w" for m in result.metrics)
    assert (train.run_dir / "summary.json").exists()
    assert (train.run_dir / "metrics" / "long.parquet").exists()


def test_train_and_evaluate_updates_log(tmp_path: Path) -> None:
    features = _write_panel(tmp_path)
    eval_cfg, _ = _cfg(tmp_path)
    data_cfg = DataConfig(features_dir=features)
    train, evaluate = train_and_evaluate(
        model="b0",
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        features_dir=features,
        apply_repair=False,
        update_log=True,
    )
    text = eval_cfg.artifacts.experiment_log_path.read_text(encoding="utf-8")
    assert train.run_id in text
    assert "b0" in text
    assert evaluate.repair_failures == 0


def test_cli_train_help_mentions_baselines() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "baselines" in result.stdout
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--run-id" in result.stdout
