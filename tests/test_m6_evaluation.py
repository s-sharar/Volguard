"""Metric oracles, DM/HLN/BH, regime labels, and evaluation harness tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from volguard.config import EvalConfig
from volguard.datasets.splits import build_split_manifest
from volguard.datasets.types import Fold
from volguard.evaluation.dm import benjamini_hochberg, diebold_mariano
from volguard.evaluation.harness import evaluate_fold
from volguard.evaluation.metrics import (
    aggregate_from_atomic,
    skill_score,
    weighted_surface_loss,
)
from volguard.evaluation.regimes import label_regime
from volguard.evaluation.weights import metric_weights, total_variance_to_iv
from volguard.models.baselines import PersistenceBaseline
from volguard.models.fold_runner import build_fold_context, run_fold_forecasts
from volguard.models.inputs import BASELINE_FEATURE_NAMES, EvalPanel
from volguard.models.types import FittedBaseline, ForecastBatch, ForecastRecord, GridSpec

_SIGNATURE = "eval-grid-signature"
_TENORS = (7.0, 14.0, 30.0, 60.0, 90.0, 180.0)
_MONEY = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def test_total_variance_to_iv_oracle() -> None:
    taus = np.array([0.25, 0.5, 1.0, 1.0, 1.0, 1.0])
    w = np.full((6, 9), 0.04)
    iv = total_variance_to_iv(w, taus)
    np.testing.assert_allclose(iv[0], 0.4)  # sqrt(0.04/0.25)
    np.testing.assert_allclose(iv[1], np.sqrt(0.08))


def test_weighted_surface_loss_oracle() -> None:
    pred = np.ones((6, 9)) * 0.05
    actual = np.ones((6, 9)) * 0.04
    weights = np.ones((6, 9))
    loss = weighted_surface_loss(pred, actual, weights)
    assert loss.mse == pytest.approx(0.0001)
    assert loss.mae == pytest.approx(0.01)
    assert loss.rmse == pytest.approx(0.01)


def test_aggregate_from_atomics_not_mean_of_rmses() -> None:
    # Two folds with different sizes — mean of RMSEs != RMSE of pooled atomics.
    fold1_sq = np.full(10, 0.01)
    fold2_sq = np.full(90, 0.0001)
    fold1_abs = np.sqrt(fold1_sq)
    fold2_abs = np.sqrt(fold2_sq)
    w1 = np.ones(10)
    w2 = np.ones(90)
    pooled = aggregate_from_atomic(
        np.concatenate([fold1_sq, fold2_sq]),
        np.concatenate([fold1_abs, fold2_abs]),
        np.concatenate([w1, w2]),
    )
    rmse1 = float(np.sqrt(np.mean(fold1_sq)))
    rmse2 = float(np.sqrt(np.mean(fold2_sq)))
    mean_rmse = 0.5 * (rmse1 + rmse2)
    assert pooled.rmse != pytest.approx(mean_rmse)
    assert pooled.mse == pytest.approx((10 * 0.01 + 90 * 0.0001) / 100)


def test_skill_score_oracle() -> None:
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(0.0, 0.0) == 0.0
    assert skill_score(1.0, 0.0) == float("-inf")


def test_regime_labels() -> None:
    assert label_regime(0.5, 0.8) == "calm"
    assert label_regime(0.9, 0.8) == "stress"
    assert label_regime(float("nan"), 0.8) == "unknown"
    assert label_regime(0.9, None) == "unknown"


def test_diebold_mariano_hln_and_identical_series() -> None:
    rng = np.random.default_rng(0)
    bench = rng.normal(size=50)
    model = bench.copy()
    result = diebold_mariano(model, bench, lag=0, hln_correction=True)
    assert result.mean_loss_diff == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.hln_correction is True

    better = bench - 0.5
    result2 = diebold_mariano(better, bench, lag=0, hln_correction=True)
    assert result2.mean_loss_diff < 0.0
    assert result2.p_value < 0.05


def test_benjamini_hochberg_monotone() -> None:
    raw = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
    adjusted = benjamini_hochberg(raw, alpha=0.05)
    assert adjusted.shape == raw.shape
    assert np.all(adjusted >= raw - 1e-15)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-15)


def _panel(n: int = 16) -> EvalPanel:
    dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n))
    grids = np.maximum(0.04 + 0.001 * np.arange(n)[:, None, None] * np.ones((n, 6, 9)), 0.01)
    k = np.zeros_like(grids)
    for t in range(n):
        for j in range(6):
            k[t, j] = np.linspace(-1.0, 1.0, 9)
    features = np.column_stack(
        [
            np.full(n, 0.2),
            np.full(n, 0.21),
            np.full(n, 0.22),
            np.linspace(0.3, 1.0, n),
            np.full(n, 0.01),
        ]
    )
    fold = Fold(
        fold_id=0,
        train_start=dates[0],
        train_end=dates[8],
        validation_start=dates[8],
        validation_end=dates[12],
        test_start=dates[12],
        test_end=dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    return EvalPanel(
        dates=dates,
        grid_spec=GridSpec.from_axes(tenors_days=_TENORS, moneyness=_MONEY, signature=_SIGNATURE),
        grid_w=grids,
        grid_k=k,
        reliability=np.ones_like(grids) * 0.9,
        dvol=features[:, 3],
        features=features,
        feature_names=BASELINE_FEATURE_NAMES,
        split_manifest=build_split_manifest(list(dates), (fold,)),
    )


def test_evaluate_fold_emits_metrics_skill_regimes_and_diagnostics() -> None:
    panel = _panel()
    cfg = EvalConfig()
    fold = Fold(
        fold_id=0,
        train_start=panel.dates[0],
        train_end=panel.dates[8],
        validation_start=panel.dates[8],
        validation_end=panel.dates[12],
        test_start=panel.dates[12],
        test_end=panel.dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    ctx = build_fold_context(panel, fold, cfg)
    model = PersistenceBaseline()
    batch = run_fold_forecasts(model, ctx, cfg=cfg)
    # Synthetic "alternative" model = persistence + constant bias.
    biased_records = []
    for rec in batch.records:
        biased_records.append(
            ForecastRecord(
                model_id="b_bias",
                fold_id=rec.fold_id,
                split=rec.split,
                issue_date=rec.issue_date,
                target_date=rec.target_date,
                raw_w=np.maximum(rec.raw_w + 0.01, 0.0),
                pre_floor_negative_count=0,
                pre_floor_min=0.0,
            )
        )
    biased = ForecastBatch(
        model_id="b_bias",
        fold_id=0,
        records=tuple(biased_records),
        fitted=FittedBaseline(
            model_id="b_bias",
            fold_id=0,
            train_start=fold.train_start,
            train_end=fold.train_end,
            hyperparameters={},
            state={},
        ),
    )
    result = evaluate_fold(biased, ctx, cfg=cfg, b0_batch=batch, apply_repair=False)
    names = {m.metric for m in result.metrics}
    assert "mse_w" in names
    assert "rmse_iv" in names
    assert "mae_w" in names
    assert "skill_mse_w" in names
    skill = next(m for m in result.metrics if m.metric == "skill_mse_w")
    assert skill.value < 0.0  # biased worse than B0
    assert any(m.scope == "regime" for m in result.metrics)
    assert any(m.scope == "tenor" for m in result.metrics)
    assert any(m.scope == "cell" for m in result.metrics)
    assert result.dm_vs_b0
    assert result.dm_vs_b0[0].p_value < 0.05
    assert result.cell_dm_pvalues is not None
    assert result.cell_dm_pvalues_bh is not None
    kinds = {d.kind for d in result.diagnostics}
    assert "butterfly" in kinds
    assert "pre_floor_negative" in kinds


def test_metric_weights_schemes_differ() -> None:
    k = np.linspace(-1, 1, 9)[None, :].repeat(6, axis=0)
    w = np.full((6, 9), 0.04)
    taus = np.array([d / 365.0 for d in _TENORS])
    rel = np.ones((6, 9))
    rel[:, 0] = 0.1
    u = metric_weights(scheme="uniform", reliability=rel, k_grid=k, w_grid=w, taus=taus)
    v = metric_weights(scheme="vega", reliability=rel, k_grid=k, w_grid=w, taus=taus)
    vr = metric_weights(scheme="vega_reliability", reliability=rel, k_grid=k, w_grid=w, taus=taus)
    assert np.allclose(u, 1.0)
    assert not np.allclose(v, u)
    assert np.allclose(vr[:, 0], v[:, 0] * 0.1)
