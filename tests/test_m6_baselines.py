"""Mathematical oracles and tuning/leakage checks for baselines B0-B4."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from volguard.config import BaselineConfig, EvalConfig
from volguard.datasets.splits import build_split_manifest
from volguard.datasets.types import Fold
from volguard.models.baselines import (
    EWMABaseline,
    PCAVARBaseline,
    PersistenceBaseline,
    RidgeBaseline,
    SVIARBaseline,
    get_baseline,
)
from volguard.models.baselines.common import ewma_forecast
from volguard.models.fold_runner import build_fold_context, run_fold_forecasts
from volguard.models.inputs import BASELINE_FEATURE_NAMES, EvalPanel
from volguard.models.types import GridSpec

_SIGNATURE = "test-grid-signature"
_TENORS = (7.0, 14.0, 30.0, 60.0, 90.0, 180.0)
_MONEY = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def _fold(n_train: int = 12, n_val: int = 4, n_test: int = 4) -> Fold:
    start = date(2024, 1, 1)
    train_end = start + timedelta(days=n_train)
    val_end = train_end + timedelta(days=n_val)
    test_end = val_end + timedelta(days=n_test)
    return Fold(
        fold_id=0,
        train_start=start,
        train_end=train_end,
        validation_start=train_end,
        validation_end=val_end,
        test_start=val_end,
        test_end=test_end,
        tune_hyperparameters=True,
    )


def _panel_from_grids(grids: np.ndarray, *, features: np.ndarray | None = None) -> EvalPanel:
    n = grids.shape[0]
    dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n))
    if features is None:
        features = np.column_stack(
            [
                np.full(n, 0.2),
                np.full(n, 0.21),
                np.full(n, 0.22),
                np.linspace(0.4, 0.4 + 0.01 * (n - 1), n),
                np.full(n, 0.01),
            ]
        )
    k = np.zeros_like(grids)
    for j, d in enumerate(_MONEY):
        # Approximate native k ~ d * sqrt(w_ATM); ATM col index 4.
        k[:, :, j] = d * np.sqrt(np.maximum(grids[:, :, 4], 1e-8))
    # Ensure strictly increasing k within each row.
    for t in range(n):
        for j in range(6):
            k[t, j] = np.linspace(k[t, j, 0], k[t, j, -1] + 1e-6, 9)
    reliability = np.ones_like(grids) * 0.8

    fold = _fold(n_train=max(n // 2, 4), n_val=max(n // 4, 2), n_test=max(n - n // 2 - n // 4, 2))
    # Clamp fold end to data span.
    data_end = dates[-1] + timedelta(days=1)
    fold = Fold(
        fold_id=0,
        train_start=dates[0],
        train_end=min(fold.train_end, dates[max(2, n // 2)]),
        validation_start=min(fold.train_end, dates[max(2, n // 2)]),
        validation_end=min(fold.validation_end, dates[max(3, (3 * n) // 4)]),
        test_start=min(fold.validation_end, dates[max(3, (3 * n) // 4)]),
        test_end=data_end,
        tune_hyperparameters=True,
    )
    manifest = build_split_manifest(list(dates), (fold,))
    return EvalPanel(
        dates=dates,
        grid_spec=GridSpec.from_axes(tenors_days=_TENORS, moneyness=_MONEY, signature=_SIGNATURE),
        grid_w=grids,
        grid_k=k,
        reliability=reliability,
        dvol=features[:, 3],
        features=features,
        feature_names=BASELINE_FEATURE_NAMES,
        split_manifest=manifest,
    )


def test_get_baseline_resolves_ids() -> None:
    assert get_baseline("b0").model_id == "b0"
    assert get_baseline("b1").model_id == "b1"
    assert get_baseline("b4").model_id == "b4"
    with pytest.raises(ValueError, match="unknown"):
        get_baseline("b9")


def test_b0_is_exact_persistence() -> None:
    rng = np.random.default_rng(0)
    grids = 0.04 + 0.01 * rng.random((20, 6, 9))
    panel = _panel_from_grids(grids)
    cfg = EvalConfig()
    fold = Fold(
        fold_id=0,
        train_start=panel.dates[0],
        train_end=panel.dates[10],
        validation_start=panel.dates[10],
        validation_end=panel.dates[14],
        test_start=panel.dates[14],
        test_end=panel.dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    batch = run_fold_forecasts(PersistenceBaseline(), build_fold_context(panel, fold, cfg), cfg=cfg)
    for rec in batch.records:
        issue_idx = panel.dates.index(rec.issue_date)
        np.testing.assert_allclose(rec.raw_w, panel.grid_w[issue_idx])


def test_b1_recovers_true_lambda_on_ewma_process() -> None:
    true_lam = 0.40
    rng = np.random.default_rng(1)
    innov = 0.002 * rng.normal(size=(40, 6, 9))
    grids = np.zeros((40, 6, 9))
    grids[0] = 0.05
    # Realized process itself is EWMA recursion with shocks (inverse: x_t = s_t when
    # forecast state tracks). Construct path where one-step EWMA at true_lam is exact:
    # set x_{t+1} = ewma_forecast(x_0..x_t, true_lam).
    for t in range(40 - 1):
        grids[t + 1] = ewma_forecast(grids[: t + 1], true_lam) + innov[t + 1]
        grids[t + 1] = np.maximum(grids[t + 1], 1e-4)

    panel = _panel_from_grids(grids)
    cfg = EvalConfig(baselines=BaselineConfig(ewma_lambdas=[0.05, 0.10, 0.20, 0.40, 0.70, 1.0]))
    fold = Fold(
        fold_id=0,
        train_start=panel.dates[0],
        train_end=panel.dates[28],
        validation_start=panel.dates[28],
        validation_end=panel.dates[34],
        test_start=panel.dates[34],
        test_end=panel.dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    ctx = build_fold_context(panel, fold, cfg)
    fitted = EWMABaseline().fit(ctx, cfg=cfg, tune=True, frozen_hyperparameters=None)
    assert fitted.hyperparameters["lambda"] == pytest.approx(true_lam)


def test_b1_frozen_hyperparameters_override_tune() -> None:
    grids = np.maximum(0.04 + 0.001 * np.arange(16)[:, None, None] * np.ones((16, 6, 9)), 0.01)
    panel = _panel_from_grids(grids)
    cfg = EvalConfig()
    fold = Fold(
        fold_id=2,
        train_start=panel.dates[0],
        train_end=panel.dates[8],
        validation_start=panel.dates[8],
        validation_end=panel.dates[12],
        test_start=panel.dates[12],
        test_end=panel.dates[-1] + timedelta(days=1),
        tune_hyperparameters=False,
    )
    ctx = build_fold_context(panel, fold, cfg)
    fitted = EWMABaseline().fit(ctx, cfg=cfg, tune=False, frozen_hyperparameters={"lambda": 0.70})
    assert fitted.hyperparameters["lambda"] == 0.70


def test_b3_recovers_low_rank_var_process() -> None:
    rng = np.random.default_rng(2)
    n_components = 3
    mean = 0.05 * np.ones(54)
    raw = rng.normal(size=(n_components, 54))
    q, _ = np.linalg.qr(raw.T)
    components = q[:, :n_components].T
    a_mat = 0.95 * np.eye(n_components)
    c = np.zeros(n_components)
    scores = np.zeros((30, n_components))
    scores[0] = rng.normal(size=n_components) * 0.02
    for t in range(29):
        scores[t + 1] = c + a_mat @ scores[t]
    grids = (mean + scores @ components).reshape(30, 6, 9)
    grids = np.maximum(grids, 1e-4)

    panel = _panel_from_grids(grids)
    cfg = EvalConfig(baselines=BaselineConfig(pca_components=[3, 4, 5]))
    fold = Fold(
        fold_id=0,
        train_start=panel.dates[0],
        train_end=panel.dates[20],
        validation_start=panel.dates[20],
        validation_end=panel.dates[24],
        test_start=panel.dates[24],
        test_end=panel.dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    ctx = build_fold_context(panel, fold, cfg)
    model = PCAVARBaseline()
    # Oracle: with the true component count frozen, reconstructions are exact.
    fitted = model.fit(ctx, cfg=cfg, tune=False, frozen_hyperparameters={"n_components": 3})
    assert fitted.hyperparameters["n_components"] == 3
    pred = model.predict_next(
        fitted,
        ctx=ctx,
        history_end=20,
        issue_date=panel.dates[19],
        target_date=panel.dates[20],
        cfg=cfg,
    )
    persist = panel.grid_w[19]
    target = panel.grid_w[20]
    pred_mse = float(np.mean((pred - target) ** 2))
    persist_mse = float(np.mean((persist - target) ** 2))
    assert pred_mse < persist_mse
    np.testing.assert_allclose(pred, target, rtol=5e-2, atol=5e-3)

    tuned = model.fit(ctx, cfg=cfg, tune=True, frozen_hyperparameters=None)
    assert tuned.hyperparameters["n_components"] in {3, 4, 5}


def test_b4_recovers_declared_ridge_design() -> None:
    # Construct a process: each cell is alpha-stable ridge of known features.
    n = 36
    # Use a simpler exact linear map: next flat grid = 0.9 * current flat + 0.01 * dvol.
    grids = np.zeros((n, 6, 9))
    grids[0] = 0.05
    dvol = np.linspace(0.4, 0.7, n)
    for t in range(n - 1):
        grids[t + 1] = 0.9 * grids[t] + 0.01 * dvol[t]
        grids[t + 1] = np.maximum(grids[t + 1], 1e-4)
    features = np.column_stack(
        [
            np.full(n, 0.2),
            np.full(n, 0.21),
            np.full(n, 0.22),
            dvol,
            np.full(n, 0.0),
        ]
    )
    panel = _panel_from_grids(grids, features=features)
    cfg = EvalConfig(
        baselines=BaselineConfig(
            ridge_alphas=[1e-6, 1e-4, 1e-2, 1.0, 100.0],
            ridge_mean_windows=[5, 22],
        )
    )
    fold = Fold(
        fold_id=0,
        train_start=panel.dates[0],
        train_end=panel.dates[24],
        validation_start=panel.dates[24],
        validation_end=panel.dates[30],
        test_start=panel.dates[30],
        test_end=panel.dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    ctx = build_fold_context(panel, fold, cfg)
    model = RidgeBaseline()
    fitted = model.fit(ctx, cfg=cfg, tune=True, frozen_hyperparameters=None)
    # Small alpha should win on this near-linear process.
    assert fitted.hyperparameters["alpha"] in {1e-6, 1e-4, 1e-2}
    pred = model.predict_next(
        fitted,
        ctx=ctx,
        history_end=24,
        issue_date=panel.dates[23],
        target_date=panel.dates[24],
        cfg=cfg,
    )
    np.testing.assert_allclose(pred, panel.grid_w[24], rtol=5e-2, atol=5e-3)


def test_b2_fits_and_forecasts_without_raising() -> None:
    # Mild smiles across tenors — enough for SVI fits to succeed most days.
    n = 18
    grids = np.zeros((n, 6, 9))
    for t in range(n):
        for j in range(6):
            atm = 0.04 + 0.001 * t + 0.01 * j
            smile = atm * (1.0 + 0.05 * np.asarray(_MONEY) ** 2)
            grids[t, j] = smile
    panel = _panel_from_grids(grids)
    cfg = EvalConfig()
    fold = Fold(
        fold_id=0,
        train_start=panel.dates[0],
        train_end=panel.dates[10],
        validation_start=panel.dates[10],
        validation_end=panel.dates[14],
        test_start=panel.dates[14],
        test_end=panel.dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    ctx = build_fold_context(panel, fold, cfg)
    model = SVIARBaseline()
    fitted = model.fit(ctx, cfg=cfg, tune=True, frozen_hyperparameters=None)
    pred = model.predict_next(
        fitted,
        ctx=ctx,
        history_end=10,
        issue_date=panel.dates[9],
        target_date=panel.dates[10],
        cfg=cfg,
    )
    assert pred.shape == (6, 9)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0)
    assert "fallback_tenors" in model.last_diagnostics


def test_future_poison_does_not_change_earlier_b1_forecasts() -> None:
    grids = np.maximum(0.04 + 0.001 * np.arange(20)[:, None, None] * np.ones((20, 6, 9)), 0.01)
    clean = _panel_from_grids(grids)
    poisoned_grids = grids.copy()
    poisoned_grids[-1] += 5.0
    poisoned = _panel_from_grids(poisoned_grids)
    fold = Fold(
        fold_id=0,
        train_start=clean.dates[0],
        train_end=clean.dates[10],
        validation_start=clean.dates[10],
        validation_end=clean.dates[14],
        test_start=clean.dates[14],
        test_end=clean.dates[-1] + timedelta(days=1),
        tune_hyperparameters=True,
    )
    cfg = EvalConfig()
    model = EWMABaseline()
    a = run_fold_forecasts(model, build_fold_context(clean, fold, cfg), cfg=cfg)
    b = run_fold_forecasts(model, build_fold_context(poisoned, fold, cfg), cfg=cfg)
    for ra, rb in zip(a.records, b.records, strict=True):
        if ra.target_date < clean.dates[-1]:
            np.testing.assert_allclose(ra.raw_w, rb.raw_w)
