"""Walk-forward, fold-local PCA, windowing, and leakage tests for M5."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volguard.config import EvalConfig
from volguard.datasets import (
    Fold,
    assert_no_feature_leakage,
    build_split_manifest,
    fit_surface_pca,
    generate_walk_forward_folds,
    make_supervised_windows,
    transform_surface_pca,
)


def _dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def test_generate_expanding_walk_forward_folds_with_validation_tail() -> None:
    dates = _dates(date(2021, 1, 1), date(2022, 12, 31))

    folds = generate_walk_forward_folds(dates, EvalConfig())

    assert len(folds) == 3
    first, second, third = folds
    assert (first.train_start, first.train_end) == (date(2021, 1, 1), date(2022, 5, 1))
    assert (first.validation_start, first.validation_end) == (
        date(2022, 5, 1),
        date(2022, 7, 1),
    )
    assert (first.test_start, first.test_end) == (date(2022, 7, 1), date(2022, 9, 1))
    assert second.train_start == first.train_start
    assert second.train_end == date(2022, 7, 1)
    assert first.tune_hyperparameters is True
    assert second.tune_hyperparameters is True
    assert third.tune_hyperparameters is False


def test_manifest_membership_is_disjoint_within_each_fold() -> None:
    dates = _dates(date(2021, 1, 1), date(2022, 12, 31))
    folds = generate_walk_forward_folds(dates, EvalConfig())

    manifest = build_split_manifest(dates, folds)

    assert manifest.select(["fold_id", "target_date"]).unique().height == manifest.height
    assert set(manifest["split"].unique()) == {"train", "validation", "test"}
    for fold in folds:
        rows = manifest.filter(pl.col("fold_id") == fold.fold_id)
        for split in ("train", "validation", "test"):
            members = rows.filter(pl.col("split") == split)["target_date"].to_list()
            assert all(fold.split_for(member) == split for member in members)


@given(history_days=st.integers(min_value=610, max_value=1_500))
@settings(max_examples=20)
def test_manifest_disjointness_property(history_days: int) -> None:
    dates = _dates(date(2021, 1, 1), date(2021, 1, 1) + timedelta(days=history_days))
    manifest = build_split_manifest(dates, generate_walk_forward_folds(dates, EvalConfig()))

    assert manifest.select(["fold_id", "target_date"]).unique().height == manifest.height


def test_pca_is_train_only_and_has_deterministic_component_signs() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(40, 54))
    unrelated_future = rng.normal(loc=1_000.0, size=(20, 54))

    model = fit_surface_pca(train, n_components=3)
    same_model = fit_surface_pca(train.copy(), n_components=3)
    future_scores = transform_surface_pca(model, unrelated_future)

    np.testing.assert_allclose(model.mean, same_model.mean)
    np.testing.assert_allclose(model.components, same_model.components)
    assert future_scores.shape == (20, 3)
    for component in model.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0


def _daily_frame(days: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, day in enumerate(days):
        snap = datetime(day.year, day.month, day.day, 8, 5, tzinfo=UTC)
        rows.append(
            {
                "snap_date": day,
                "snap_ts": snap,
                "max_source_ts": snap - timedelta(minutes=5),
                "grid_w": [0.01 + index / 10_000 + cell / 100_000 for cell in range(54)],
                "grid_cell_n_obs": [index + 1] * 54,
                "grid_interp_flag": [cell == 1 for cell in range(54)],
                "grid_extrap_flag": [cell == 2 for cell in range(54)],
                "grid_fit_rmse": [0.001 * (index + 1)] * 54,
                "grid_quality_weight": [max(0.0, 1.0 - index / 20)] * 54,
                "surface_quality_flags": [],
                "dvol": None if index == 1 else 0.5 + index / 100,
                "day_of_week": day.weekday(),
            }
        )
    return pl.DataFrame(rows)


def _covering_fold() -> Fold:
    return generate_walk_forward_folds(_dates(date(2022, 7, 1), date(2024, 8, 31)), EvalConfig())[
        -1
    ]


def test_windows_are_consecutive_and_target_date_sets_membership() -> None:
    days = _dates(date(2024, 1, 1), date(2024, 1, 12))
    daily = _daily_frame(days)
    fold = _covering_fold()
    pca = fit_surface_pca(np.asarray(daily["grid_w"].to_list()), n_components=2, fit_dates=days)

    dataset = make_supervised_windows(daily, fold, pca, lookback_days=3, horizon_days=1)

    assert dataset.x_grid.shape == (9, 3, 6, 9)
    assert dataset.x_grid_quality.shape == (9, 3, 6, 9, 5)
    assert dataset.y_grid.shape == (9, 6, 9)
    assert dataset.y_grid_weight.shape == (9, 6, 9)
    assert dataset.quality_channel_names == (
        "log1p_cell_n_obs",
        "is_interpolated",
        "is_extrapolated",
        "fit_rmse",
        "reliability_weight",
    )
    assert dataset.x_features.shape[-1] == len(dataset.feature_names)
    assert "surface_pca_1" in dataset.feature_names
    dvol_index = dataset.feature_names.index("dvol")
    assert np.isnan(dataset.x_features[0, 1, dvol_index])
    assert not dataset.x_feature_mask[0, 1, dvol_index]
    for input_dates, target, split in zip(
        dataset.input_dates, dataset.target_dates, dataset.splits, strict=True
    ):
        assert input_dates == tuple(input_dates[0] + timedelta(days=offset) for offset in range(3))
        assert target == input_dates[-1] + timedelta(days=1)
        assert fold.split_for(target) == split

    np.testing.assert_allclose(dataset.x_grid_quality[0, 0, :, :, 0], np.log1p(1))
    assert np.all(dataset.x_grid_quality[0, 0, :, :, 1].reshape(-1) == np.eye(1, 54, 1)[0])
    assert np.all(dataset.x_grid_quality[0, 0, :, :, 2].reshape(-1) == np.eye(1, 54, 2)[0])
    np.testing.assert_allclose(dataset.x_grid_quality[0, 0, :, :, 3], 0.001)
    np.testing.assert_allclose(dataset.x_grid_quality[0, 0, :, :, 4], 1.0)
    np.testing.assert_allclose(dataset.y_grid_weight[0], 0.85)
    np.testing.assert_array_equal(
        dataset.x_grid[0].reshape(3, 54), np.asarray(daily["grid_w"].to_list()[:3])
    )
    np.testing.assert_array_equal(
        dataset.y_grid[0].reshape(54), np.asarray(daily["grid_w"].to_list()[3])
    )


def test_target_quality_does_not_leak_into_inputs_and_zero_weight_target_is_retained() -> None:
    days = _dates(date(2024, 1, 1), date(2024, 1, 12))
    daily = _daily_frame(days)
    changed = daily.with_columns(
        pl.when(pl.col("snap_date") == date(2024, 1, 4))
        .then(pl.lit([0.0] * 54))
        .otherwise(pl.col("grid_quality_weight"))
        .alias("grid_quality_weight")
    )
    fold = _covering_fold()
    pca = fit_surface_pca(np.asarray(daily["grid_w"].to_list()), n_components=2, fit_dates=days)

    baseline = make_supervised_windows(daily, fold, pca, lookback_days=3)
    modified = make_supervised_windows(changed, fold, pca, lookback_days=3)

    sample = modified.target_dates.index(date(2024, 1, 4))
    assert np.all(modified.y_grid_weight[sample] == 0.0)
    np.testing.assert_array_equal(modified.x_grid[sample], baseline.x_grid[sample])
    np.testing.assert_array_equal(modified.x_grid_quality[sample], baseline.x_grid_quality[sample])
    np.testing.assert_array_equal(modified.y_grid[sample], baseline.y_grid[sample])


def test_quality_list_columns_are_not_scalar_features_but_scalar_qc_is() -> None:
    days = _dates(date(2024, 1, 1), date(2024, 1, 5))
    daily = _daily_frame(days).with_columns(pl.lit(0.25).alias("surface_interp_fraction"))
    fold = _covering_fold()
    pca = fit_surface_pca(np.asarray(daily["grid_w"].to_list()), n_components=2, fit_dates=days)

    dataset = make_supervised_windows(daily, fold, pca, lookback_days=3)

    assert "surface_interp_fraction" in dataset.feature_names
    for name in (
        "grid_cell_n_obs",
        "grid_interp_flag",
        "grid_extrap_flag",
        "grid_fit_rmse",
        "grid_quality_weight",
        "surface_quality_flags",
    ):
        assert name not in dataset.feature_names


def test_empty_window_dataset_has_aligned_quality_shapes() -> None:
    days = _dates(date(2024, 1, 1), date(2024, 1, 2))
    daily = _daily_frame(days)
    pca = fit_surface_pca(np.asarray(daily["grid_w"].to_list()), n_components=1, fit_dates=days)

    dataset = make_supervised_windows(daily, _covering_fold(), pca, lookback_days=3)

    assert dataset.x_grid.shape == (0, 3, 6, 9)
    assert dataset.x_grid_quality.shape == (0, 3, 6, 9, 5)
    assert dataset.y_grid.shape == dataset.y_grid_weight.shape == (0, 6, 9)


def test_missing_calendar_date_breaks_windows() -> None:
    days = _dates(date(2024, 1, 1), date(2024, 1, 12))
    missing = date(2024, 1, 6)
    daily = _daily_frame([day for day in days if day != missing])
    fold = _covering_fold()
    available_days = [day for day in days if day != missing]
    pca = fit_surface_pca(
        np.asarray(daily["grid_w"].to_list()), n_components=2, fit_dates=available_days
    )

    dataset = make_supervised_windows(daily, fold, pca, lookback_days=3, horizon_days=1)

    assert all(
        missing not in (*inputs, target)
        for inputs, target in zip(dataset.input_dates, dataset.target_dates, strict=True)
    )


def test_hard_leakage_check_rejects_future_source_timestamp() -> None:
    daily = _daily_frame([date(2024, 1, 1)]).with_columns(
        (pl.col("snap_ts") + timedelta(seconds=1)).alias("max_source_ts")
    )

    with pytest.raises(ValueError, match="max_source_ts"):
        assert_no_feature_leakage(daily)


@given(future_seconds=st.integers(min_value=1, max_value=86_400))
@settings(max_examples=20)
def test_max_source_timestamp_property(future_seconds: int) -> None:
    daily = _daily_frame([date(2024, 1, 1)]).with_columns(
        (pl.col("snap_ts") + timedelta(seconds=future_seconds)).alias("max_source_ts")
    )

    with pytest.raises(ValueError, match="max_source_ts"):
        assert_no_feature_leakage(daily)


def test_windows_reject_pca_fitted_beyond_train_range() -> None:
    days = _dates(date(2024, 1, 1), date(2024, 1, 12))
    daily = _daily_frame(days)
    fold = _covering_fold()
    future_fit_dates = [fold.train_end + timedelta(days=index) for index in range(len(days))]
    pca = fit_surface_pca(
        np.asarray(daily["grid_w"].to_list()),
        n_components=2,
        fit_dates=future_fit_dates,
    )

    with pytest.raises(ValueError, match="PCA fit interval"):
        make_supervised_windows(daily, fold, pca, lookback_days=3)


def test_windows_require_pca_fit_date_provenance() -> None:
    days = _dates(date(2024, 1, 1), date(2024, 1, 12))
    daily = _daily_frame(days)
    pca = fit_surface_pca(np.asarray(daily["grid_w"].to_list()), n_components=2)

    with pytest.raises(ValueError, match="fit-date provenance"):
        make_supervised_windows(daily, _covering_fold(), pca, lookback_days=3)


@pytest.mark.parametrize("bad_components", [0, 41, 55])
def test_pca_rejects_invalid_component_count(bad_components: int) -> None:
    grids = np.ones((40, 54))
    with pytest.raises(ValueError, match="n_components"):
        fit_surface_pca(grids, n_components=bad_components)
