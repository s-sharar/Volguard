"""Dependency-free, fold-local surface PCA."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volguard.datasets.types import SurfacePCA

_GRID_CELLS = 54
_MATRIX_DIMENSIONS = 2


def _matrix(grids: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(grids, dtype=np.float64)
    if matrix.ndim != _MATRIX_DIMENSIONS or matrix.shape[1] != _GRID_CELLS:
        raise ValueError("surface grids must have shape (n_samples, 54)")
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("surface grids must be non-empty and finite")
    return matrix


def fit_surface_pca(
    train_grids: ArrayLike,
    n_components: int = 3,
    *,
    fit_dates: Sequence[date] | None = None,
) -> SurfacePCA:
    """Fit PCA on training grids only and normalize signs deterministically."""
    matrix = _matrix(train_grids)
    if n_components < 1 or n_components > min(matrix.shape):
        raise ValueError("n_components must be between 1 and min(n_samples, 54)")
    if fit_dates is not None and len(fit_dates) != matrix.shape[0]:
        raise ValueError("fit_dates must align with train_grids")
    mean = matrix.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt[:n_components].copy()
    for index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            components[index] = -component
    squared = singular_values**2
    total = float(squared.sum())
    ratio = np.zeros(n_components) if total == 0.0 else squared[:n_components] / total
    if fit_dates:
        fit_start = min(fit_dates)
        fit_end = max(fit_dates) + timedelta(days=1)
    else:
        fit_start = None
        fit_end = None
    return SurfacePCA(
        mean=mean,
        components=components,
        explained_variance_ratio=ratio,
        fit_start=fit_start,
        fit_end=fit_end,
    )


def transform_surface_pca(model: SurfacePCA, grids: ArrayLike) -> NDArray[np.float64]:
    """Project grids with frozen fold-local PCA state."""
    matrix = _matrix(grids)
    return np.asarray((matrix - model.mean) @ model.components.T, dtype=np.float64)
