"""Layer 3 — windowing, walk-forward split generation, leakage checks."""

from volguard.datasets.leakage import assert_no_feature_leakage
from volguard.datasets.pca import fit_surface_pca, transform_surface_pca
from volguard.datasets.schemas import SPLIT_MANIFEST
from volguard.datasets.splits import (
    build_split_manifest,
    generate_walk_forward_folds,
    write_split_manifest,
)
from volguard.datasets.types import Fold, SurfacePCA, WindowedDataset
from volguard.datasets.windows import make_supervised_windows

__all__ = [
    "SPLIT_MANIFEST",
    "Fold",
    "SurfacePCA",
    "WindowedDataset",
    "assert_no_feature_leakage",
    "build_split_manifest",
    "fit_surface_pca",
    "generate_walk_forward_folds",
    "make_supervised_windows",
    "transform_surface_pca",
    "write_split_manifest",
]
