"""Load and fingerprint M5 daily features + split manifests for evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from volguard.datasets.schemas import SPLIT_MANIFEST
from volguard.datasets.schemas import validate as validate_manifest
from volguard.models.types import GridSpec

FloatArray = NDArray[np.float64]
_GRID_ROWS = 6
_GRID_COLUMNS = 9
_GRID_CELLS = _GRID_ROWS * _GRID_COLUMNS
_REQUIRED_DAILY = (
    "snap_date",
    "grid_signature",
    "grid_w",
    "grid_k",
    "grid_quality_weight",
    "dvol",
)
# Scalar market features required by B4 (may be null; imputed train-only).
BASELINE_FEATURE_NAMES = (
    "rv_parkinson_1d",
    "rv_parkinson_5d",
    "rv_parkinson_22d",
    "dvol",
    "dvol_change_5d",
)


@dataclass(frozen=True, slots=True)
class EvalPanel:
    """Aligned daily surfaces and fold membership for the M6 harness."""

    dates: tuple[date, ...]
    grid_spec: GridSpec
    grid_w: FloatArray
    grid_k: FloatArray
    reliability: FloatArray
    dvol: FloatArray
    features: FloatArray
    feature_names: tuple[str, ...]
    split_manifest: pl.DataFrame

    def __post_init__(self) -> None:
        n = len(self.dates)
        for name, array in (
            ("grid_w", self.grid_w),
            ("grid_k", self.grid_k),
            ("reliability", self.reliability),
        ):
            if array.shape != (n, _GRID_ROWS, _GRID_COLUMNS):
                raise ValueError(f"{name} must have shape (n, 6, 9)")
        if self.dvol.shape != (n,):
            raise ValueError("dvol must have shape (n,)")
        if self.features.shape != (n, len(self.feature_names)):
            raise ValueError("features must have shape (n, n_features)")
        if self.feature_names != BASELINE_FEATURE_NAMES:
            raise ValueError("feature_names must match the frozen B4 feature contract")
        object.__setattr__(self, "grid_w", _freeze(self.grid_w))
        object.__setattr__(self, "grid_k", _freeze(self.grid_k))
        object.__setattr__(self, "reliability", _freeze(self.reliability))
        object.__setattr__(self, "dvol", _freeze(self.dvol))
        object.__setattr__(self, "features", _freeze(self.features))

    def index_of(self, snap_date: date) -> int:
        """Return the panel index for ``snap_date``."""
        try:
            return self.dates.index(snap_date)
        except ValueError as exc:
            raise KeyError(f"snap_date not in panel: {snap_date}") from exc


def _freeze(value: FloatArray) -> FloatArray:
    copied = np.array(value, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


def _reshape_grid(values: list[float]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.size != _GRID_CELLS:
        raise ValueError(f"grid list must contain {_GRID_CELLS} cells")
    return array.reshape(_GRID_ROWS, _GRID_COLUMNS)


def require_single_grid_spec(daily: pl.DataFrame) -> GridSpec:
    """Require exactly one grid signature across the daily feature table."""
    if "grid_signature" not in daily.columns:
        raise ValueError("daily features missing grid_signature")
    signatures = daily["grid_signature"].unique().to_list()
    if len(signatures) != 1:
        raise ValueError(
            f"daily features must share a single grid_signature; found {len(signatures)}"
        )
    signature = str(signatures[0])
    # Axes are frozen by M5 surface config defaults; signature is the contract key.
    return GridSpec.from_axes(
        tenors_days=(7, 14, 30, 60, 90, 180),
        moneyness=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        signature=signature,
    )


def _read_daily_features(features_dir: Path) -> pl.DataFrame:
    daily_dir = features_dir / "daily"
    if not daily_dir.exists():
        raise FileNotFoundError(f"missing daily features directory: {daily_dir}")
    parts = sorted(daily_dir.glob("date=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(f"no daily feature partitions under {daily_dir}")
    frame = pl.concat([pl.read_parquet(path) for path in parts], how="vertical_relaxed")
    missing = [column for column in _REQUIRED_DAILY if column not in frame.columns]
    if missing:
        raise ValueError(f"daily features missing required columns: {missing}")
    if frame["snap_date"].n_unique() != frame.height:
        raise ValueError("daily features contain duplicate snap_date rows")
    return frame.sort("snap_date")


def _read_split_manifest(features_dir: Path) -> pl.DataFrame:
    path = features_dir / "splits" / "part.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing split manifest: {path}")
    return validate_manifest(pl.read_parquet(path), SPLIT_MANIFEST)


def load_eval_panel(features_dir: str | Path) -> EvalPanel:
    """Load M5 daily features + split manifest into an aligned eval panel."""
    root = Path(features_dir)
    daily = _read_daily_features(root)
    manifest = _read_split_manifest(root)
    grid_spec = require_single_grid_spec(daily)

    dates = tuple(daily["snap_date"].to_list())
    grid_w = np.stack([_reshape_grid(list(row)) for row in daily["grid_w"].to_list()])
    grid_k = np.stack([_reshape_grid(list(row)) for row in daily["grid_k"].to_list()])
    reliability = np.stack(
        [_reshape_grid(list(row)) for row in daily["grid_quality_weight"].to_list()]
    )
    dvol = np.asarray(
        [np.nan if value is None else float(value) for value in daily["dvol"].to_list()],
        dtype=np.float64,
    )
    feature_cols: list[FloatArray] = []
    for name in BASELINE_FEATURE_NAMES:
        if name not in daily.columns:
            raise ValueError(f"daily features missing required column: {name}")
        feature_cols.append(
            np.asarray(
                [np.nan if value is None else float(value) for value in daily[name].to_list()],
                dtype=np.float64,
            )
        )
    features = np.column_stack(feature_cols)
    if not np.all(np.isfinite(grid_w)) or np.any(grid_w < 0.0):
        raise ValueError("grid_w must be finite and nonnegative")
    if not np.all(np.isfinite(grid_k)):
        raise ValueError("grid_k must be finite")
    if not np.all(np.isfinite(reliability)) or np.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError("grid_quality_weight must be finite and in [0, 1]")

    return EvalPanel(
        dates=dates,
        grid_spec=grid_spec,
        grid_w=grid_w,
        grid_k=grid_k,
        reliability=reliability,
        dvol=dvol,
        features=features,
        feature_names=BASELINE_FEATURE_NAMES,
        split_manifest=manifest,
    )


def fingerprint_eval_inputs(panel: EvalPanel) -> str:
    """Stable sha256 over dates, signature, grids, reliability, features, and splits."""
    hasher = hashlib.sha256()
    hasher.update(panel.grid_spec.signature.encode())
    hasher.update(json.dumps([d.isoformat() for d in panel.dates], separators=(",", ":")).encode())
    for array in (panel.grid_w, panel.grid_k, panel.reliability, panel.dvol, panel.features):
        hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    split_payload = panel.split_manifest.select(
        ["fold_id", "target_date", "split", "tune_hyperparameters"]
    ).sort(["fold_id", "target_date"])
    buffer = BytesIO()
    split_payload.write_ipc(buffer)
    hasher.update(buffer.getvalue())
    return hasher.hexdigest()
