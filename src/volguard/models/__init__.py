"""Layer 4 — baselines (B0-B4) and ML forecasters (Model A/B), training loop."""

from volguard.models.baselines import (
    EWMABaseline,
    PCAVARBaseline,
    PersistenceBaseline,
    RidgeBaseline,
    SVIARBaseline,
    get_baseline,
)
from volguard.models.fold_runner import (
    BaselineModel,
    FoldContext,
    build_fold_context,
    folds_from_manifest,
    run_fold_forecasts,
)
from volguard.models.inputs import EvalPanel, fingerprint_eval_inputs, load_eval_panel
from volguard.models.types import (
    FittedBaseline,
    ForecastBatch,
    ForecastRecord,
    GridSpec,
    MetricRecord,
    RunManifest,
)

__all__ = [
    "BaselineModel",
    "EWMABaseline",
    "EvalPanel",
    "FittedBaseline",
    "FoldContext",
    "ForecastBatch",
    "ForecastRecord",
    "GridSpec",
    "MetricRecord",
    "PCAVARBaseline",
    "PersistenceBaseline",
    "RidgeBaseline",
    "RunManifest",
    "SVIARBaseline",
    "build_fold_context",
    "fingerprint_eval_inputs",
    "folds_from_manifest",
    "get_baseline",
    "load_eval_panel",
    "run_fold_forecasts",
]
