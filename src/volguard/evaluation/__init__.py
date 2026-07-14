"""Layer 5 — forecast metrics, significance tests, aggregation."""

from volguard.evaluation.diagnostics import DiagnosticRecord
from volguard.evaluation.dm import DMResult, benjamini_hochberg, diebold_mariano
from volguard.evaluation.harness import DayEvaluation, EvaluationResult, evaluate_fold
from volguard.evaluation.metrics import (
    SurfaceLoss,
    aggregate_from_atomic,
    skill_score,
    weighted_surface_loss,
)
from volguard.evaluation.regimes import RegimeLabel, label_regime
from volguard.evaluation.weights import metric_weights, total_variance_to_iv

__all__ = [
    "DMResult",
    "DayEvaluation",
    "DiagnosticRecord",
    "EvaluationResult",
    "RegimeLabel",
    "SurfaceLoss",
    "aggregate_from_atomic",
    "benjamini_hochberg",
    "diebold_mariano",
    "evaluate_fold",
    "label_regime",
    "metric_weights",
    "skill_score",
    "total_variance_to_iv",
    "weighted_surface_loss",
]
