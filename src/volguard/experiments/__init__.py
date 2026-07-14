"""Append-only experiment runs and DuckDB registry (M6)."""

from __future__ import annotations

from volguard.experiments.pipeline import evaluate_run, train_baselines
from volguard.experiments.registry import ExperimentRegistry
from volguard.experiments.store import RunStore

__all__ = [
    "ExperimentRegistry",
    "RunStore",
    "evaluate_run",
    "train_baselines",
]
