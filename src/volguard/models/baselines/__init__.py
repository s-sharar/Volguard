"""Baseline forecasters (B0-B4)."""

from __future__ import annotations

from volguard.models.baselines.b0 import PersistenceBaseline
from volguard.models.baselines.b1 import EWMABaseline
from volguard.models.baselines.b2 import SVIARBaseline
from volguard.models.baselines.b3 import PCAVARBaseline
from volguard.models.baselines.b4 import RidgeBaseline

__all__ = [
    "EWMABaseline",
    "PCAVARBaseline",
    "PersistenceBaseline",
    "RidgeBaseline",
    "SVIARBaseline",
]


def get_baseline(
    model_id: str,
) -> PersistenceBaseline | EWMABaseline | SVIARBaseline | PCAVARBaseline | RidgeBaseline:
    """Resolve a baseline model id to an instance."""
    mapping = {
        "b0": PersistenceBaseline,
        "b1": EWMABaseline,
        "b2": SVIARBaseline,
        "b3": PCAVARBaseline,
        "b4": RidgeBaseline,
    }
    try:
        return mapping[model_id]()
    except KeyError as exc:
        raise ValueError(f"unknown baseline model_id: {model_id}") from exc
