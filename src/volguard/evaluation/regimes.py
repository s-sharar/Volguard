"""Calm / stress / unknown regime labels from fold-training DVOL thresholds."""

from __future__ import annotations

from typing import Literal

import numpy as np

RegimeLabel = Literal["calm", "stress", "unknown"]


def label_regime(dvol: float, threshold: float | None) -> RegimeLabel:
    """Stress iff finite target DVOL exceeds the fold-training quantile threshold."""
    if threshold is None or not np.isfinite(dvol):
        return "unknown"
    if float(dvol) > float(threshold):
        return "stress"
    return "calm"
