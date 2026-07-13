"""Shared surface-stage types (M4).

Holds the small dataclasses shared across the M4 surface modules so that
``ssvi.py``, ``calendar_fit.py``, and ``pipeline.py`` can all import them
without a circular import (``pipeline`` imports ``calendar_fit`` imports
``ssvi``). The canonical per-expiry observation bundle produced by the loader
lives here (design Component 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExpiryObs:
    """One expiry's clean observations for fitting (design Component 1)."""

    expiry: datetime
    tau: float  # years
    k: FloatArray  # log-moneyness ln(K/F)
    iv: FloatArray  # observed IV (fraction)
    vega: FloatArray  # Black-76 vega weights
    n_obs: int
