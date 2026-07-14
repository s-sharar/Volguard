"""Immutable values returned by the M5 daily feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FeatureRunSummary:
    """Auditable outcome of a bounded feature-pipeline run."""

    accepted_dates: tuple[date, ...]
    rejected_dates: tuple[date, ...]
    daily_dir: Path
    qc_path: Path
    split_manifest_path: Path | None

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_dates)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_dates)
