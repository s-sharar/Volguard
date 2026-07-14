"""Arbitrage / repair / pre-floor diagnostic rows with explicit denominators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from volguard.features.surface_quality import SurfaceDomainQuality
from volguard.surface.arbitrage import ArbitrageReport

DiagKind = Literal[
    "butterfly",
    "vertical",
    "calendar",
    "pre_floor_negative",
    "repair_failure",
    "repair_distance_l2",
    "repair_distance_linf",
]


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    """One diagnostic with an explicit count or magnitude and denominator."""

    model_id: str
    fold_id: int | None
    split: str
    variant: str
    kind: DiagKind
    value: float
    numerator: float
    denominator: float
    scope_key: str | None = None


def arb_rates_from_report(
    report: ArbitrageReport | SurfaceDomainQuality,
    *,
    model_id: str,
    fold_id: int | None,
    split: str,
    variant: str,
    n_days: int,
) -> tuple[DiagnosticRecord, ...]:
    """Convert an arb report into rate diagnostics (violations / days)."""
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    return (
        DiagnosticRecord(
            model_id=model_id,
            fold_id=fold_id,
            split=split,
            variant=variant,
            kind="butterfly",
            value=report.butterfly_violations / n_days,
            numerator=float(report.butterfly_violations),
            denominator=float(n_days),
        ),
        DiagnosticRecord(
            model_id=model_id,
            fold_id=fold_id,
            split=split,
            variant=variant,
            kind="vertical",
            value=report.vertical_violations / n_days,
            numerator=float(report.vertical_violations),
            denominator=float(n_days),
        ),
        DiagnosticRecord(
            model_id=model_id,
            fold_id=fold_id,
            split=split,
            variant=variant,
            kind="calendar",
            value=report.calendar_violations / n_days,
            numerator=float(report.calendar_violations),
            denominator=float(n_days),
        ),
    )
