"""Evaluate forecast batches: metrics, skill, regimes, DM, repair, diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from volguard.config import EvalConfig
from volguard.evaluation.diagnostics import DiagnosticRecord
from volguard.evaluation.dm import DMResult, benjamini_hochberg, diebold_mariano
from volguard.evaluation.metrics import (
    aggregate_from_atomic,
    cell_abs_errors,
    cell_squared_errors,
    skill_score,
    weighted_surface_loss,
)
from volguard.evaluation.regimes import RegimeLabel, label_regime
from volguard.evaluation.weights import metric_weights, total_variance_to_iv
from volguard.features.surface_quality import check_model_grid
from volguard.models.fold_runner import FoldContext
from volguard.models.types import ForecastBatch, ForecastRecord, MetricRecord
from volguard.surface.repair import RepairConvergenceError, RepairResult, repair_surface_native

FloatArray = NDArray[np.float64]
RepairStatus = Literal["ok", "failed", "skipped"]
AtomKey = tuple[str, str, str]  # variant, scheme, domain


@dataclass(frozen=True, slots=True)
class DayEvaluation:
    """Per-day raw (and optional repaired) evaluation payload."""

    record: ForecastRecord
    target_w: FloatArray
    target_iv: FloatArray
    pred_iv: FloatArray
    regime: RegimeLabel
    daily_mse_w_vega: float
    daily_mse_iv_vega: float
    repaired_w: FloatArray | None
    repaired_iv: FloatArray | None
    repair_status: RepairStatus
    repair_distance_l2: float | None
    repair_distance_linf: float | None
    pre_floor_negative_count: int
    pre_floor_min: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Full evaluation for one model fold (optionally vs B0)."""

    model_id: str
    fold_id: int
    days: tuple[DayEvaluation, ...]
    metrics: tuple[MetricRecord, ...]
    diagnostics: tuple[DiagnosticRecord, ...]
    dm_vs_b0: tuple[DMResult, ...] = ()
    cell_dm_pvalues: FloatArray | None = None
    cell_dm_pvalues_bh: FloatArray | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class _DayAccumulators:
    atom_sq: dict[AtomKey, list[FloatArray]]
    atom_abs: dict[AtomKey, list[FloatArray]]
    atom_w: dict[AtomKey, list[FloatArray]]
    daily_model_mse: list[float]
    daily_b0_mse: list[float]
    cell_loss_model: list[FloatArray]
    cell_loss_b0: list[FloatArray]
    daily_model_mse_rep: list[float]
    daily_b0_mse_rep: list[float]
    repair_fail: int = 0
    repair_ok: int = 0
    pre_floor_neg: int = 0
    pre_floor_days: int = 0
    dist_l2: list[float] = field(default_factory=list)
    dist_linf: list[float] = field(default_factory=list)
    raw_arb_bfly: int = 0
    raw_arb_vert: int = 0
    raw_arb_cal: int = 0
    rep_arb_bfly: int = 0
    rep_arb_vert: int = 0
    rep_arb_cal: int = 0


def _try_repair(
    ctx: FoldContext,
    record: ForecastRecord,
    cfg: EvalConfig,
) -> tuple[FloatArray | None, RepairStatus, float | None, float | None]:
    # Geometry is frozen from the issue/forecast state's native axes (target k retained
    # for scoring only). Use issue-date k to avoid target leakage into repair.
    issue_idx = ctx.dates.index(record.issue_date)
    k_issue = ctx.grid_k[issue_idx]
    try:
        result: RepairResult = repair_surface_native(
            k_issue,
            record.raw_w,
            ctx.taus,
            max_iter=cfg.repair.max_iter,
            move_tol=cfg.repair.move_tol,
            calendar_points=cfg.repair.calendar_points,
        )
        return (
            result.repaired_w,
            "ok",
            result.repair_distance_l2,
            result.repair_distance_linf,
        )
    except (RepairConvergenceError, ValueError):
        return None, "failed", None, None


def _schemes(cfg: EvalConfig) -> tuple[str, ...]:
    primary = cfg.metrics.primary_weight
    rest = [s for s in cfg.metrics.robustness_weights if s != primary]
    return (primary, *rest)


def _vega_weights(ctx: FoldContext, target_idx: int, target_w: FloatArray) -> FloatArray:
    return metric_weights(
        scheme="vega",
        reliability=ctx.reliability[target_idx],
        k_grid=ctx.grid_k[target_idx],
        w_grid=target_w,
        taus=ctx.taus,
    )


def _append_atoms(
    acc: _DayAccumulators,
    *,
    variant: str,
    schemes: tuple[str, ...],
    reliability: FloatArray,
    target_k: FloatArray,
    target_w: FloatArray,
    target_iv: FloatArray,
    pred_w: FloatArray,
    pred_iv: FloatArray,
    taus: FloatArray,
) -> None:
    for scheme in schemes:
        weights = metric_weights(
            scheme=scheme,
            reliability=reliability,
            k_grid=target_k,
            w_grid=target_w,
            taus=taus,
        )
        for domain, pred, actual in (
            ("w", pred_w, target_w),
            ("iv", pred_iv, target_iv),
        ):
            key: AtomKey = (variant, scheme, domain)
            acc.atom_sq.setdefault(key, []).append(cell_squared_errors(pred, actual))
            acc.atom_abs.setdefault(key, []).append(cell_abs_errors(pred, actual))
            acc.atom_w.setdefault(key, []).append(weights)


def _count_arb(
    k_issue: FloatArray,
    w_grid: FloatArray,
    taus: FloatArray,
    calendar_points: int,
) -> tuple[int, int, int]:
    report = check_model_grid(k_issue, w_grid, taus, calendar_points=calendar_points)
    return report.butterfly_violations, report.vertical_violations, report.calendar_violations


def _evaluate_one_day(
    record: ForecastRecord,
    ctx: FoldContext,
    *,
    cfg: EvalConfig,
    schemes: tuple[str, ...],
    b0_by_target: dict,
    b0_repaired: dict,
    apply_repair: bool,
    acc: _DayAccumulators,
) -> DayEvaluation:
    target_idx = ctx.dates.index(record.target_date)
    target_w = ctx.grid_w[target_idx]
    target_k = ctx.grid_k[target_idx]
    reliability = ctx.reliability[target_idx]
    target_iv = total_variance_to_iv(target_w, ctx.taus)
    pred_iv = total_variance_to_iv(record.raw_w, ctx.taus)
    regime = label_regime(float(ctx.dvol[target_idx]), ctx.dvol_stress_threshold)

    repaired_w = repaired_iv = None
    repair_status: RepairStatus = "skipped"
    dist_l2_v = dist_linf_v = None
    if apply_repair:
        repaired_w, repair_status, dist_l2_v, dist_linf_v = _try_repair(ctx, record, cfg)
        if repair_status == "ok" and repaired_w is not None:
            acc.repair_ok += 1
            repaired_iv = total_variance_to_iv(repaired_w, ctx.taus)
            if dist_l2_v is not None:
                acc.dist_l2.append(dist_l2_v)
            if dist_linf_v is not None:
                acc.dist_linf.append(dist_linf_v)
        elif repair_status == "failed":
            acc.repair_fail += 1

    acc.pre_floor_days += 1
    acc.pre_floor_neg += record.pre_floor_negative_count

    issue_idx = ctx.dates.index(record.issue_date)
    k_issue = ctx.grid_k[issue_idx]
    bfly, vert, cal = _count_arb(k_issue, record.raw_w, ctx.taus, cfg.repair.calendar_points)
    acc.raw_arb_bfly += bfly
    acc.raw_arb_vert += vert
    acc.raw_arb_cal += cal
    if repaired_w is not None:
        bfly, vert, cal = _count_arb(k_issue, repaired_w, ctx.taus, cfg.repair.calendar_points)
        acc.rep_arb_bfly += bfly
        acc.rep_arb_vert += vert
        acc.rep_arb_cal += cal

    vega_w = _vega_weights(ctx, target_idx, target_w)
    loss_w = weighted_surface_loss(record.raw_w, target_w, vega_w)
    loss_iv = weighted_surface_loss(pred_iv, target_iv, vega_w)
    acc.daily_model_mse.append(loss_w.mse)

    if record.target_date in b0_by_target:
        b0_rec = b0_by_target[record.target_date]
        acc.daily_b0_mse.append(weighted_surface_loss(b0_rec.raw_w, target_w, vega_w).mse)
        acc.cell_loss_model.append(cell_squared_errors(record.raw_w, target_w))
        acc.cell_loss_b0.append(cell_squared_errors(b0_rec.raw_w, target_w))

    _append_atoms(
        acc,
        variant="raw",
        schemes=schemes,
        reliability=reliability,
        target_k=target_k,
        target_w=target_w,
        target_iv=target_iv,
        pred_w=record.raw_w,
        pred_iv=pred_iv,
        taus=ctx.taus,
    )
    if repaired_w is not None and repaired_iv is not None:
        _append_atoms(
            acc,
            variant="repaired",
            schemes=schemes,
            reliability=reliability,
            target_k=target_k,
            target_w=target_w,
            target_iv=target_iv,
            pred_w=repaired_w,
            pred_iv=repaired_iv,
            taus=ctx.taus,
        )
        if record.target_date in b0_repaired:
            b0_rep_w = b0_repaired[record.target_date]
            acc.daily_model_mse_rep.append(weighted_surface_loss(repaired_w, target_w, vega_w).mse)
            acc.daily_b0_mse_rep.append(weighted_surface_loss(b0_rep_w, target_w, vega_w).mse)

    return DayEvaluation(
        record=record,
        target_w=target_w,
        target_iv=target_iv,
        pred_iv=pred_iv,
        regime=regime,
        daily_mse_w_vega=loss_w.mse,
        daily_mse_iv_vega=loss_iv.mse,
        repaired_w=repaired_w,
        repaired_iv=repaired_iv,
        repair_status=repair_status,
        repair_distance_l2=dist_l2_v,
        repair_distance_linf=dist_linf_v,
        pre_floor_negative_count=record.pre_floor_negative_count,
        pre_floor_min=record.pre_floor_min,
    )


def _metric_row(
    *,
    model_id: str,
    fold_id: int,
    variant: str,
    scope: str,
    metric: str,
    value: float,
    n: int,
    weight_scheme: str,
    scope_key: str | None = None,
) -> MetricRecord:
    return MetricRecord(
        model_id=model_id,
        fold_id=fold_id,
        split="all",
        variant=variant,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        metric=metric,
        value=value,
        n=n,
        weight_scheme=weight_scheme,
        scope_key=scope_key,
    )


def _overall_metrics(
    batch: ForecastBatch,
    acc: _DayAccumulators,
) -> list[MetricRecord]:
    rows: list[MetricRecord] = []
    for (variant, scheme, domain), sq_list in acc.atom_sq.items():
        agg = aggregate_from_atomic(
            np.stack(sq_list),
            np.stack(acc.atom_abs[(variant, scheme, domain)]),
            np.stack(acc.atom_w[(variant, scheme, domain)]),
        )
        for name, value in (
            (f"mse_{domain}", agg.mse),
            (f"rmse_{domain}", agg.rmse),
            (f"mae_{domain}", agg.mae),
        ):
            rows.append(
                _metric_row(
                    model_id=batch.model_id,
                    fold_id=batch.fold_id,
                    variant=variant,
                    scope="overall",
                    metric=name,
                    value=value,
                    n=agg.n_cells,
                    weight_scheme=scheme,
                )
            )
    return rows


def _regime_metrics(
    batch: ForecastBatch,
    days: list[DayEvaluation],
    ctx: FoldContext,
) -> list[MetricRecord]:
    rows: list[MetricRecord] = []
    for regime_name in ("calm", "stress", "unknown"):
        subset = [d for d in days if d.regime == regime_name]
        if not subset:
            continue
        sq = np.stack([cell_squared_errors(d.record.raw_w, d.target_w) for d in subset])
        ab = np.stack([cell_abs_errors(d.record.raw_w, d.target_w) for d in subset])
        ww = np.stack(
            [_vega_weights(ctx, ctx.dates.index(d.record.target_date), d.target_w) for d in subset]
        )
        agg = aggregate_from_atomic(sq, ab, ww)
        rows.append(
            _metric_row(
                model_id=batch.model_id,
                fold_id=batch.fold_id,
                variant="raw",
                scope="regime",
                metric="mse_w",
                value=agg.mse,
                n=agg.n_cells,
                weight_scheme="vega",
                scope_key=regime_name,
            )
        )
    return rows


def _skill_and_dm(
    batch: ForecastBatch,
    days: list[DayEvaluation],
    ctx: FoldContext,
    cfg: EvalConfig,
    acc: _DayAccumulators,
    b0_by_target: dict,
) -> tuple[list[MetricRecord], list[DMResult], FloatArray | None, FloatArray | None]:
    metrics: list[MetricRecord] = []
    dm_results: list[DMResult] = []
    cell_p = cell_p_bh = None
    if not (acc.daily_b0_mse and len(acc.daily_b0_mse) == len(acc.daily_model_mse)):
        return metrics, dm_results, cell_p, cell_p_bh

    if acc.cell_loss_model:
        m_sq = np.stack(acc.cell_loss_model)
        b_sq = np.stack(acc.cell_loss_b0)
        paired = [d for d in days if d.record.target_date in b0_by_target]
        ww = np.stack(
            [_vega_weights(ctx, ctx.dates.index(d.record.target_date), d.target_w) for d in paired]
        )
        m_agg = aggregate_from_atomic(m_sq, np.sqrt(m_sq), ww)
        b_agg = aggregate_from_atomic(b_sq, np.sqrt(b_sq), ww)
        sk = skill_score(m_agg.mse, b_agg.mse)
    else:
        sk = skill_score(float(np.mean(acc.daily_model_mse)), float(np.mean(acc.daily_b0_mse)))
    metrics.append(
        _metric_row(
            model_id=batch.model_id,
            fold_id=batch.fold_id,
            variant="raw",
            scope="overall",
            metric="skill_mse_w",
            value=sk,
            n=len(acc.daily_model_mse),
            weight_scheme="vega",
        )
    )
    dm_results.append(
        diebold_mariano(
            np.asarray(acc.daily_model_mse),
            np.asarray(acc.daily_b0_mse),
            lag=cfg.significance.dm_lag,
            hln_correction=cfg.significance.hln_correction,
            two_sided=cfg.significance.two_sided,
        )
    )
    if acc.daily_model_mse_rep and len(acc.daily_model_mse_rep) == len(acc.daily_b0_mse_rep):
        metrics.append(
            _metric_row(
                model_id=batch.model_id,
                fold_id=batch.fold_id,
                variant="repaired",
                scope="overall",
                metric="skill_mse_w",
                value=skill_score(
                    float(np.mean(acc.daily_model_mse_rep)),
                    float(np.mean(acc.daily_b0_mse_rep)),
                ),
                n=len(acc.daily_model_mse_rep),
                weight_scheme="vega",
            )
        )
        dm_results.append(
            diebold_mariano(
                np.asarray(acc.daily_model_mse_rep),
                np.asarray(acc.daily_b0_mse_rep),
                lag=cfg.significance.dm_lag,
                hln_correction=cfg.significance.hln_correction,
                two_sided=cfg.significance.two_sided,
            )
        )

    if acc.cell_loss_model:
        m_cells = np.stack(acc.cell_loss_model)
        b_cells = np.stack(acc.cell_loss_b0)
        pvals = np.ones((6, 9), dtype=np.float64)
        for j in range(6):
            for i in range(9):
                try:
                    result = diebold_mariano(
                        m_cells[:, j, i],
                        b_cells[:, j, i],
                        lag=cfg.significance.dm_lag,
                        hln_correction=cfg.significance.hln_correction,
                        two_sided=cfg.significance.two_sided,
                    )
                    pvals[j, i] = result.p_value
                except ValueError:
                    pvals[j, i] = 1.0
        cell_p = pvals
        if cfg.significance.bh_adjust_cells:
            cell_p_bh = benjamini_hochberg(pvals.ravel()).reshape(6, 9)
    return metrics, dm_results, cell_p, cell_p_bh


def _diagnostics(
    batch: ForecastBatch,
    days: list[DayEvaluation],
    acc: _DayAccumulators,
) -> list[DiagnosticRecord]:
    n_days = max(len(days), 1)
    rows = [
        DiagnosticRecord(
            model_id=batch.model_id,
            fold_id=batch.fold_id,
            split="all",
            variant="raw",
            kind="butterfly",
            value=acc.raw_arb_bfly / n_days,
            numerator=float(acc.raw_arb_bfly),
            denominator=float(n_days),
        ),
        DiagnosticRecord(
            model_id=batch.model_id,
            fold_id=batch.fold_id,
            split="all",
            variant="raw",
            kind="vertical",
            value=acc.raw_arb_vert / n_days,
            numerator=float(acc.raw_arb_vert),
            denominator=float(n_days),
        ),
        DiagnosticRecord(
            model_id=batch.model_id,
            fold_id=batch.fold_id,
            split="all",
            variant="raw",
            kind="calendar",
            value=acc.raw_arb_cal / n_days,
            numerator=float(acc.raw_arb_cal),
            denominator=float(n_days),
        ),
        DiagnosticRecord(
            model_id=batch.model_id,
            fold_id=batch.fold_id,
            split="all",
            variant="raw",
            kind="pre_floor_negative",
            value=acc.pre_floor_neg / max(acc.pre_floor_days * 54, 1),
            numerator=float(acc.pre_floor_neg),
            denominator=float(acc.pre_floor_days * 54),
        ),
        DiagnosticRecord(
            model_id=batch.model_id,
            fold_id=batch.fold_id,
            split="all",
            variant="repaired",
            kind="repair_failure",
            value=acc.repair_fail / n_days,
            numerator=float(acc.repair_fail),
            denominator=float(n_days),
        ),
    ]
    if acc.dist_l2:
        rows.append(
            DiagnosticRecord(
                model_id=batch.model_id,
                fold_id=batch.fold_id,
                split="all",
                variant="repaired",
                kind="repair_distance_l2",
                value=float(np.mean(acc.dist_l2)),
                numerator=float(np.sum(acc.dist_l2)),
                denominator=float(len(acc.dist_l2)),
            )
        )
    if acc.dist_linf:
        rows.append(
            DiagnosticRecord(
                model_id=batch.model_id,
                fold_id=batch.fold_id,
                split="all",
                variant="repaired",
                kind="repair_distance_linf",
                value=float(np.mean(acc.dist_linf)),
                numerator=float(np.sum(acc.dist_linf)),
                denominator=float(len(acc.dist_linf)),
            )
        )
    if acc.repair_ok:
        repaired_kinds: tuple[tuple[Literal["butterfly", "vertical", "calendar"], int], ...] = (
            ("butterfly", acc.rep_arb_bfly),
            ("vertical", acc.rep_arb_vert),
            ("calendar", acc.rep_arb_cal),
        )
        for kind, num in repaired_kinds:
            rows.append(
                DiagnosticRecord(
                    model_id=batch.model_id,
                    fold_id=batch.fold_id,
                    split="all",
                    variant="repaired",
                    kind=kind,
                    value=num / acc.repair_ok,
                    numerator=float(num),
                    denominator=float(acc.repair_ok),
                )
            )
    return rows


def _slice_metrics(
    batch: ForecastBatch,
    days: list[DayEvaluation],
    ctx: FoldContext,
) -> list[MetricRecord]:
    if not days:
        return []
    sq_all = np.stack([cell_squared_errors(d.record.raw_w, d.target_w) for d in days])
    abs_all = np.stack([cell_abs_errors(d.record.raw_w, d.target_w) for d in days])
    w_all = np.stack(
        [_vega_weights(ctx, ctx.dates.index(d.record.target_date), d.target_w) for d in days]
    )
    rows: list[MetricRecord] = []
    for j, tenor in enumerate(ctx.grid_spec.tenors_days):
        agg = aggregate_from_atomic(sq_all[:, j, :], abs_all[:, j, :], w_all[:, j, :])
        rows.append(
            _metric_row(
                model_id=batch.model_id,
                fold_id=batch.fold_id,
                variant="raw",
                scope="tenor",
                metric="mse_w",
                value=agg.mse,
                n=agg.n_cells,
                weight_scheme="vega",
                scope_key=str(tenor),
            )
        )
    for i, money in enumerate(ctx.grid_spec.moneyness):
        agg = aggregate_from_atomic(sq_all[:, :, i], abs_all[:, :, i], w_all[:, :, i])
        rows.append(
            _metric_row(
                model_id=batch.model_id,
                fold_id=batch.fold_id,
                variant="raw",
                scope="moneyness",
                metric="mse_w",
                value=agg.mse,
                n=agg.n_cells,
                weight_scheme="vega",
                scope_key=str(money),
            )
        )
    for j in range(6):
        for i in range(9):
            agg = aggregate_from_atomic(sq_all[:, j, i], abs_all[:, j, i], w_all[:, j, i])
            rows.append(
                _metric_row(
                    model_id=batch.model_id,
                    fold_id=batch.fold_id,
                    variant="raw",
                    scope="cell",
                    metric="mse_w",
                    value=agg.mse,
                    n=agg.n_cells,
                    weight_scheme="vega",
                    scope_key=f"{j}:{i}",
                )
            )
    return rows


def evaluate_fold(
    batch: ForecastBatch,
    ctx: FoldContext,
    *,
    cfg: EvalConfig,
    b0_batch: ForecastBatch | None = None,
    apply_repair: bool = True,
) -> EvaluationResult:
    """Score raw (+ repaired) forecasts against the fold panel targets."""
    if batch.fold_id != ctx.fold.fold_id:
        raise ValueError("ForecastBatch fold_id must match FoldContext")
    if b0_batch is not None and b0_batch.fold_id != batch.fold_id:
        raise ValueError("B0 batch must match fold_id")

    b0_by_target = (
        {rec.target_date: rec for rec in b0_batch.records} if b0_batch is not None else {}
    )
    b0_repaired: dict = {}
    if apply_repair and b0_by_target:
        for target_date, b0_rec in b0_by_target.items():
            repaired, status, _, _ = _try_repair(ctx, b0_rec, cfg)
            if status == "ok" and repaired is not None:
                b0_repaired[target_date] = repaired
    schemes = _schemes(cfg)
    acc = _DayAccumulators(
        atom_sq={},
        atom_abs={},
        atom_w={},
        daily_model_mse=[],
        daily_b0_mse=[],
        cell_loss_model=[],
        cell_loss_b0=[],
        daily_model_mse_rep=[],
        daily_b0_mse_rep=[],
    )
    days = [
        _evaluate_one_day(
            record,
            ctx,
            cfg=cfg,
            schemes=schemes,
            b0_by_target=b0_by_target,
            b0_repaired=b0_repaired,
            apply_repair=apply_repair,
            acc=acc,
        )
        for record in batch.records
    ]

    metrics = _overall_metrics(batch, acc)
    metrics.extend(_regime_metrics(batch, days, ctx))
    skill_rows, dm_results, cell_p, cell_p_bh = _skill_and_dm(
        batch, days, ctx, cfg, acc, b0_by_target
    )
    metrics.extend(skill_rows)
    metrics.extend(_slice_metrics(batch, days, ctx))
    diagnostics = _diagnostics(batch, days, acc)

    return EvaluationResult(
        model_id=batch.model_id,
        fold_id=batch.fold_id,
        days=tuple(days),
        metrics=tuple(metrics),
        diagnostics=tuple(diagnostics),
        dm_vs_b0=tuple(dm_results),
        cell_dm_pvalues=cell_p,
        cell_dm_pvalues_bh=cell_p_bh,
    )
