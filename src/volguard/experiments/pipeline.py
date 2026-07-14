"""Train baselines and evaluate runs end-to-end."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from volguard.config import DataConfig, EvalConfig, load_config
from volguard.evaluation.harness import EvaluationResult, evaluate_fold
from volguard.experiments.provenance import (
    config_hash,
    dependency_versions,
    git_commit,
    lockfile_hash,
    new_run_id,
    platform_info,
)
from volguard.experiments.registry import ExperimentRegistry
from volguard.experiments.store import RunStore
from volguard.models.baselines import get_baseline
from volguard.models.fold_runner import (
    HyperParams,
    build_fold_context,
    folds_from_manifest,
    run_fold_forecasts,
)
from volguard.models.inputs import fingerprint_eval_inputs, load_eval_panel
from volguard.models.types import ForecastBatch, MetricRecord, RunManifest

logger = logging.getLogger(__name__)

BASELINE_IDS = ("b0", "b1", "b2", "b3", "b4")
_FREEZE_START_FOLD = 2


@dataclass(frozen=True, slots=True)
class TrainResult:
    run_id: str
    model_ids: tuple[str, ...]
    run_dir: Path
    manifest: RunManifest
    batches: dict[str, list[ForecastBatch]]


@dataclass(frozen=True, slots=True)
class EvaluateResult:
    run_id: str
    evaluations: dict[str, list[EvaluationResult]]
    metrics: tuple[MetricRecord, ...]
    summary: dict[str, Any]
    repair_failures: int


def _resolve_model_ids(model: str) -> tuple[str, ...]:
    key = model.strip().lower()
    if key in ("baselines", "all"):
        return BASELINE_IDS
    if key in BASELINE_IDS:
        return (key,)
    raise ValueError(f"unknown model id for M6 train: {model}")


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    def _default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if hasattr(value, "isoformat"):
            return value.isoformat()  # type: ignore[no-any-return]
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

    return json.loads(json.dumps(state, default=_default))


def _should_pass_frozen(fold_id: int, frozen: HyperParams | None) -> HyperParams | None:
    """Fold 0 and 1 may tune; later folds always receive frozen hyperparameters."""
    if fold_id >= _FREEZE_START_FOLD:
        if frozen is None:
            raise ValueError("folds 2+ require frozen hyperparameters from fold 0/1")
        return frozen
    return None


def _persist_fold_artifacts(
    *,
    store: RunStore,
    registry: ExperimentRegistry,
    rid: str,
    fold,
    mid: str,
    batch: ForecastBatch,
    freeze: HyperParams | None,
    tuning_trace: list[dict[str, Any]],
) -> None:
    tuning_trace.append(
        {
            "model_id": mid,
            "fold_id": fold.fold_id,
            "tune": fold.tune_hyperparameters and freeze is None,
            "hyperparameters": dict(batch.fitted.hyperparameters),
        }
    )
    state_rel = f"models/{mid}/fold_{fold.fold_id}.json"
    store.write_json(
        rid,
        state_rel,
        {
            "model_id": mid,
            "fold_id": fold.fold_id,
            "hyperparameters": dict(batch.fitted.hyperparameters),
            "state": _serialize_state(dict(batch.fitted.state)),
            "train_start": batch.fitted.train_start.isoformat(),
            "train_end": batch.fitted.train_end.isoformat(),
        },
    )
    store.save_forecast_batch(rid, batch)
    registry.upsert_fold(
        run_id=rid,
        fold_id=fold.fold_id,
        model_id=mid,
        train_start=fold.train_start,
        train_end=fold.train_end,
        validation_start=fold.validation_start,
        validation_end=fold.validation_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        tune_hyperparameters=fold.tune_hyperparameters,
    )
    registry.upsert_fit(
        run_id=rid,
        fold_id=fold.fold_id,
        model_id=mid,
        hyperparameters=dict(batch.fitted.hyperparameters),
        artifact_path=state_rel,
    )
    registry.log_event(
        rid,
        "fold_trained",
        {"model_id": mid, "fold_id": fold.fold_id, "n_forecasts": len(batch.records)},
    )
    logger.info("trained %s fold %s (%s forecasts)", mid, fold.fold_id, len(batch.records))


def train_baselines(
    *,
    model: str = "baselines",
    eval_cfg: EvalConfig | None = None,
    data_cfg: DataConfig | None = None,
    features_dir: Path | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> TrainResult:
    """Fit / forecast requested baselines over all folds; persist artifacts + registry."""
    eval_cfg = eval_cfg or load_config("eval", EvalConfig)
    data_cfg = data_cfg or load_config("data", DataConfig)
    features_root = Path(features_dir or data_cfg.features_dir)
    model_ids = _resolve_model_ids(model)
    seed_value = int(eval_cfg.seeds[0] if seed is None else seed)

    panel = load_eval_panel(features_root)
    fingerprint = fingerprint_eval_inputs(panel)
    folds = folds_from_manifest(panel)
    if not folds:
        raise ValueError("split manifest contains no folds")

    rid = run_id or new_run_id()
    store = RunStore(eval_cfg.artifacts.runs_dir)
    run_dir = store.create_run_dir(rid)
    registry = ExperimentRegistry(eval_cfg.artifacts.registry_path)

    cfg_hash = config_hash(eval_cfg)
    commit = git_commit()
    lock_hash = lockfile_hash()
    plat = platform_info()
    deps = dependency_versions()

    manifest = RunManifest(
        run_id=rid,
        created_at=datetime.now(UTC),
        model_ids=model_ids,
        config_hash=cfg_hash,
        data_fingerprint=fingerprint,
        git_commit=commit,
        seed=seed_value,
    )

    # Always train B0 alongside so skill comparisons have a matched benchmark.
    train_order = model_ids if "b0" in model_ids else ("b0", *model_ids)
    batches: dict[str, list[ForecastBatch]] = {mid: [] for mid in train_order}
    frozen: dict[str, HyperParams | None] = dict.fromkeys(train_order)
    tuning_trace: list[dict[str, Any]] = []

    try:
        registry.register_run(
            run_id=rid,
            model_ids=model_ids,
            config_hash=cfg_hash,
            data_fingerprint=fingerprint,
            git_commit=commit,
            lockfile_hash=lock_hash,
            seed=seed_value,
            platform=plat,
            dependencies=deps,
            status="started",
        )
        registry.log_event(rid, "train_started", {"model_ids": list(model_ids)})

        store.write_yaml(rid, "config.yaml", eval_cfg.model_dump(mode="json"))
        store.write_json(
            rid,
            "input_manifest.json",
            {
                "features_dir": str(features_root),
                "data_fingerprint": fingerprint,
                "n_dates": len(panel.dates),
                "date_start": panel.dates[0].isoformat(),
                "date_end": panel.dates[-1].isoformat(),
                "grid_signature": panel.grid_spec.signature,
                "n_folds": len(folds),
            },
        )
        store.write_json(
            rid,
            "provenance.json",
            {
                "git_commit": commit,
                "lockfile_hash": lock_hash,
                "config_hash": cfg_hash,
                "platform": plat,
                "dependencies": deps,
                "seed": seed_value,
                "repair": eval_cfg.repair.model_dump(mode="json"),
            },
        )
        store.write_json(
            rid,
            "run_manifest.json",
            {
                "run_id": manifest.run_id,
                "created_at": manifest.created_at.isoformat(),
                "model_ids": list(manifest.model_ids),
                "config_hash": manifest.config_hash,
                "data_fingerprint": manifest.data_fingerprint,
                "git_commit": manifest.git_commit,
                "seed": manifest.seed,
            },
        )

        for fold in folds:
            ctx = build_fold_context(panel, fold, eval_cfg)
            for mid in train_order:
                freeze = _should_pass_frozen(fold.fold_id, frozen[mid])
                batch = run_fold_forecasts(
                    get_baseline(mid),
                    ctx,
                    cfg=eval_cfg,
                    frozen_hyperparameters=freeze,
                )
                batches[mid].append(batch)
                if fold.fold_id < _FREEZE_START_FOLD:
                    frozen[mid] = dict(batch.fitted.hyperparameters)
                _persist_fold_artifacts(
                    store=store,
                    registry=registry,
                    rid=rid,
                    fold=fold,
                    mid=mid,
                    batch=batch,
                    freeze=freeze,
                    tuning_trace=tuning_trace,
                )

        store.write_json(rid, "tuning_trace.json", tuning_trace)
        registry.set_status(rid, "trained")
        registry.log_event(rid, "train_completed", {"model_ids": list(train_order)})
    except Exception as exc:
        registry.set_status(rid, "failed")
        registry.log_event(rid, "train_failed", {"error": str(exc)})
        raise
    finally:
        registry.close()

    return TrainResult(
        run_id=rid,
        model_ids=model_ids,
        run_dir=run_dir,
        manifest=manifest,
        batches=batches,
    )


def _aggregate_summary(
    evaluations: dict[str, list[EvaluationResult]],
) -> tuple[dict[str, Any], int]:
    summary: dict[str, Any] = {"models": {}}
    total_repair_failures = 0
    for model_id, fold_results in evaluations.items():
        mse_rows: list[float] = []
        skill_rows: list[float] = []
        repair_fail = 0
        for result in fold_results:
            for metric in result.metrics:
                if (
                    metric.scope == "overall"
                    and metric.variant == "raw"
                    and metric.metric == "mse_w"
                    and metric.weight_scheme == "vega"
                ):
                    mse_rows.append(metric.value)
                if (
                    metric.scope == "overall"
                    and metric.metric == "skill_mse_w"
                    and metric.weight_scheme == "vega"
                    and metric.variant == "raw"
                ):
                    skill_rows.append(metric.value)
            for diag in result.diagnostics:
                if diag.kind == "repair_failure":
                    repair_fail += int(diag.numerator)
        total_repair_failures += repair_fail
        summary["models"][model_id] = {
            "mse_w_vega_raw_fold_mean": (
                float(sum(mse_rows) / len(mse_rows)) if mse_rows else None
            ),
            "skill_mse_w_vega_raw_fold_mean": (
                float(sum(skill_rows) / len(skill_rows)) if skill_rows else None
            ),
            "repair_failure_count": repair_fail,
            "n_folds": len(fold_results),
        }
    summary["repair_failure_total"] = total_repair_failures
    return summary, total_repair_failures


def evaluate_run(
    *,
    run_id: str | None = None,
    eval_cfg: EvalConfig | None = None,
    data_cfg: DataConfig | None = None,
    features_dir: Path | None = None,
    apply_repair: bool = True,
    batches: dict[str, list[ForecastBatch]] | None = None,
) -> EvaluateResult:
    """Evaluate a trained run (latest trained/evaluated if ``run_id`` omitted)."""
    eval_cfg = eval_cfg or load_config("eval", EvalConfig)
    data_cfg = data_cfg or load_config("data", DataConfig)
    features_root = Path(features_dir or data_cfg.features_dir)
    registry = ExperimentRegistry(eval_cfg.artifacts.registry_path)
    store = RunStore(eval_cfg.artifacts.runs_dir)

    rid = run_id or registry.latest_successful_run_id()
    if rid is None:
        registry.close()
        raise FileNotFoundError("no trained run found; pass --run-id or train first")
    if batches is None:
        batches = store.load_all_batches(rid)
        if not batches:
            registry.close()
            raise FileNotFoundError(f"no forecast artifacts found for run {rid}")

    panel = load_eval_panel(features_root)
    folds = folds_from_manifest(panel)
    b0_folds = {batch.fold_id: batch for batch in batches.get("b0", [])}
    evaluations: dict[str, list[EvaluationResult]] = {}
    all_metrics: list[MetricRecord] = []
    summary: dict[str, Any] = {}
    repair_failures = 0

    try:
        registry.log_event(rid, "evaluate_started", {"run_id": rid})
        for mid, fold_batches in batches.items():
            evaluations[mid] = []
            for batch in fold_batches:
                fold = next(f for f in folds if f.fold_id == batch.fold_id)
                ctx = build_fold_context(panel, fold, eval_cfg)
                b0_batch = b0_folds.get(batch.fold_id) if mid != "b0" else None
                result = evaluate_fold(
                    batch,
                    ctx,
                    cfg=eval_cfg,
                    b0_batch=b0_batch,
                    apply_repair=apply_repair,
                )
                evaluations[mid].append(result)
                all_metrics.extend(result.metrics)
                registry.log_event(
                    rid,
                    "fold_evaluated",
                    {
                        "model_id": mid,
                        "fold_id": batch.fold_id,
                        "n_metrics": len(result.metrics),
                    },
                )

        summary, repair_failures = _aggregate_summary(evaluations)
        store.save_metrics(rid, all_metrics)
        store.write_json(rid, "summary.json", summary)
        store.write_parquet(
            rid,
            "results/summary_by_model.parquet",
            pl.DataFrame([{"model_id": mid, **vals} for mid, vals in summary["models"].items()]),
        )
        registry.insert_metrics(
            [
                {
                    "run_id": rid,
                    "model_id": m.model_id,
                    "fold_id": m.fold_id,
                    "split": m.split,
                    "variant": m.variant,
                    "scope": m.scope,
                    "scope_key": m.scope_key,
                    "metric": m.metric,
                    "value": m.value,
                    "n": m.n,
                    "weight_scheme": m.weight_scheme,
                }
                for m in all_metrics
            ]
        )
        status = "evaluated" if repair_failures == 0 else "evaluated_with_repair_failures"
        registry.set_status(rid, status)
        registry.log_event(
            rid,
            "evaluate_completed",
            {"repair_failures": repair_failures, "n_metrics": len(all_metrics)},
        )
    except Exception as exc:
        registry.set_status(rid, "evaluate_failed")
        registry.log_event(rid, "evaluate_failed", {"error": str(exc)})
        raise
    finally:
        registry.close()

    return EvaluateResult(
        run_id=rid,
        evaluations=evaluations,
        metrics=tuple(all_metrics),
        summary=summary,
        repair_failures=repair_failures,
    )


def append_experiment_log(
    *,
    eval_cfg: EvalConfig,
    train: TrainResult,
    evaluate: EvaluateResult,
) -> None:
    """Append a concise headline block to ``docs/experiment-log.md``."""
    path = Path(eval_cfg.artifacts.experiment_log_path)
    lines = [
        "",
        f"## {train.run_id}",
        "",
        f"- Models: {', '.join(train.model_ids)}",
        f"- Config hash: `{train.manifest.config_hash[:12]}`",
        f"- Data fingerprint: `{train.manifest.data_fingerprint[:12]}`",
        f"- Git: `{train.manifest.git_commit or 'unknown'}`",
        f"- Repair failures: **{evaluate.repair_failures}**",
        "",
        "| Model | MSE_w (vega, raw, fold-mean) | Skill vs B0 | Repair fails |",
        "|-------|------------------------------|-------------|--------------|",
    ]
    for mid, vals in evaluate.summary["models"].items():
        mse = vals["mse_w_vega_raw_fold_mean"]
        skill = vals["skill_mse_w_vega_raw_fold_mean"]
        mse_s = "—" if mse is None else f"{mse:.6g}"
        skill_s = "—" if skill is None else f"{skill:.4f}"
        lines.append(f"| {mid} | {mse_s} | {skill_s} | {vals['repair_failure_count']} |")
    lines.append("")
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    placeholder = "_(empty - first entries land with M6 baselines)_"
    placeholder_legacy = "_(empty — first entries land with M6 baselines)_"
    if placeholder in existing:
        existing = existing.replace(placeholder, "").rstrip() + "\n"
    elif placeholder_legacy in existing:
        existing = existing.replace(placeholder_legacy, "").rstrip() + "\n"
    path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def train_and_evaluate(
    *,
    model: str = "baselines",
    eval_cfg: EvalConfig | None = None,
    data_cfg: DataConfig | None = None,
    features_dir: Path | None = None,
    apply_repair: bool = True,
    update_log: bool = True,
) -> tuple[TrainResult, EvaluateResult]:
    """Convenience: train then evaluate, optionally updating the experiment log."""
    eval_cfg = eval_cfg or load_config("eval", EvalConfig)
    data_cfg = data_cfg or load_config("data", DataConfig)
    train = train_baselines(
        model=model,
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        features_dir=features_dir,
    )
    evaluate = evaluate_run(
        run_id=train.run_id,
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        features_dir=features_dir,
        apply_repair=apply_repair,
        batches=train.batches,
    )
    if update_log:
        append_experiment_log(eval_cfg=eval_cfg, train=train, evaluate=evaluate)
    return train, evaluate
