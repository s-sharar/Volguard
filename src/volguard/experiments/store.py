"""Filesystem store for append-only experiment run directories."""

from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray

from volguard.experiments.provenance import sha256_file
from volguard.models.types import FittedBaseline, ForecastBatch, ForecastRecord, MetricRecord

FloatArray = NDArray[np.float64]


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


class RunStore:
    """Atomically materialize ``experiments/runs/<run_id>/`` trees."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def create_run_dir(self, run_id: str) -> Path:
        """Create a staging directory then atomically promote it."""
        final = self.run_dir(run_id)
        if final.exists():
            raise FileExistsError(f"run directory already exists: {final}")
        staging = self.runs_dir / f".{run_id}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        (staging / ".created").write_text("", encoding="utf-8")
        os.replace(staging, final)
        return final

    def write_json(self, run_id: str, relative: str, payload: Any) -> Path:
        path = self.run_dir(run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    def read_json(self, run_id: str, relative: str) -> Any:
        path = self.run_dir(run_id) / relative
        return json.loads(path.read_text(encoding="utf-8"))

    def write_yaml(self, run_id: str, relative: str, payload: Any) -> Path:
        path = self.run_dir(run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    def write_parquet(self, run_id: str, relative: str, frame: pl.DataFrame) -> Path:
        path = self.run_dir(run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        frame.write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
        return path

    def write_npz(self, run_id: str, relative: str, arrays: dict[str, FloatArray]) -> Path:
        path = self.run_dir(run_id) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(str(temporary), **arrays)  # type: ignore[arg-type]
        os.replace(temporary, path)
        return path

    def artifact_digest(self, run_id: str, relative: str) -> str:
        return sha256_file(self.run_dir(run_id) / relative)

    def save_forecast_batch(self, run_id: str, batch: ForecastBatch) -> Path:
        rows = []
        arrays: dict[str, FloatArray] = {}
        for index, record in enumerate(batch.records):
            key = f"raw_{index}"
            arrays[key] = np.asarray(record.raw_w, dtype=np.float64)
            rows.append(
                {
                    "model_id": record.model_id,
                    "fold_id": record.fold_id,
                    "split": record.split,
                    "issue_date": record.issue_date.isoformat(),
                    "target_date": record.target_date.isoformat(),
                    "array_key": key,
                    "pre_floor_negative_count": record.pre_floor_negative_count,
                    "pre_floor_min": record.pre_floor_min,
                }
            )
        relative = f"forecasts/{batch.model_id}/fold_{batch.fold_id}.npz"
        path = self.write_npz(run_id, relative, arrays)
        meta_rel = f"forecasts/{batch.model_id}/fold_{batch.fold_id}_index.parquet"
        self.write_parquet(run_id, meta_rel, pl.DataFrame(rows))
        self.write_json(
            run_id,
            f"forecasts/{batch.model_id}/fold_{batch.fold_id}_batch.json",
            {
                "model_id": batch.model_id,
                "fold_id": batch.fold_id,
                "fitted_path": f"models/{batch.model_id}/fold_{batch.fold_id}.json",
                "n_records": len(batch.records),
            },
        )
        return path

    def load_forecast_batch(self, run_id: str, model_id: str, fold_id: int) -> ForecastBatch:
        meta = pl.read_parquet(
            self.run_dir(run_id) / f"forecasts/{model_id}/fold_{fold_id}_index.parquet"
        )
        arrays = np.load(self.run_dir(run_id) / f"forecasts/{model_id}/fold_{fold_id}.npz")
        fitted_payload = self.read_json(run_id, f"models/{model_id}/fold_{fold_id}.json")
        fitted = FittedBaseline(
            model_id=fitted_payload["model_id"],
            fold_id=int(fitted_payload["fold_id"]),
            train_start=date.fromisoformat(fitted_payload["train_start"]),
            train_end=date.fromisoformat(fitted_payload["train_end"]),
            hyperparameters=fitted_payload["hyperparameters"],
            state=fitted_payload["state"],
        )
        records: list[ForecastRecord] = []
        for row in meta.iter_rows(named=True):
            records.append(
                ForecastRecord(
                    model_id=row["model_id"],
                    fold_id=int(row["fold_id"]),
                    split=row["split"],  # type: ignore[arg-type]
                    issue_date=date.fromisoformat(row["issue_date"]),
                    target_date=date.fromisoformat(row["target_date"]),
                    raw_w=np.asarray(arrays[row["array_key"]], dtype=np.float64),
                    pre_floor_negative_count=int(row["pre_floor_negative_count"]),
                    pre_floor_min=float(row["pre_floor_min"]),
                )
            )
        return ForecastBatch(
            model_id=model_id,
            fold_id=fold_id,
            records=tuple(records),
            fitted=fitted,
        )

    def load_all_batches(self, run_id: str) -> dict[str, list[ForecastBatch]]:
        root = self.run_dir(run_id) / "forecasts"
        if not root.exists():
            return {}
        out: dict[str, list[ForecastBatch]] = {}
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            mid = model_dir.name
            fold_ids = sorted(
                {int(path.stem.split("_")[1]) for path in model_dir.glob("fold_*_index.parquet")}
            )
            out[mid] = [self.load_forecast_batch(run_id, mid, fid) for fid in fold_ids]
        return out

    def save_metrics(self, run_id: str, metrics: list[MetricRecord]) -> Path:
        frame = pl.DataFrame(
            [
                {
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
                for m in metrics
            ]
        )
        return self.write_parquet(run_id, "metrics/long.parquet", frame)

    def list_run_ids(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            path.name
            for path in self.runs_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
