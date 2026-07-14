"""DuckDB experiment registry: runs, events, folds, fits, artifacts, metrics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    status VARCHAR NOT NULL,
    model_ids VARCHAR NOT NULL,
    config_hash VARCHAR NOT NULL,
    data_fingerprint VARCHAR NOT NULL,
    git_commit VARCHAR,
    lockfile_hash VARCHAR,
    seed INTEGER NOT NULL,
    platform_json VARCHAR NOT NULL,
    dependencies_json VARCHAR NOT NULL,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS events (
    run_id VARCHAR NOT NULL,
    event_ts TIMESTAMP NOT NULL,
    event_type VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS folds (
    run_id VARCHAR NOT NULL,
    fold_id INTEGER NOT NULL,
    model_id VARCHAR NOT NULL,
    train_start DATE,
    train_end DATE,
    validation_start DATE,
    validation_end DATE,
    test_start DATE,
    test_end DATE,
    tune_hyperparameters BOOLEAN,
    PRIMARY KEY (run_id, fold_id, model_id)
);

CREATE TABLE IF NOT EXISTS fits (
    run_id VARCHAR NOT NULL,
    fold_id INTEGER NOT NULL,
    model_id VARCHAR NOT NULL,
    hyperparameters_json VARCHAR NOT NULL,
    artifact_path VARCHAR,
    PRIMARY KEY (run_id, fold_id, model_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id VARCHAR NOT NULL,
    artifact_kind VARCHAR NOT NULL,
    relative_path VARCHAR NOT NULL,
    sha256 VARCHAR,
    PRIMARY KEY (run_id, relative_path)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    fold_id INTEGER,
    split VARCHAR NOT NULL,
    variant VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    scope_key VARCHAR,
    metric VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    n INTEGER NOT NULL,
    weight_scheme VARCHAR NOT NULL
);
"""


class ExperimentRegistry:
    """Transactional registry backed by a single DuckDB file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ExperimentRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def register_run(
        self,
        *,
        run_id: str,
        model_ids: tuple[str, ...],
        config_hash: str,
        data_fingerprint: str,
        git_commit: str | None,
        lockfile_hash: str | None,
        seed: int,
        platform: dict[str, str],
        dependencies: dict[str, str],
        status: str = "started",
        notes: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                datetime.now(UTC).replace(tzinfo=None),
                status,
                ",".join(model_ids),
                config_hash,
                data_fingerprint,
                git_commit,
                lockfile_hash,
                seed,
                json.dumps(platform, sort_keys=True),
                json.dumps(dependencies, sort_keys=True),
                notes,
            ],
        )

    def set_status(self, run_id: str, status: str) -> None:
        self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", [status, run_id])

    def log_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?)",
            [
                run_id,
                datetime.now(UTC).replace(tzinfo=None),
                event_type,
                json.dumps(payload, sort_keys=True, default=str),
            ],
        )

    def upsert_fold(
        self,
        *,
        run_id: str,
        fold_id: int,
        model_id: str,
        train_start,
        train_end,
        validation_start,
        validation_end,
        test_start,
        test_end,
        tune_hyperparameters: bool,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO folds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                fold_id,
                model_id,
                train_start,
                train_end,
                validation_start,
                validation_end,
                test_start,
                test_end,
                tune_hyperparameters,
            ],
        )

    def upsert_fit(
        self,
        *,
        run_id: str,
        fold_id: int,
        model_id: str,
        hyperparameters: dict[str, Any],
        artifact_path: str | None,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fits VALUES (?, ?, ?, ?, ?)
            """,
            [
                run_id,
                fold_id,
                model_id,
                json.dumps(hyperparameters, sort_keys=True, default=str),
                artifact_path,
            ],
        )

    def register_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        relative_path: str,
        sha256: str | None,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?)
            """,
            [run_id, kind, relative_path, sha256],
        )

    def insert_metrics(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    row["run_id"],
                    row["model_id"],
                    row["fold_id"],
                    row["split"],
                    row["variant"],
                    row["scope"],
                    row.get("scope_key"),
                    row["metric"],
                    row["value"],
                    row["n"],
                    row["weight_scheme"],
                ]
                for row in rows
            ],
        )

    def latest_successful_run_id(self) -> str | None:
        frame = self._conn.execute(
            """
            SELECT run_id FROM runs
            WHERE status IN ('trained', 'evaluated', 'evaluated_with_repair_failures')
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        return None if frame is None else str(frame[0])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", [run_id])
        columns = [desc[0] for desc in cur.description]
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(columns, row, strict=True))
