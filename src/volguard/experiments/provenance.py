"""Run-id generation and content hashing helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from volguard.config import REPO_ROOT, EvalConfig


def new_run_id(*, when: datetime | None = None) -> str:
    """UTC timestamp + short random hex suffix."""
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha1(stamp.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{stamp}-{suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return sha256_bytes(encoded)


def config_hash(cfg: EvalConfig) -> str:
    return sha256_json(cfg.model_dump(mode="json"))


def git_commit(repo: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def lockfile_hash(repo: Path = REPO_ROOT) -> str | None:
    lock = repo / "uv.lock"
    if not lock.exists():
        return None
    return sha256_file(lock)


def platform_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "polars", "duckdb", "osqp", "cvxpy", "pydantic"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions
