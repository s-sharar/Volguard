"""Config loading: typed pydantic models backed by YAML files in ``configs/``.

Every pipeline stage takes a config object. Keeping the schema here (rather than
scattered across modules) makes the frozen "contract" between stages explicit,
which is what the parallel-agent workstreams in the plan rely on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Repo root = three levels up from this file: src/volguard/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


class DataConfig(BaseModel):
    """Data ingestion / storage locations and windows.

    Covers all three M2 sources: the Deribit history-API trades backfill, the
    Tardis free-day chain downloads, and the underlying OHLC/DVOL/funding pulls.
    Per-dataset directories are derived from ``raw_dir`` so the layout stays in
    one place (plan section 6).
    """

    currency: str = "BTC"
    history_start: str = "2021-01-01"
    raw_dir: Path = DATA_DIR / "raw"
    curated_dir: Path = DATA_DIR / "curated"
    features_dir: Path = DATA_DIR / "features"
    ref_dir: Path = DATA_DIR / "ref"

    # Deribit history API serves the full trade history since launch. The DVOL,
    # funding, delivery and instrument endpoints are only served by the main
    # host, so underlying pulls use ``underlying_base_url`` instead.
    history_base_url: str = "https://history.deribit.com/api/v2"
    underlying_base_url: str = "https://www.deribit.com/api/v2"
    rate_limit_rps: float = 5.0
    page_count: int = 1000
    request_timeout_s: float = 30.0
    max_retries: int = 5
    retry_backoff_s: float = 1.0

    # Underlying series.
    ohlc_resolution: str = "60"  # minutes per futures/index candle
    dvol_resolution: str = "3600"  # seconds per DVOL candle
    perpetual: str = "BTC-PERPETUAL"
    # Instrument used as the underlying-index OHLC proxy (Deribit exposes no
    # public index candle endpoint; the perp tracks the index closely).
    index_instrument: str = "BTC-PERPETUAL"
    # Explicit dated-future instruments to pull OHLC for; empty => derive the
    # liquid set from the expired-instruments reference table.
    future_instruments: list[str] = Field(default_factory=list)

    # Tardis free-sample option chains (first of every month since 2019-04).
    tardis_start: str = "2019-04-01"
    tardis_base_url: str = "https://datasets.tardis.dev/v1/deribit/options_chain"

    @property
    def checkpoint_dir(self) -> Path:
        """Resumable-backfill checkpoint location."""
        return self.raw_dir / "_checkpoints"

    def raw_table_dir(self, name: str) -> Path:
        """Directory for a raw dataset, e.g. ``raw_table_dir("trades_options")``."""
        return self.raw_dir / name


class SurfaceConfig(BaseModel):
    """Surface-construction parameters (snap time, filters, SVI fit)."""

    snap_hour_utc: int = 8
    snap_minute_utc: int = 5
    min_tau_days: float = 2.0
    delta_min: float = 0.02
    delta_max: float = 0.98
    iv_min: float = 0.01
    iv_max: float = 5.0
    tenor_grid_days: list[float] = Field(default_factory=lambda: [7, 14, 30, 60, 90, 180])
    moneyness_grid: list[float] = Field(
        default_factory=lambda: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    )


class EvalConfig(BaseModel):
    """Walk-forward evaluation settings."""

    initial_train_months: int = 18
    val_months: int = 2
    test_months: int = 2
    step_months: int = 2
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])


class CollectorConfig(BaseModel):
    """Live VPS collector (5-minute Deribit poller) settings."""

    currency: str = "BTC"
    base_url: str = "https://www.deribit.com/api/v2"
    poll_seconds: int = 300
    # Cap ticker fan-out to the most liquid instruments (by 24h volume) to keep
    # each poll cheap; book_summary already returns all instruments in one call.
    max_ticker_instruments: int = 300
    request_timeout_s: float = 15.0
    max_retries: int = 4
    retry_backoff_s: float = 2.0
    # Where the poller writes newline-delimited JSON snapshots before upload.
    out_dir: Path = DATA_DIR / "raw" / "ticker_snapshots"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict. Returns ``{}`` for empty files."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config[C: BaseModel](name: str, model: type[C]) -> C:
    """Load ``configs/<name>.yaml`` and validate it against ``model``.

    Falls back to model defaults if the file does not exist yet, so early
    milestones can run before every config is written.
    """
    path = CONFIGS_DIR / f"{name}.yaml"
    raw = load_yaml(path) if path.exists() else {}
    return model.model_validate(raw)
