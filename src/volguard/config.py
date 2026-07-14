"""Config loading: typed pydantic models backed by YAML files in ``configs/``.

Every pipeline stage takes a config object. Keeping the schema here (rather than
scattered across modules) makes the frozen "contract" between stages explicit,
which is what the parallel-agent workstreams in the plan rely on.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Raw-SVI has five free parameters, so a slice needs at least five points to fit
# (matches the floor in ``surface/fit.py``); a snap needs two expiries for any
# calendar structure; convexity checks need at least three strikes.
_RAW_SVI_MIN_OBS = 5
_MIN_CALENDAR_EXPIRIES = 2
_MIN_ARB_POINTS = 3
_FEATURE_RV_HORIZONS = (1, 5, 22)
_FEATURE_DVOL_CHANGE_DAYS = 5
_OHLC_RESOLUTIONS_MINUTES = frozenset({1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1_440})
# Deribit also exposes one-second DVOL candles, but that cadence is unsafe for
# this full-history batch pipeline and unnecessary for daily features.
_DVOL_RESOLUTIONS_SECONDS = frozenset({60, 3_600, 43_200, 86_400})

# Repo root = three levels up from this file: src/volguard/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _canonical_resolution(
    value: object, allowed: frozenset[int], units_per_day: int, field: str
) -> str:
    normalized = str(value).strip().upper()
    if normalized == "1D":
        units = units_per_day
    else:
        try:
            units = int(normalized)
        except ValueError as exc:
            raise ValueError(f"{field} must be a supported integer or '1D'") from exc
    if units not in allowed:
        raise ValueError(f"{field} is not a supported batch resolution")
    return "1D" if units == units_per_day else str(units)


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

    @field_validator("ohlc_resolution", mode="before")
    @classmethod
    def _normalize_ohlc_resolution(cls, value: object) -> str:
        return _canonical_resolution(value, _OHLC_RESOLUTIONS_MINUTES, 1_440, "ohlc_resolution")

    @field_validator("dvol_resolution", mode="before")
    @classmethod
    def _normalize_dvol_resolution(cls, value: object) -> str:
        return _canonical_resolution(value, _DVOL_RESOLUTIONS_SECONDS, 86_400, "dvol_resolution")

    @property
    def ohlc_resolution_minutes(self) -> int:
        """Configured OHLC candle duration in minutes."""
        return 1_440 if self.ohlc_resolution == "1D" else int(self.ohlc_resolution)

    @property
    def dvol_resolution_seconds(self) -> int:
        """Configured DVOL candle duration in seconds."""
        return 86_400 if self.dvol_resolution == "1D" else int(self.dvol_resolution)

    @property
    def checkpoint_dir(self) -> Path:
        """Resumable-backfill checkpoint location."""
        return self.raw_dir / "_checkpoints"

    def raw_table_dir(self, name: str) -> Path:
        """Directory for a raw dataset, e.g. ``raw_table_dir("trades_options")``."""
        return self.raw_dir / name


class SurfaceConfig(BaseModel):
    """Surface-construction parameters (snap time, filters, SVI fit).

    Carries the M4 fitting/fallback knobs on top of the snap time, bands, and
    output grids: the per-slice observation floors that route between raw-SVI
    and the SSVI fallback, the butterfly reject-and-refit penalties, the
    calendar-ordering penalty, and the fixed-k grid used for arbitrage checks.
    """

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

    # Fitting / fallback knobs (M4).
    min_obs_svi: int = 5  # below this -> SSVI fallback (>= fit.py floor of 5)
    min_obs_slice: int = 3  # below this -> skip the expiry entirely
    min_expiries_per_snap: int = 2  # below this -> skip the snap (no calendar structure)
    butterfly_penalty: float = 10.0  # initial soft penalty (fit.py default)
    butterfly_penalty_escalation: float = 10.0  # multiply penalty on reject-refit
    max_refit_attempts: int = 3
    calendar_penalty: float = 10.0  # weight on w_j(k) < w_{j-1}(k) crossings
    arb_check_k_min: float = -2.0  # fixed-k grid bounds for surface arb checks
    arb_check_k_max: float = 2.0
    arb_check_points: int = 9  # shared calendar k-grid resolution

    @model_validator(mode="after")
    def _check_surface_knobs(self) -> SurfaceConfig:
        """Reject invalid observation floors, penalties, and grid definitions."""
        if self.min_obs_svi < _RAW_SVI_MIN_OBS:
            raise ValueError(
                f"min_obs_svi ({self.min_obs_svi}) must be >= {_RAW_SVI_MIN_OBS} "
                "(raw-SVI needs 5 points)"
            )
        if self.min_obs_slice > self.min_obs_svi:
            raise ValueError(
                f"min_obs_slice ({self.min_obs_slice}) must be <= min_obs_svi ({self.min_obs_svi})"
            )
        if self.min_expiries_per_snap < _MIN_CALENDAR_EXPIRIES:
            raise ValueError(
                f"min_expiries_per_snap ({self.min_expiries_per_snap}) must be "
                f">= {_MIN_CALENDAR_EXPIRIES}"
            )
        if self.max_refit_attempts < 1:
            raise ValueError(f"max_refit_attempts ({self.max_refit_attempts}) must be >= 1")
        if self.butterfly_penalty <= 0:
            raise ValueError(f"butterfly_penalty ({self.butterfly_penalty}) must be > 0")
        if self.butterfly_penalty_escalation <= 1:
            raise ValueError(
                f"butterfly_penalty_escalation ({self.butterfly_penalty_escalation}) must be > 1"
            )
        if self.calendar_penalty < 0:
            raise ValueError(f"calendar_penalty ({self.calendar_penalty}) must be >= 0")
        if not self.arb_check_k_min < self.arb_check_k_max:
            raise ValueError(
                f"arb_check_k_min ({self.arb_check_k_min}) must be "
                f"< arb_check_k_max ({self.arb_check_k_max})"
            )
        if self.arb_check_points < _MIN_ARB_POINTS:
            raise ValueError(
                f"arb_check_points ({self.arb_check_points}) must be >= {_MIN_ARB_POINTS} "
                "(convexity needs 3 strikes)"
            )
        self._check_grids()
        return self

    def _check_grids(self) -> None:
        """Reject empty/non-monotonic tenor and moneyness grids."""
        if not self.tenor_grid_days:
            raise ValueError("tenor_grid_days must be non-empty")
        if any(t <= 0 for t in self.tenor_grid_days):
            raise ValueError(f"tenor_grid_days ({self.tenor_grid_days}) must be strictly positive")
        if any(b <= a for a, b in pairwise(self.tenor_grid_days)):
            raise ValueError(
                f"tenor_grid_days ({self.tenor_grid_days}) must be strictly increasing"
            )
        if any(b <= a for a, b in pairwise(self.moneyness_grid)):
            raise ValueError(f"moneyness_grid ({self.moneyness_grid}) must be strictly increasing")


class CurateConfig(BaseModel):
    """M3 curation parameters: snap window, forward inference, filters.

    Companion to :class:`SurfaceConfig` (which already carries snap time and
    bands); where they overlap M3 reuses the same defaults but adds the
    curation-specific knobs for the snap-window builder, three-tier forward
    inference, and the quality filter cascade (design Model 1).
    """

    # Snap window (plan section 7).
    snap_hour_utc: int = 8
    snap_minute_utc: int = 5
    window_minutes: int = 60  # base [07:05, 08:05]
    widen_step_minutes: int = 60  # widen when sparse
    max_window_minutes: int = 360  # cap widening at 6h back
    min_trades_per_expiry: int = 4  # sparsity trigger for widening
    recency_half_life_s: float = 900.0  # 15-min exp-decay half life

    # Forward inference.
    pcp_pair_window_s: float = 60.0  # max C/P timestamp gap for a PCP pair
    min_pcp_pairs: int = 1

    # Filters (bands mirror SurfaceConfig defaults).
    tau_min_days: float = 2.0
    delta_min: float = 0.02
    delta_max: float = 0.98
    iv_min: float = 0.01
    iv_max: float = 5.0
    mad_multiplier: float = 5.0
    min_size_btc: float = 0.1

    # IV cross-check.
    iv_divergence_tol: float = 0.02  # 2 vol points, in fraction units

    @property
    def tau_min_years(self) -> float:
        """Minimum time-to-expiry in years, derived from ``tau_min_days``."""
        return self.tau_min_days / 365.0

    @model_validator(mode="after")
    def _check_bands_and_positivity(self) -> CurateConfig:
        """Reject invalid band ordering, non-positive knobs, and window bounds."""
        if not self.delta_min < self.delta_max:
            raise ValueError(f"delta_min ({self.delta_min}) must be < delta_max ({self.delta_max})")
        if not self.iv_min < self.iv_max:
            raise ValueError(f"iv_min ({self.iv_min}) must be < iv_max ({self.iv_max})")
        if self.delta_min <= 0 or self.delta_max <= 0:
            raise ValueError("delta band must be strictly positive")
        if self.iv_min <= 0 or self.iv_max <= 0:
            raise ValueError("iv band must be strictly positive")
        if self.window_minutes > self.max_window_minutes:
            raise ValueError(
                f"window_minutes ({self.window_minutes}) must be "
                f"<= max_window_minutes ({self.max_window_minutes})"
            )
        if self.min_trades_per_expiry < 1:
            raise ValueError("min_trades_per_expiry must be >= 1")
        if self.iv_divergence_tol <= 0:
            raise ValueError("iv_divergence_tol must be strictly positive")
        if self.mad_multiplier <= 0:
            raise ValueError("mad_multiplier must be strictly positive")
        if self.recency_half_life_s <= 0:
            raise ValueError("recency_half_life_s must be strictly positive")
        return self


class RepairEvalConfig(BaseModel):
    """Post-forecast arbitrage-repair tolerances (shared + native)."""

    max_iter: int = Field(default=10, gt=0)
    move_tol: float = Field(default=1e-10, gt=0.0)
    calendar_points: int = Field(default=9, ge=3)
    eps: float = Field(default=1e-10, gt=0.0)


class BaselineConfig(BaseModel):
    """Hyperparameter grids for B0-B4 (tuned on fold 0/1 validation only)."""

    ewma_lambdas: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20, 0.40, 0.70, 1.0])
    pca_components: list[int] = Field(default_factory=lambda: [3, 4, 5])
    ridge_alphas: list[float] = Field(default_factory=lambda: [1e-6, 1e-4, 1e-2, 1.0, 100.0])
    ridge_mean_windows: list[int] = Field(default_factory=lambda: [5, 22])
    ridge_rv_horizons: list[int] = Field(default_factory=lambda: [1, 5, 22])

    @model_validator(mode="after")
    def _check_baseline_grids(self) -> BaselineConfig:
        if not self.ewma_lambdas or any(not 0.0 < lam <= 1.0 for lam in self.ewma_lambdas):
            raise ValueError("ewma_lambdas must be in (0, 1]")
        if not self.pca_components or any(n < 1 for n in self.pca_components):
            raise ValueError("pca_components must be positive")
        if not self.ridge_alphas or any(alpha <= 0.0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must be positive")
        if not self.ridge_mean_windows or any(n < 1 for n in self.ridge_mean_windows):
            raise ValueError("ridge_mean_windows must be positive")
        if not self.ridge_rv_horizons or any(n < 1 for n in self.ridge_rv_horizons):
            raise ValueError("ridge_rv_horizons must be positive")
        return self


class MetricsConfig(BaseModel):
    """Primary / robustness weight schemes for the evaluation harness."""

    primary_weight: str = "vega"
    robustness_weights: list[str] = Field(default_factory=lambda: ["uniform", "vega_reliability"])

    @model_validator(mode="after")
    def _check_weight_names(self) -> MetricsConfig:
        allowed = {"vega", "uniform", "vega_reliability"}
        if self.primary_weight not in allowed:
            raise ValueError(f"primary_weight must be one of {sorted(allowed)}")
        if not self.robustness_weights or any(
            name not in allowed for name in self.robustness_weights
        ):
            raise ValueError(f"robustness_weights must be a non-empty subset of {sorted(allowed)}")
        return self


class SignificanceConfig(BaseModel):
    """Diebold-Mariano settings for one-day surface-loss comparisons."""

    dm_lag: int = Field(default=0, ge=0)
    hln_correction: bool = True
    two_sided: bool = True
    bh_adjust_cells: bool = True
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class RegimeConfig(BaseModel):
    """Calm / stress labeling from fold-training DVOL quantiles."""

    stress_dvol_quantile: float = Field(default=0.8, gt=0.0, lt=1.0)


class ArtifactPathsConfig(BaseModel):
    """Append-only experiment artifact locations (gitignored runs, tracked log)."""

    experiments_dir: Path = Field(default_factory=lambda: REPO_ROOT / "experiments")
    registry_filename: str = "registry.duckdb"
    experiment_log_path: Path = Field(
        default_factory=lambda: REPO_ROOT / "docs" / "experiment-log.md"
    )

    @property
    def runs_dir(self) -> Path:
        """Directory for per-run artifact trees."""
        return self.experiments_dir / "runs"

    @property
    def registry_path(self) -> Path:
        """DuckDB registry used to record runs, folds, and metrics."""
        return self.experiments_dir / self.registry_filename


class EvalConfig(BaseModel):
    """Walk-forward settings plus M6 harness / baseline / artifact contracts."""

    initial_train_months: int = 18
    val_months: int = 2
    test_months: int = 2
    step_months: int = 2
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    repair: RepairEvalConfig = Field(default_factory=RepairEvalConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    significance: SignificanceConfig = Field(default_factory=SignificanceConfig)
    regimes: RegimeConfig = Field(default_factory=RegimeConfig)
    artifacts: ArtifactPathsConfig = Field(default_factory=ArtifactPathsConfig)
    # Nonnegative total-variance floor applied before labeling forecasts "raw".
    w_floor: float = Field(default=0.0, ge=0.0)

    @property
    def initial_fit_months(self) -> int:
        """Months fitted before the validation tail in the initial window."""
        return self.initial_train_months - self.val_months

    @model_validator(mode="after")
    def _check_windows(self) -> EvalConfig:
        if self.initial_fit_months < 1:
            raise ValueError("initial_train_months must exceed val_months")
        if self.test_months < 1 or self.step_months < 1:
            raise ValueError("test_months and step_months must be positive")
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(seed < 0 for seed in self.seeds)
        ):
            raise ValueError("seeds must be non-empty, unique, and nonnegative")
        return self


class FeatureConfig(BaseModel):
    """M5 feature, source-staleness, PCA, and supervised-window settings."""

    realized_horizons_days: list[int] = Field(default_factory=lambda: [1, 5, 22])
    dvol_change_days: int = 5
    jump_lookback_days: int = 22
    jump_sigma_threshold: float = 3.0
    ohlc_max_age_s: float = 7_200.0
    dvol_max_age_s: float = 7_200.0
    funding_max_age_s: float = 43_200.0
    basis_max_age_s: float = 86_400.0
    oi_max_age_s: float = 86_400.0
    basis_target_days: int = 30
    basis_min_days: int = 7
    pca_components: int = 3
    lookback_days: int = 20
    forecast_horizon_days: int = 1
    model_domain_calendar_points: int = Field(default=9, ge=3)
    quality_n_obs_reference: int = Field(default=5, gt=0)
    quality_interp_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_extrap_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_rmse_half_life: float = Field(default=0.05, gt=0.0)

    @model_validator(mode="after")
    def _check_feature_knobs(self) -> FeatureConfig:
        horizons = self.realized_horizons_days
        if not horizons or any(day < 1 for day in horizons):
            raise ValueError("realized_horizons_days must contain positive values")
        if any(b <= a for a, b in pairwise(horizons)):
            raise ValueError("realized_horizons_days must be strictly increasing")
        if tuple(horizons) != _FEATURE_RV_HORIZONS:
            raise ValueError(
                "realized_horizons_days must be [1, 5, 22] for the frozen daily feature schema"
            )
        if self.dvol_change_days != _FEATURE_DVOL_CHANGE_DAYS:
            raise ValueError("dvol_change_days must be 5 for the frozen daily feature schema")
        positive_ints = {
            "dvol_change_days": self.dvol_change_days,
            "jump_lookback_days": self.jump_lookback_days,
            "basis_target_days": self.basis_target_days,
            "basis_min_days": self.basis_min_days,
            "pca_components": self.pca_components,
            "lookback_days": self.lookback_days,
            "forecast_horizon_days": self.forecast_horizon_days,
        }
        invalid = [name for name, value in positive_ints.items() if value < 1]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if self.jump_sigma_threshold <= 0:
            raise ValueError("jump_sigma_threshold must be positive")
        ages = (
            self.ohlc_max_age_s,
            self.dvol_max_age_s,
            self.funding_max_age_s,
            self.basis_max_age_s,
            self.oi_max_age_s,
        )
        if any(age < 0 for age in ages):
            raise ValueError("source maximum ages must be nonnegative")
        return self


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
