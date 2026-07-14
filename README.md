# VolGuard — Arbitrage-Aware IV Surface Forecasting on Deribit BTC Options

Research-grade system that ingests free Deribit BTC options data, constructs
arbitrage-checked implied-volatility surfaces, forecasts next-day surfaces with
ML under no-arbitrage penalties/constraints, and evaluates those forecasts
economically (hedging + relative-value trading with costs) under strict
walk-forward methodology.

See [`volguard_iv_surface_forecasting_eca4a5d2.plan.md`](./volguard_iv_surface_forecasting_eca4a5d2.plan.md)
for the full plan.

> **Status:** M0–M5 implemented. Collection, ingestion, curation, soft-constrained
> fitted surfaces with QC, daily features, and walk-forward dataset construction
> are available; model training and evaluation remain future milestones.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                 # create .venv and install deps + dev tools
uv run volguard version # sanity check the CLI
uv run pytest           # run the smoke tests
```

The arbitrage-repair QP (cvxpy + osqp) is a core dependency, so a plain
`uv sync` installs everything needed to run the full test suite.

Optional dependency groups (kept out of the default sync to stay fast):

```bash
uv sync --extra ml         # torch (modeling)
uv sync --extra viz        # matplotlib + plotly
uv sync --extra collector  # aiohttp (live poller)
```

## CLI

One command per pipeline stage:

```bash
uv run volguard ingest deribit-history   # Layer 0 — raw data pulls (M2)
uv run volguard build-surfaces           # Layer 2 — SVI surfaces (M4)
uv run volguard features --start 2022-01-01 --end 2022-12-31
                                            # Layer 3 — features + split manifest (M5)
uv run volguard train b0                 # Layer 4 — baselines / ML (M6/M7)
uv run volguard evaluate                 # Layer 5 — metric suite (M6)
uv run volguard backtest                 # Layer 5 — economic eval (M8)
uv run volguard report                   # Layer 6 — figures + memo (M10)
```

The feature stage keeps the raw fitted M4 grid as the canonical model state and
target. It hard-rejects only structurally invalid partitions (wrong 6×9 axes or
signature, missing provenance, or nonfinite/negative values). Residual M4
arbitrage counts, soft-certification failures, model-domain checks, fit quality,
and interpolation/extrapolation are persisted as reason-coded diagnostics and
per-cell reliability weights rather than deleting the date. Accepted dates are
written under `data/features/daily/`; structural rejects and accepted warnings
are audited in `data/features/qc/part.parquet`. Nullable market inputs such as
dated-future basis and open interest remain null with explicit availability,
age, and source-timestamp columns. Walk-forward target membership is stored in
`data/features/splits/part.parquet`; PCA and any fold-calibrated transformations
are fitted only on each fold's training data. The repair QP remains a
post-forecast M6/M7 comparison, not a rewrite of the observed M4 target.

## Repo map

```
configs/            data.yaml, surface.yaml, eval.yaml, models/*.yaml
src/volguard/
  ingest/           Deribit history API, Tardis free days, underlying
  collector/        live 5-min VPS poller (dev only)
  curate/           normalize, forwards, Black-76 IV, filters
  surface/          SVI fit, arbitrage checker, grid, repair QP
  features/         realized vol, surface factors, market state
  datasets/         walk-forward windows/splits, leakage checks
  models/           baselines B0-B4, grid forecaster, constrained decoder
  backtest/         cost model, hedging, relative value, risk gates
  evaluation/       forecast metrics, significance, aggregation
  viz/              surfaces, error maps, PnL tearsheets
  config.py         typed config loading (pydantic + YAML)
  cli.py            typer entry point
tests/              unit / property / golden
data/               gitignored: raw/ curated/ features/
experiments/        gitignored run artifacts + registry.duckdb
```

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pyright             # type-check
uv run pytest --cov        # tests with coverage
```
