# VolGuard — Arbitrage-Aware IV Surface Forecasting on Deribit BTC Options

Research-grade system that ingests free Deribit BTC options data, constructs
arbitrage-checked implied-volatility surfaces, forecasts next-day surfaces with
ML under no-arbitrage penalties/constraints, and evaluates those forecasts
economically (hedging + relative-value trading with costs) under strict
walk-forward methodology.

See [`volguard_iv_surface_forecasting_eca4a5d2.plan.md`](./volguard_iv_surface_forecasting_eca4a5d2.plan.md)
for the full plan.

> **Status:** M0 scaffold. Pipeline stages are stubs; math core lands in M1.

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
uv run volguard features                 # Layer 3 — feature table (M5)
uv run volguard train b0                 # Layer 4 — baselines / ML (M6/M7)
uv run volguard evaluate                 # Layer 5 — metric suite (M6)
uv run volguard backtest                 # Layer 5 — economic eval (M8)
uv run volguard report                   # Layer 6 — figures + memo (M10)
```

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
