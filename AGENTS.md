# AGENTS.md

Universal agent guidance for this project. Read by Claude Code, Cursor, Codex, and Antigravity.

## Working rules
- Make small, focused diffs. Do not modify unrelated files.
- Ask before adding new production dependencies.
- Never touch secrets, `.env` files, keys, or credentials.
- Run the smallest relevant test/typecheck after code changes.
- Do not commit unless explicitly asked.
- Prefer editing existing files over creating new ones.

## Project context

VolGuard forecasts arbitrage-free implied-volatility surfaces for Deribit BTC
options and evaluates the forecasts economically. Full plan lives in
`volguard_iv_surface_forecasting_eca4a5d2.plan.md`.

- **Stack:** Python 3.12 (via `uv`), Polars + DuckDB + Parquet for data,
  NumPy/SciPy for the math core, PyTorch for models (optional extra), cvxpy+OSQP
  for the arbitrage-repair QP (optional extra), typer CLI, pydantic+YAML configs.
- **Setup:** `uv sync` (add `--extra ml|opt|viz|collector` as needed).
- **Run/dev command:** `uv run volguard <stage>` (ingest / build-surfaces /
  features / train / evaluate / backtest / report).
- **Test command:** `uv run pytest` (add `--cov` for coverage).
- **Lint / types:** `uv run ruff check .` and `uv run pyright`.
- **Notable directories:** `src/volguard/` (one subpackage per pipeline layer),
  `configs/` (frozen stage contracts), `tests/` (unit/property/golden),
  `data/` and `experiments/runs/` are gitignored artifacts.

## Conventions
- Pipeline stages are pure where possible and communicate via Parquet tables
  with pandera schemas as frozen contracts (see plan sections 6, 18).
- The math core (`curate/blackiv.py`, `surface/`) has zero data dependencies and
  is property-tested with hypothesis.
- Never introduce look-ahead: every feature row carries a max-source-timestamp;
  a leakage test asserts it is ≤ snap time.
