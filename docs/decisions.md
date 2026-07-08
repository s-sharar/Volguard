# Architecture Decision Records

Short, dated records of non-obvious choices. Newest first.

## ADR-0001 — Scaffold stack (M0)
- **Date:** 2026-07-07
- **Decision:** Python 3.12 pinned via `uv`; Polars/DuckDB/Parquet for data;
  own NumPy/SciPy Black-76 + IV solver (no py_vollib); plain PyTorch (no
  Lightning); cvxpy+OSQP for the repair QP; typer CLI; pydantic+YAML configs.
- **Why:** maximize wheel compatibility and reproducibility on a Windows laptop,
  keep dependencies light, retain full control over walk-forward refits and
  custom no-arbitrage penalties. Heavy deps (torch, cvxpy, plotting, aiohttp)
  are optional extras so `uv sync` and CI stay fast.
