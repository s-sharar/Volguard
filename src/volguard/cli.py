"""VolGuard command-line interface.

One typer command per pipeline stage (plan section 15). Commands are stubs at
M0 — they parse config and print intent so the wiring and ``--help`` tree are
testable before the stages are implemented in later milestones.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from volguard import __version__
from volguard.collector.poller import run_forever
from volguard.config import CollectorConfig, load_config

app = typer.Typer(
    name="volguard",
    help="Arbitrage-aware IV surface forecasting on Deribit BTC options.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _todo(stage: str) -> None:
    console.print(f"[yellow]{stage}[/yellow]: not implemented yet (M0 scaffold).")


@app.command()
def version() -> None:
    """Print the VolGuard version."""
    console.print(f"volguard {__version__}")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Source: deribit-history | tardis | underlying"),
) -> None:
    """Layer 0 — pull raw data into ``data/raw/`` (M2)."""
    _todo(f"ingest[{source}]")


@app.command(name="build-surfaces")
def build_surfaces() -> None:
    """Layer 2 — fit SVI surfaces and write ``data/curated/surfaces_daily`` (M4)."""
    _todo("build-surfaces")


@app.command()
def features() -> None:
    """Layer 3 — build the feature table for modeling (M5)."""
    _todo("features")


@app.command()
def train(
    model: str = typer.Argument(..., help="Model id: b0..b4 | model-a | model-b"),
) -> None:
    """Layer 4 — train a baseline or ML forecaster (M6/M7)."""
    _todo(f"train[{model}]")


@app.command()
def evaluate() -> None:
    """Layer 5 — run the forecast + arbitrage metric suite (M6)."""
    _todo("evaluate")


@app.command()
def backtest() -> None:
    """Layer 5 — hedging + relative-value economic evaluation (M8)."""
    _todo("backtest")


@app.command()
def report() -> None:
    """Layer 6 — render figures and assemble the research memo (M10)."""
    _todo("report")


@app.command()
def collect() -> None:
    """Run the live Deribit poller (M0 collector; deploy on the VPS)."""
    cfg = load_config("collector", CollectorConfig)
    console.print(f"[green]Starting collector[/green]: every {cfg.poll_seconds}s -> {cfg.out_dir}")
    try:
        asyncio.run(run_forever(cfg))
    except KeyboardInterrupt:
        console.print("[yellow]collector stopped[/yellow]")


if __name__ == "__main__":
    app()
