"""VolGuard command-line interface.

One typer command per pipeline stage (plan section 15). Commands are stubs at
M0 — they parse config and print intent so the wiring and ``--help`` tree are
testable before the stages are implemented in later milestones.
"""

from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console

from volguard import __version__
from volguard.collector.poller import run_forever
from volguard.config import (
    CollectorConfig,
    CurateConfig,
    DataConfig,
    EvalConfig,
    FeatureConfig,
    SurfaceConfig,
    load_config,
)

_INGEST_SOURCES = ("deribit-history", "tardis", "underlying", "all")

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
    source: str = typer.Argument(..., help="Source: deribit-history | tardis | underlying | all"),
    start: str | None = typer.Option(None, help="Override start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="Override end date (YYYY-MM-DD)."),
    kinds: str = typer.Option(
        "option,future",
        help="deribit-history trade kinds to pull: option, future, or both (comma-sep).",
    ),
) -> None:
    """Layer 0 — pull raw data into ``data/raw/`` (M2).

    ``deribit-history`` backfills trades (``--kinds`` selects option/future);
    ``tardis`` downloads free-day option chains; ``underlying`` pulls
    OHLC/DVOL/funding/deliveries. ``all`` runs every source in sequence.
    """
    if source not in _INGEST_SOURCES:
        raise typer.BadParameter(f"source must be one of {', '.join(_INGEST_SOURCES)}")
    trade_kinds = tuple(k.strip() for k in kinds.split(",") if k.strip())
    if any(k not in ("option", "future") for k in trade_kinds):
        raise typer.BadParameter("kinds must be a comma-separated subset of: option, future")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Imported lazily so the CLI (and its --help tree / tests) load without the
    # ingestion stack being importable-heavy.
    from volguard.ingest import deribit_history, tardis_free, underlying  # noqa: PLC0415

    cfg = load_config("data", DataConfig)

    if source in ("deribit-history", "all"):
        console.print(f"[green]ingest[/green]: Deribit trades backfill ({', '.join(trade_kinds)})")
        asyncio.run(deribit_history.run_backfill(cfg, kinds=trade_kinds, start=start, end=end))
    if source in ("tardis", "all"):
        console.print("[green]ingest[/green]: Tardis free-day chains")
        tardis_free.run_tardis(cfg, start=start, end=end)
    if source in ("underlying", "all"):
        console.print("[green]ingest[/green]: underlying OHLC/DVOL/funding")
        asyncio.run(underlying.run_underlying(cfg, start=start, end=end))


@app.command()
def curate(
    start: str | None = typer.Option(None, help="Override start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="Override end date (YYYY-MM-DD)."),
) -> None:
    """Layer 1 — normalize → forwards → IV cross-check → filters → curated/quotes_norm (M3)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Imported lazily so the CLI (and its --help tree / tests) load without the
    # curation stack being importable-heavy (matches the ``ingest`` pattern).
    from volguard.curate import pipeline  # noqa: PLC0415

    data_cfg = load_config("data", DataConfig)
    curate_cfg = load_config("curate", CurateConfig)
    console.print("[green]curate[/green]: building curated/quotes_norm")
    pipeline.run_curate(curate_cfg, data_cfg, start=start, end=end)


@app.command(name="build-surfaces")
def build_surfaces(
    start: str | None = typer.Option(None, help="Override start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="Override end date (YYYY-MM-DD)."),
) -> None:
    """Layer 2 — fit SVI surfaces + grids + QC → curated/surfaces_daily (M4)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Imported lazily so the CLI (and its --help tree / tests) load without the
    # surface stack being importable-heavy (matches the ``curate`` pattern).
    from volguard.surface import pipeline  # noqa: PLC0415

    data_cfg = load_config("data", DataConfig)
    surface_cfg = load_config("surface", SurfaceConfig)
    console.print("[green]build-surfaces[/green]: building curated/surfaces_daily")
    pipeline.run_build_surfaces(surface_cfg, data_cfg, start=start, end=end)


@app.command()
def features(
    start: str | None = typer.Option(None, help="Override start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="Override end date (YYYY-MM-DD)."),
) -> None:
    """Layer 3 — build the feature table for modeling (M5)."""
    from datetime import date  # noqa: PLC0415

    from volguard.features import pipeline  # noqa: PLC0415

    data_cfg = load_config("data", DataConfig)
    surface_cfg = load_config("surface", SurfaceConfig)
    feature_cfg = load_config("features", FeatureConfig)
    eval_cfg = load_config("eval", EvalConfig)
    summary = pipeline.run_features(
        feature_cfg,
        data_cfg,
        surface_cfg,
        eval_cfg,
        start=None if start is None else date.fromisoformat(start),
        end=None if end is None else date.fromisoformat(end),
    )
    console.print(
        f"[green]features[/green]: wrote {summary.accepted_count} dates; "
        f"rejected {summary.rejected_count}"
    )


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
