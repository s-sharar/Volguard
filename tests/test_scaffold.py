"""M0 smoke tests: the scaffold imports, the CLI runs, and configs load."""

from __future__ import annotations

import importlib

from typer.testing import CliRunner

from volguard import __version__
from volguard.cli import app
from volguard.config import DataConfig, EvalConfig, SurfaceConfig, load_config

runner = CliRunner()


def test_version_string() -> None:
    assert __version__


def test_cli_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "volguard" in result.stdout


def test_cli_help_lists_stages() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for stage in ("ingest", "build-surfaces", "features", "train", "evaluate", "backtest"):
        assert stage in result.stdout


def test_load_surface_config_from_yaml() -> None:
    cfg = load_config("surface", SurfaceConfig)
    assert isinstance(cfg, SurfaceConfig)
    assert cfg.snap_hour_utc == 8
    assert len(cfg.tenor_grid_days) == 6
    assert len(cfg.moneyness_grid) == 9


def test_load_data_and_eval_configs() -> None:
    data = load_config("data", DataConfig)
    assert isinstance(data, DataConfig)
    assert data.currency == "BTC"

    ev = load_config("eval", EvalConfig)
    assert isinstance(ev, EvalConfig)
    assert ev.seeds == [0, 1, 2]


def test_all_subpackages_importable() -> None:
    for mod in (
        "ingest",
        "collector",
        "curate",
        "surface",
        "features",
        "datasets",
        "models",
        "backtest",
        "evaluation",
        "viz",
    ):
        assert importlib.import_module(f"volguard.{mod}")
