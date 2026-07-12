"""Unit tests for the ``curate`` CLI command (design Component 6, R11).

Verify the command is wired into the typer app and that its ``--start`` /
``--end`` overrides are parsed and passed through to ``run_curate`` (the driver
is mocked so the CLI contract is tested in isolation, no data I/O).
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from volguard.cli import app
from volguard.config import CurateConfig, DataConfig
from volguard.curate import pipeline

runner = CliRunner()


def test_cli_help_lists_curate() -> None:
    """R11.1: the ``curate`` command appears in the CLI help tree."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "curate" in result.stdout


def test_curate_passes_start_end_to_run_curate(monkeypatch: pytest.MonkeyPatch) -> None:
    """R11.2/R11.3: ``--start``/``--end`` are parsed and forwarded to run_curate."""
    captured: dict[str, Any] = {}

    def _fake_run_curate(cfg: Any, data_cfg: Any, start: Any, end: Any) -> None:
        captured["cfg"] = cfg
        captured["data_cfg"] = data_cfg
        captured["start"] = start
        captured["end"] = end

    monkeypatch.setattr(pipeline, "run_curate", _fake_run_curate)

    result = runner.invoke(app, ["curate", "--start", "2022-04-01", "--end", "2022-04-01"])
    assert result.exit_code == 0
    assert captured["start"] == "2022-04-01"
    assert captured["end"] == "2022-04-01"
    # The command loads both the data and curate config sections (R11.3).
    assert isinstance(captured["cfg"], CurateConfig)
    assert isinstance(captured["data_cfg"], DataConfig)


def test_curate_defaults_start_end_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """R11.2: with no overrides, start/end default to None (config-driven range)."""
    captured: dict[str, Any] = {}

    def _fake_run_curate(cfg: Any, data_cfg: Any, start: Any, end: Any) -> None:
        captured["start"] = start
        captured["end"] = end

    monkeypatch.setattr(pipeline, "run_curate", _fake_run_curate)

    result = runner.invoke(app, ["curate"])
    assert result.exit_code == 0
    assert captured["start"] is None
    assert captured["end"] is None
