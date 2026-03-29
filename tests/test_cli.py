"""Tests for mara/cli/run.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from mara.cli.run import app

runner = CliRunner()


def _make_certified_report(query: str = "q", report: str = "the report"):
    from mara.agents.types import CertifiedReport
    from mara.merkle.forest import ForestTree

    return CertifiedReport(
        original_query=query,
        report=report,
        forest_tree=ForestTree(algorithm="sha256"),
        chunks=(),
    )


def test_run_prints_report(tmp_path) -> None:
    certified = _make_certified_report(report="Synthesised output.")

    with (
        patch("mara.cli.run.ResearchConfig", return_value=MagicMock()),
        patch("mara.cli.run.configure_logging"),
        patch("mara.cli.run.run_research", new=AsyncMock(return_value=certified)),
        patch("mara.cli.run.asyncio.run", return_value=certified),
    ):
        result = runner.invoke(app, ["What is AI?"])

    assert result.exit_code == 0
    assert "Synthesised output." in result.output


def test_run_json_output(tmp_path) -> None:
    certified = _make_certified_report(query="Q", report="R")

    with (
        patch("mara.cli.run.ResearchConfig", return_value=MagicMock()),
        patch("mara.cli.run.configure_logging"),
        patch("mara.cli.run.asyncio.run", return_value=certified),
    ):
        result = runner.invoke(app, ["Q", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["original_query"] == "Q"
    assert data["report"] == "R"
    assert "forest_root" in data
    assert "chunk_count" in data


def test_run_config_error_exits_nonzero() -> None:
    with patch("mara.cli.run.ResearchConfig", side_effect=Exception("missing key")):
        result = runner.invoke(app, ["test query"])

    assert result.exit_code == 1
    assert "Configuration error" in result.output
