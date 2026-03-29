"""``mara run`` — entry point for the MARA research pipeline CLI."""

from __future__ import annotations

import asyncio
import json

import typer

from mara.agent.graph import run_research
from mara.config import ResearchConfig
from mara.logging import configure_logging

app = typer.Typer(help="MARA — Multi-Agent Research Assistant")


@app.command()
def run(
    query: str = typer.Argument(..., help="Research question to investigate."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the CertifiedReport as JSON instead of plain text."
    ),
) -> None:
    """Run the MARA research pipeline for QUERY."""
    # Agent imports must happen here so @agent() decorators populate _REGISTRY
    # before build_graph() is called inside run_research.
    import mara.agents.arxiv  # noqa: F401
    import mara.agents.biorxiv  # noqa: F401
    import mara.agents.citation_graph  # noqa: F401
    import mara.agents.core  # noqa: F401
    import mara.agents.pubmed  # noqa: F401
    import mara.agents.pwc  # noqa: F401
    import mara.agents.semantic_scholar  # noqa: F401
    import mara.agents.web  # noqa: F401

    try:
        config = ResearchConfig()
    except Exception as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1)

    configure_logging(config.log_level)

    certified = asyncio.run(run_research(query, config))

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "original_query": certified.original_query,
                    "report": certified.report,
                    "forest_root": certified.forest_tree.root,
                    "chunk_count": len(certified.chunks),
                }
            )
        )
    else:
        typer.echo(certified.report)
