import pytest

import mara.agents.citation_graph.agent  # noqa: F401 — ensure @agent decorator runs before snapshot
import mara.agents.registry as reg_mod

# Capture real registration (including config) at import time, before
# isolate_registry (parent conftest) can clear it.
_CITATION_GRAPH_REGISTRATION = reg_mod._REGISTRY["citation_graph"]


@pytest.fixture(autouse=True)
def register_citation_graph_agent():
    """Re-register CitationGraphAgent after isolate_registry clears the registry."""
    reg_mod._REGISTRY["citation_graph"] = _CITATION_GRAPH_REGISTRATION
    yield
