import pytest

import mara.agents.biorxiv.agent  # noqa: F401 — ensure @agent decorator runs before snapshot
import mara.agents.registry as reg_mod

# Capture real registration (including config) at import time, before
# isolate_registry (parent conftest) can clear it.
_BIORXIV_REGISTRATION = reg_mod._REGISTRY["biorxiv"]


@pytest.fixture(autouse=True)
def register_biorxiv_agent():
    """Re-register BioRxivAgent after isolate_registry clears the registry."""
    reg_mod._REGISTRY["biorxiv"] = _BIORXIV_REGISTRATION
    yield
