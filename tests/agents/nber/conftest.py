import pytest

import mara.agents.nber.agent  # noqa: F401 — ensure @agent decorator runs before snapshot
import mara.agents.registry as reg_mod

# Capture real registration (including config) at import time, before
# isolate_registry (parent conftest) can clear it.
_NBER_REGISTRATION = reg_mod._REGISTRY["nber"]


@pytest.fixture(autouse=True)
def register_nber_agent():
    """Re-register NBERAgent after isolate_registry clears the registry."""
    reg_mod._REGISTRY["nber"] = _NBER_REGISTRATION
    yield
