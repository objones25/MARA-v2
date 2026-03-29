import pytest

import mara.agents.pwc.agent  # noqa: F401 — ensure @agent decorator runs before snapshot
import mara.agents.registry as reg_mod

# Capture real registration (including config) at import time, before
# isolate_registry (parent conftest) can clear it.
_PWC_REGISTRATION = reg_mod._REGISTRY["pwc"]


@pytest.fixture(autouse=True)
def register_pwc_agent():
    """Re-register PapersWithCodeAgent after isolate_registry clears the registry."""
    reg_mod._REGISTRY["pwc"] = _PWC_REGISTRATION
    yield
