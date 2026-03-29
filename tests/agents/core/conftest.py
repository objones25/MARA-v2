import pytest

import mara.agents.core.agent as core_mod
import mara.agents.registry as reg_mod
from mara.agents.core.agent import COREAgent


@pytest.fixture(autouse=True)
def register_core_agent():
    """Re-register COREAgent after isolate_registry (parent conftest) clears it."""
    reg_mod._REGISTRY["core"] = COREAgent
    yield


@pytest.fixture(autouse=True)
def reset_core_lock():
    """Reset the module-level lock singleton between tests."""
    core_mod._CORE_LOCK = None
    yield
    core_mod._CORE_LOCK = None
