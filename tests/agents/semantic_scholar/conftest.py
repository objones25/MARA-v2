import pytest

import mara.agents.registry as reg_mod
import mara.agents.semantic_scholar.agent as s2_mod
from mara.agents.registry import AgentRegistration
from mara.agents.semantic_scholar.agent import SemanticScholarAgent


@pytest.fixture(autouse=True)
def register_s2_agent():
    """Re-register SemanticScholarAgent after isolate_registry (parent conftest) clears it."""
    reg_mod._REGISTRY["s2"] = AgentRegistration(cls=SemanticScholarAgent)
    yield


@pytest.fixture(autouse=True)
def reset_s2_lock():
    """Reset the module-level lock singleton between tests."""
    s2_mod._S2_LOCK = None
    yield
    s2_mod._S2_LOCK = None
