import pytest

import mara.agents.pubmed.agent as pubmed_mod
import mara.agents.registry as reg_mod
from mara.agents.pubmed.agent import PubMedAgent
from mara.agents.registry import AgentRegistration


@pytest.fixture(autouse=True)
def register_pubmed_agent():
    """Re-register PubMedAgent after isolate_registry (parent conftest) clears it."""
    reg_mod._REGISTRY["pubmed"] = AgentRegistration(cls=PubMedAgent)
    yield


@pytest.fixture(autouse=True)
def reset_pubmed_lock():
    """Reset the module-level lock singleton between tests."""
    pubmed_mod._PUBMED_LOCK = None
    yield
    pubmed_mod._PUBMED_LOCK = None
