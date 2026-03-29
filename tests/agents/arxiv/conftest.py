import pytest

import mara.agents.registry as reg_mod
from mara.agents.arxiv.agent import ArxivAgent
from mara.config import ResearchConfig

_REQUIRED = {
    "brave_api_key": "brave-key",
    "hf_token": "hf-token",
    "firecrawl_api_key": "fc-key",
    "core_api_key": "core-key",
    "s2_api_key": "s2-key",
    "ncbi_api_key": "ncbi-key",
}


@pytest.fixture()
def config() -> ResearchConfig:
    return ResearchConfig(**_REQUIRED)


@pytest.fixture(autouse=True)
def register_arxiv_agent():
    """Re-register ArxivAgent after isolate_registry (parent conftest) clears it."""
    reg_mod._REGISTRY["arxiv"] = ArxivAgent
    yield
