import pytest

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
def clear_mara_env(monkeypatch):
    for key in _REQUIRED:
        monkeypatch.delenv(f"MARA_{key.upper()}", raising=False)


@pytest.fixture(autouse=True)
def isolate_registry():
    """Save/clear/restore _REGISTRY around each test for isolation."""
    import mara.agents.registry as reg_mod

    saved = dict(reg_mod._REGISTRY)
    reg_mod._REGISTRY.clear()
    yield
    reg_mod._REGISTRY.clear()
    reg_mod._REGISTRY.update(saved)
