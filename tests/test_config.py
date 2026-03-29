import pytest
from pydantic import ValidationError

from mara.config import ResearchConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_REQUIRED = {
    "brave_api_key": "brave-key",
    "hf_token": "hf-token",
    "firecrawl_api_key": "fc-key",
    "core_api_key": "core-key",
    "s2_api_key": "s2-key",
    "ncbi_api_key": "ncbi-key",
}


def _valid(**overrides) -> ResearchConfig:
    return ResearchConfig(**{**_ALL_REQUIRED, **overrides})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_mara_env(monkeypatch):
    """Remove any MARA_* env vars that could satisfy required fields."""
    for key in _ALL_REQUIRED:
        monkeypatch.delenv(f"MARA_{key.upper()}", raising=False)
    monkeypatch.delenv("MARA_MODEL_OVERRIDES", raising=False)


# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------


class TestRequiredKeys:
    @pytest.mark.parametrize("missing_key", _ALL_REQUIRED.keys())
    def test_missing_required_key_raises(self, missing_key):
        kwargs = {k: v for k, v in _ALL_REQUIRED.items() if k != missing_key}
        with pytest.raises(ValidationError):
            ResearchConfig(**kwargs)

    def test_all_required_keys_present_succeeds(self):
        config = _valid()
        assert config.brave_api_key == "brave-key"
        assert config.hf_token == "hf-token"
        assert config.firecrawl_api_key == "fc-key"
        assert config.core_api_key == "core-key"
        assert config.s2_api_key == "s2-key"
        assert config.ncbi_api_key == "ncbi-key"


# ---------------------------------------------------------------------------
# LLM defaults (Qwen3-30B-A3B-Instruct-2507 model card values)
# ---------------------------------------------------------------------------


class TestLLMDefaults:
    def test_default_model(self):
        assert _valid().default_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def test_model_overrides_empty_by_default(self):
        assert _valid().model_overrides == {}

    def test_model_overrides_can_be_set(self):
        config = _valid(model_overrides={"web": "smaller-model"})
        assert config.model_overrides == {"web": "smaller-model"}

    def test_llm_provider_default(self):
        assert _valid().llm_provider == "featherless-ai"

    def test_llm_temperature_default(self):
        assert _valid().llm_temperature == pytest.approx(0.7)

    def test_llm_top_p_default(self):
        assert _valid().llm_top_p == pytest.approx(0.8)

    def test_llm_top_k_default(self):
        assert _valid().llm_top_k == 20

    def test_llm_min_p_default(self):
        assert _valid().llm_min_p == pytest.approx(0.0)

    def test_llm_max_tokens_default(self):
        assert _valid().llm_max_tokens == 16384


# ---------------------------------------------------------------------------
# Hash algorithm
# ---------------------------------------------------------------------------


class TestHashAlgorithm:
    def test_default_is_sha256(self):
        assert _valid().hash_algorithm == "sha256"

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValidationError):
            _valid(hash_algorithm="sha512")


# ---------------------------------------------------------------------------
# Web agent
# ---------------------------------------------------------------------------


class TestWebAgent:
    def test_web_max_results_default(self):
        assert _valid().web_max_results == 20

    def test_web_llm_url_ranking_default_true(self):
        assert _valid().web_llm_url_ranking is True

    def test_web_timeout_seconds_default(self):
        assert _valid().web_timeout_seconds == pytest.approx(30.0)

    def test_scrape_timeout_seconds_default(self):
        assert _valid().scrape_timeout_seconds == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Per-agent parameters
# ---------------------------------------------------------------------------


class TestAgentParams:
    def test_s2_max_rps_default(self):
        assert _valid().s2_max_rps == pytest.approx(1.0)

    def test_pubmed_max_results_default(self):
        assert _valid().pubmed_max_results == 20

    def test_pubmed_rate_limit_default(self):
        assert _valid().pubmed_rate_limit_per_second == pytest.approx(3.0)

    def test_core_max_results_default(self):
        assert _valid().core_max_results == 20

    def test_arxiv_max_results_default(self):
        assert _valid().arxiv_max_results == 20


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogLevel:
    def test_log_level_default_info(self):
        assert _valid().log_level == "INFO"

    def test_log_level_can_be_overridden(self):
        assert _valid(log_level="DEBUG").log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Env-var loading
# ---------------------------------------------------------------------------


class TestEnvLoading:
    def test_config_loads_field_from_env_var(self, monkeypatch):
        monkeypatch.setenv("MARA_BRAVE_API_KEY", "env-brave-key")
        config = ResearchConfig(
            hf_token="hf",
            firecrawl_api_key="fc",
            core_api_key="core",
            s2_api_key="s2",
            ncbi_api_key="ncbi",
        )
        assert config.brave_api_key == "env-brave-key"

    def test_model_overrides_parsed_from_json_env(self, monkeypatch):
        monkeypatch.setenv("MARA_MODEL_OVERRIDES", '{"web": "small-model"}')
        config = _valid()
        assert config.model_overrides == {"web": "small-model"}

    def test_init_kwargs_take_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("MARA_BRAVE_API_KEY", "env-value")
        config = _valid(brave_api_key="kwarg-value")
        assert config.brave_api_key == "kwarg-value"
