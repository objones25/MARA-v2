from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from mara.agents.registry import AgentConfig
    from mara.agents.cache import SearchCache
    from mara.agents.filtering import ChunkFilter


class ResearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required API keys
    brave_api_key: str
    hf_token: str
    firecrawl_api_key: str
    core_api_key: str
    s2_api_key: str
    ncbi_api_key: str

    # LLM
    default_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_overrides: dict[str, str] = Field(default_factory=dict)
    llm_provider: str = "featherless-ai"
    llm_temperature: float = 0.7
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    llm_min_p: float = 0.0
    llm_max_tokens: int = 16384

    # Hash
    hash_algorithm: Literal["sha256"] = "sha256"

    # Web agent
    web_max_scrape_urls: int = 10
    web_llm_url_ranking: bool = True
    web_timeout_seconds: float = 30.0
    scrape_timeout_seconds: float = 60.0
    brave_freshness: str = ""

    # Retry / back-off
    max_retries: int = 3
    retry_backoff_base: float = 2.0

    # Per-agent config overrides (api_key, max_results, rate_limit_rps).
    # Keyed by agent name (e.g. "s2", "pubmed", "core").
    # API keys for s2/core/pubmed are wired in automatically by _wire_api_keys.
    # Runtime type is dict[str, AgentConfig]. Declared Any to avoid a circular import:
    #   config → agents.registry → agents (via __init__) → agents.base → config
    agent_config_overrides: Any = Field(default_factory=dict)

    # Logging
    log_level: str = "INFO"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Runtime type is ChunkFilter (see TYPE_CHECKING import above).
    # Declared Any to avoid a module-level circular import:
    #   config → agents.filtering → agents (via __init__) → agents.base → config
    # The default is set lazily in _set_defaults.
    chunk_filter: Any = Field(default=None)
    # Runtime type is SearchCache (see TYPE_CHECKING import above).
    # Same lazy-default pattern as chunk_filter.
    search_cache: Any = Field(default=None)

    @model_validator(mode="after")
    def _set_defaults(self) -> "ResearchConfig":
        if self.chunk_filter is None:
            from mara.agents.filtering import CapFilter

            self.chunk_filter = CapFilter()
        if self.search_cache is None:
            from mara.agents.cache import NoOpCache

            self.search_cache = NoOpCache()
        return self

    @model_validator(mode="after")
    def _wire_api_keys(self) -> "ResearchConfig":
        """Push flat API keys into agent_config_overrides, preserving other fields."""
        from mara.agents.registry import AgentConfig, _REGISTRY

        overrides = dict(self.agent_config_overrides)

        for agent_name, api_key in [
            ("s2", self.s2_api_key),
            ("core", self.core_api_key),
            ("pubmed", self.ncbi_api_key),
        ]:
            base = overrides.get(agent_name)
            if base is None:
                reg = _REGISTRY.get(agent_name)
                base = reg.config if reg is not None else AgentConfig()
            overrides[agent_name] = AgentConfig(
                api_key=api_key,
                max_results=base.max_results,
                rate_limit_rps=base.rate_limit_rps,
            )

        self.agent_config_overrides = overrides
        return self
