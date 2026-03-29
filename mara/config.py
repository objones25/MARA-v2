from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
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
    web_max_results: int = 20
    web_max_scrape_urls: int = 10
    web_llm_url_ranking: bool = True
    web_timeout_seconds: float = 30.0
    scrape_timeout_seconds: float = 60.0
    brave_freshness: str = ""

    # Semantic Scholar
    s2_max_rps: float = 1.0

    # PubMed
    pubmed_max_results: int = 20
    pubmed_rate_limit_per_second: float = 3.0

    # CORE
    core_max_results: int = 20

    # ArXiv
    arxiv_max_results: int = 20

    # Logging
    log_level: str = "INFO"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Runtime type is ChunkFilter (see TYPE_CHECKING import above).
    # Declared Any to avoid a module-level circular import:
    #   config → agents.filtering → agents (via __init__) → agents.base → config
    # The default is set lazily in _default_chunk_filter.
    chunk_filter: Any = Field(default=None)

    @model_validator(mode="after")
    def _default_chunk_filter(self) -> "ResearchConfig":
        if self.chunk_filter is None:
            from mara.agents.filtering import CapFilter

            self.chunk_filter = CapFilter()
        return self
