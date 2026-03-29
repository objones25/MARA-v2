from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MARA_",
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
    web_llm_url_ranking: bool = True
    web_timeout_seconds: float = 30.0
    scrape_timeout_seconds: float = 60.0

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
