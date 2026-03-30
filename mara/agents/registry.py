"""Agent registry — maps agent type names to SpecialistAgent subclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mara.agents.base import SpecialistAgent
    from mara.config import ResearchConfig

_log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Per-agent tunable parameters.

    Stored as the default on ``AgentRegistration`` and merged with runtime
    overrides from ``ResearchConfig.agent_config_overrides`` at agent
    instantiation time.
    """

    api_key: str = field(default="", repr=False)
    max_results: int = 20
    rate_limit_rps: float = 0.0  # 0 = no rate limiting (use class-level default)
    max_concurrent: int = 0  # 0 = unlimited concurrent _search() calls
    retry_backoff_base: float = 0.0  # 0 = use ResearchConfig.retry_backoff_base
    max_retries: int = 0  # 0 = use ResearchConfig.max_retries
    max_sub_queries: int = 0  # 0 = unlimited; >0 caps how many sub-queries the planner may route here


@dataclass
class AgentRegistration:
    """Metadata for a registered specialist agent.

    All list fields default to empty so callers can omit any they don't need.
    """

    cls: type[SpecialistAgent]
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    example_queries: list[str] = field(default_factory=list)
    config: AgentConfig = field(default_factory=AgentConfig)


_REGISTRY: dict[str, AgentRegistration] = {}


def agent(
    name: str,
    description: str = "",
    capabilities: list[str] | None = None,
    limitations: list[str] | None = None,
    example_queries: list[str] | None = None,
    config: AgentConfig | None = None,
):
    """Class decorator that registers a SpecialistAgent subclass by name.

    Args:
        name: Unique agent identifier used throughout the pipeline.
        description: One-sentence summary of what the agent retrieves.
        capabilities: What this agent does well (bullet points for the planner).
        limitations: Known weaknesses or content types to avoid routing here.
        example_queries: Representative sub-queries ideal for this agent.
        config: Default ``AgentConfig`` for this agent (rate limits, max results).

    Raises:
        ValueError: If *name* is already registered.
    """

    def _decorator(cls: type[SpecialistAgent]) -> type[SpecialistAgent]:
        if name in _REGISTRY:
            raise ValueError(
                f"Agent name {name!r} is already registered by {_REGISTRY[name].cls!r}"
            )
        _REGISTRY[name] = AgentRegistration(
            cls=cls,
            description=description,
            capabilities=capabilities or [],
            limitations=limitations or [],
            example_queries=example_queries or [],
            config=config or AgentConfig(),
        )
        _log.debug("registered agent %r → %s", name, cls.__qualname__)
        return cls

    return _decorator


def get_agents(config: ResearchConfig) -> list[SpecialistAgent]:
    """Instantiate every registered agent with *config* and return the list."""
    overrides: dict[str, AgentConfig] = getattr(config, "agent_config_overrides", {})
    agents = [reg.cls(config, overrides.get(name, reg.config)) for name, reg in _REGISTRY.items()]
    _log.debug("instantiated %d agent(s): %s", len(agents), list(_REGISTRY.keys()))
    return agents


def get_registry_summary() -> str:
    """Return a formatted roster of registered agents for use in LLM prompts.

    Format per agent:
        [name] description
          Capabilities: ...
          Limitations: ...
          Example queries: ...

    Returns a placeholder string when no agents are registered.
    """
    if not _REGISTRY:
        return "(no agents registered)"

    sections: list[str] = []
    for name, reg in _REGISTRY.items():
        lines = [f"[{name}] {reg.description}"]
        if reg.capabilities:
            caps = "; ".join(reg.capabilities)
            lines.append(f"  Capabilities: {caps}")
        if reg.limitations:
            limits = "; ".join(reg.limitations)
            lines.append(f"  Limitations: {limits}")
        if reg.example_queries:
            examples = "; ".join(f'"{q}"' for q in reg.example_queries)
            lines.append(f"  Example queries: {examples}")
        if reg.config.max_sub_queries > 0:
            lines.append(f"  Max sub-queries: {reg.config.max_sub_queries}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
