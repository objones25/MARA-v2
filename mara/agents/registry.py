"""Agent registry — maps agent type names to SpecialistAgent subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mara.agents.base import SpecialistAgent
    from mara.config import ResearchConfig

_REGISTRY: dict[str, type[SpecialistAgent]] = {}


def agent(name: str):
    """Class decorator that registers a SpecialistAgent subclass by name.

    Raises:
        ValueError: If *name* is already registered.
    """

    def _decorator(cls: type[SpecialistAgent]) -> type[SpecialistAgent]:
        if name in _REGISTRY:
            raise ValueError(
                f"Agent name {name!r} is already registered by {_REGISTRY[name]!r}"
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_agents(config: ResearchConfig) -> list[SpecialistAgent]:
    """Instantiate every registered agent with *config* and return the list."""
    return [cls(config) for cls in _REGISTRY.values()]
