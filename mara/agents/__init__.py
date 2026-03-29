from mara.agents.base import SpecialistAgent
from mara.agents.registry import agent, get_agents
from mara.agents.types import AgentFindings, RawChunk, SubQuery, VerifiedChunk

# Agent modules must be imported here to trigger @agent(...) registration.
import mara.agents.arxiv  # noqa: E402, F401
import mara.agents.web.agent  # noqa: E402, F401

__all__ = [
    "SpecialistAgent",
    "agent",
    "get_agents",
    "AgentFindings",
    "RawChunk",
    "SubQuery",
    "VerifiedChunk",
]
