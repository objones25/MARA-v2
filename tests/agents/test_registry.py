"""Tests for mara/agents/registry.py — @agent decorator and get_agents."""

import pytest

import mara.agents.registry as reg_mod
from mara.agents.registry import agent, get_agents


# ---------------------------------------------------------------------------
# @agent decorator
# ---------------------------------------------------------------------------


class TestAgentDecorator:
    def test_registers_class_by_name(self):
        @agent("test_agent")
        class TestAgent:
            def __init__(self, config):
                self.config = config

        assert "test_agent" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["test_agent"] is TestAgent

    def test_decorator_returns_class_unchanged(self):
        @agent("identity_agent")
        class IdentityAgent:
            pass

        assert IdentityAgent.__name__ == "IdentityAgent"

    def test_collision_raises_value_error(self):
        @agent("duplicate")
        class First:
            pass

        with pytest.raises(ValueError, match="duplicate"):
            @agent("duplicate")
            class Second:
                pass

    def test_different_names_both_registered(self):
        @agent("alpha")
        class Alpha:
            pass

        @agent("beta")
        class Beta:
            pass

        assert "alpha" in reg_mod._REGISTRY
        assert "beta" in reg_mod._REGISTRY


# ---------------------------------------------------------------------------
# get_agents
# ---------------------------------------------------------------------------


class TestGetAgents:
    def test_empty_registry_returns_empty_list(self, config):
        assert get_agents(config) == []

    def test_instantiates_each_registered_class(self, config):
        instances = []

        @agent("recorder")
        class Recorder:
            def __init__(self, cfg):
                instances.append(cfg)

        result = get_agents(config)
        assert len(result) == 1
        assert len(instances) == 1
        assert instances[0] is config

    def test_returns_one_instance_per_registered_agent(self, config):
        @agent("one")
        class One:
            def __init__(self, cfg):
                pass

        @agent("two")
        class Two:
            def __init__(self, cfg):
                pass

        result = get_agents(config)
        assert len(result) == 2

    def test_instance_types_match_registered_classes(self, config):
        @agent("typed")
        class TypedAgent:
            def __init__(self, cfg):
                pass

        result = get_agents(config)
        assert isinstance(result[0], TypedAgent)
