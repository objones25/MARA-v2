"""Tests for mara/agents/registry.py — @agent decorator and get_agents."""

import pytest

import mara.agents.registry as reg_mod
from mara.agents.registry import AgentRegistration, agent, get_agents, get_registry_summary


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
        assert reg_mod._REGISTRY["test_agent"].cls is TestAgent

    def test_registration_is_agent_registration_instance(self):
        @agent("typed_agent")
        class TypedAgent:
            pass

        assert isinstance(reg_mod._REGISTRY["typed_agent"], AgentRegistration)

    def test_decorator_returns_class_unchanged(self):
        @agent("identity_agent")
        class IdentityAgent:
            pass

        assert IdentityAgent.__name__ == "IdentityAgent"

    def test_description_stored(self):
        @agent("described", description="Fetches papers from somewhere.")
        class Described:
            pass

        assert reg_mod._REGISTRY["described"].description == "Fetches papers from somewhere."

    def test_description_defaults_to_empty(self):
        @agent("nodesc")
        class NoDesc:
            pass

        assert reg_mod._REGISTRY["nodesc"].description == ""

    def test_capabilities_stored(self):
        @agent("capable", capabilities=["fast", "accurate"])
        class Capable:
            pass

        assert reg_mod._REGISTRY["capable"].capabilities == ["fast", "accurate"]

    def test_capabilities_defaults_to_empty_list(self):
        @agent("nocaps")
        class NoCaps:
            pass

        assert reg_mod._REGISTRY["nocaps"].capabilities == []

    def test_limitations_stored(self):
        @agent("limited", limitations=["rate-limited", "no full text"])
        class Limited:
            pass

        assert reg_mod._REGISTRY["limited"].limitations == ["rate-limited", "no full text"]

    def test_limitations_defaults_to_empty_list(self):
        @agent("nolimits")
        class NoLimits:
            pass

        assert reg_mod._REGISTRY["nolimits"].limitations == []

    def test_example_queries_stored(self):
        @agent("exemplary", example_queries=["what is X?", "how does Y work?"])
        class Exemplary:
            pass

        assert reg_mod._REGISTRY["exemplary"].example_queries == ["what is X?", "how does Y work?"]

    def test_example_queries_defaults_to_empty_list(self):
        @agent("noexamples")
        class NoExamples:
            pass

        assert reg_mod._REGISTRY["noexamples"].example_queries == []

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


# ---------------------------------------------------------------------------
# get_registry_summary
# ---------------------------------------------------------------------------


class TestGetRegistrySummary:
    def test_empty_registry_returns_placeholder(self):
        assert get_registry_summary() == "(no agents registered)"

    def test_single_agent_includes_name_and_description(self):
        @agent("summary_agent", description="Does something useful.")
        class SummaryAgent:
            pass

        summary = get_registry_summary()
        assert "[summary_agent]" in summary
        assert "Does something useful." in summary

    def test_capabilities_appear_in_summary(self):
        @agent("cap_agent", capabilities=["fast retrieval", "full text"])
        class CapAgent:
            pass

        summary = get_registry_summary()
        assert "fast retrieval" in summary
        assert "full text" in summary

    def test_limitations_appear_in_summary(self):
        @agent("lim_agent", limitations=["rate-limited"])
        class LimAgent:
            pass

        summary = get_registry_summary()
        assert "rate-limited" in summary

    def test_example_queries_appear_in_summary(self):
        @agent("ex_agent", example_queries=["what is X?"])
        class ExAgent:
            pass

        summary = get_registry_summary()
        assert "what is X?" in summary

    def test_multiple_agents_all_present(self):
        @agent("first_agent", description="First.")
        class First:
            pass

        @agent("second_agent", description="Second.")
        class Second:
            pass

        summary = get_registry_summary()
        assert "[first_agent]" in summary
        assert "[second_agent]" in summary

    def test_optional_fields_omitted_when_empty(self):
        @agent("bare_agent", description="Bare.")
        class BareAgent:
            pass

        summary = get_registry_summary()
        assert "Capabilities:" not in summary
        assert "Limitations:" not in summary
        assert "Example queries:" not in summary
