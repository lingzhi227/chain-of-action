"""Unit tests for chain-of-action core modules. No LLM calls."""
from __future__ import annotations

import pytest

from src.core.action_type import ActionCatalog, ActionType, default_salary_catalog
from src.core.advisor import ActionAdvisor, ToolDef
from src.core.context import ActionStep, ExecutionContext
from src.core.engine import Engine
from src.monitoring.tracker import TokenStats, TokenTracker
from src.llm.base import TokenUsage


# ── ActionType / ActionCatalog ──────────────────────────────────────────


class TestActionType:
    def test_basic_creation(self):
        at = ActionType(name="compute", description="do math")
        assert at.name == "compute"
        assert at.description == "do math"
        assert at.suggested_next == []
        assert at.tools == []

    def test_with_suggestions_and_tools(self):
        at = ActionType(
            name="compute",
            description="do math",
            suggested_next=["verify", "compute"],
            tools=["calc", "compound"],
        )
        assert at.suggested_next == ["verify", "compute"]
        assert at.tools == ["calc", "compound"]


class TestActionCatalog:
    def test_add_and_get(self):
        catalog = ActionCatalog()
        at = ActionType(name="analyze", description="understand problem")
        catalog.add(at)
        assert catalog.get("analyze") is at
        assert catalog.get("missing") is None

    def test_type_names(self):
        catalog = ActionCatalog()
        catalog.add(ActionType(name="a", description=""))
        catalog.add(ActionType(name="b", description=""))
        assert catalog.type_names() == ["a", "b"]

    def test_get_suggestions(self):
        catalog = ActionCatalog()
        catalog.add(ActionType(name="compute", description="", suggested_next=["verify"]))
        assert catalog.get_suggestions("compute") == ["verify"]
        assert catalog.get_suggestions("nonexistent") == []

    def test_to_prompt_section(self):
        catalog = ActionCatalog()
        catalog.add(ActionType(
            name="compute",
            description="do math",
            suggested_next=["verify"],
            tools=["calc"],
        ))
        prompt = catalog.to_prompt_section()
        assert "## Action Types" in prompt
        assert "compute" in prompt
        assert "do math" in prompt
        assert "calc" in prompt
        assert "verify" in prompt

    def test_to_prompt_section_terminal(self):
        catalog = ActionCatalog()
        catalog.add(ActionType(name="done", description="finished"))
        prompt = catalog.to_prompt_section()
        assert "(terminal)" in prompt

    def test_default_salary_catalog(self):
        catalog = default_salary_catalog()
        names = catalog.type_names()
        assert "analyze" in names
        assert "plan" in names
        assert "compute" in names
        assert "verify" in names
        assert "synthesize" in names
        assert "done" in names
        # done is terminal
        assert catalog.get_suggestions("done") == []
        # compute suggests verify or compute
        assert "verify" in catalog.get_suggestions("compute")


# ── ActionAdvisor ───────────────────────────────────────────────────────


class TestActionAdvisor:
    def _make_advisor(self, with_tools=True):
        catalog = default_salary_catalog()
        tools = {}
        if with_tools:
            tools = {
                "calc": ToolDef(name="calc", description="Basic arithmetic"),
                "compound": ToolDef(name="compound", description="Compound interest"),
            }
        return ActionAdvisor(catalog, tools)

    def test_build_system_prompt_contains_key_sections(self):
        advisor = self._make_advisor()
        prompt = advisor.build_system_prompt()
        assert "Action Types" in prompt
        assert "Available Tools" in prompt
        assert "action_type" in prompt
        assert "self-classify" in prompt
        assert "suggestions only" in prompt

    def test_build_system_prompt_no_tools(self):
        advisor = self._make_advisor(with_tools=False)
        prompt = advisor.build_system_prompt()
        assert "No tools available" in prompt

    def test_build_recommendation_with_suggestions(self):
        advisor = self._make_advisor()
        rec = advisor.build_recommendation("compute", [])
        assert "compute" in rec
        assert "verify" in rec

    def test_build_recommendation_no_suggestions(self):
        advisor = self._make_advisor()
        rec = advisor.build_recommendation("done", [])
        assert "done" in rec

    def test_build_recommendation_repeated_actions(self):
        advisor = self._make_advisor()
        history = [
            ActionStep(action_type="compute"),
            ActionStep(action_type="compute"),
            ActionStep(action_type="compute"),
        ]
        rec = advisor.build_recommendation("compute", history)
        assert "consecutive" in rec

    def test_build_response_schema_with_tools(self):
        advisor = self._make_advisor()
        schema = advisor.build_response_schema()
        assert schema["type"] == "object"
        props = schema["properties"]
        assert "action_type" in props
        assert "thinking" in props
        assert "response" in props
        assert "tool_name" in props
        assert "tool_args" in props
        assert "is_done" in props
        # tool_name has enum with tool names + "none"
        assert "none" in props["tool_name"]["enum"]
        assert "calc" in props["tool_name"]["enum"]
        assert set(schema["required"]) == {
            "action_type", "thinking", "response", "tool_name", "tool_args", "is_done"
        }

    def test_build_response_schema_no_tools(self):
        advisor = self._make_advisor(with_tools=False)
        schema = advisor.build_response_schema()
        props = schema["properties"]
        assert "tool_name" not in props
        assert "tool_args" not in props
        assert set(schema["required"]) == {"action_type", "thinking", "response", "is_done"}


# ── ExecutionContext ────────────────────────────────────────────────────


class TestExecutionContext:
    def test_action_type_counts(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [
            ActionStep(action_type="compute"),
            ActionStep(action_type="compute"),
            ActionStep(action_type="verify"),
        ]
        counts = ctx.action_type_counts()
        assert counts == {"compute": 2, "verify": 1}

    def test_action_type_counts_empty(self):
        ctx = ExecutionContext(task="test")
        assert ctx.action_type_counts() == {}

    def test_transition_matrix(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [
            ActionStep(action_type="analyze"),
            ActionStep(action_type="compute"),
            ActionStep(action_type="verify"),
            ActionStep(action_type="compute"),
        ]
        matrix = ctx.transition_matrix()
        assert matrix == {
            "analyze": {"compute": 1},
            "compute": {"verify": 1},
            "verify": {"compute": 1},
        }

    def test_transition_matrix_empty(self):
        ctx = ExecutionContext(task="test")
        assert ctx.transition_matrix() == {}

    def test_transition_matrix_single_step(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [ActionStep(action_type="analyze")]
        assert ctx.transition_matrix() == {}

    def test_adherence_rate_all_followed(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [
            ActionStep(action_type="analyze", recommendation=[]),
            ActionStep(action_type="compute", recommendation=["compute", "plan"], followed_recommendation=True),
            ActionStep(action_type="verify", recommendation=["verify", "compute"], followed_recommendation=True),
        ]
        assert ctx.adherence_rate() == 1.0

    def test_adherence_rate_none_followed(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [
            ActionStep(action_type="analyze", recommendation=[]),
            ActionStep(action_type="random", recommendation=["compute"], followed_recommendation=False),
            ActionStep(action_type="random2", recommendation=["verify"], followed_recommendation=False),
        ]
        assert ctx.adherence_rate() == 0.0

    def test_adherence_rate_partial(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [
            ActionStep(action_type="analyze", recommendation=[]),
            ActionStep(action_type="compute", recommendation=["compute"], followed_recommendation=True),
            ActionStep(action_type="random", recommendation=["verify"], followed_recommendation=False),
        ]
        assert ctx.adherence_rate() == 0.5

    def test_adherence_rate_empty(self):
        ctx = ExecutionContext(task="test")
        assert ctx.adherence_rate() == 0.0

    def test_adherence_rate_single_step(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [ActionStep(action_type="analyze")]
        # No steps with recommendations to evaluate
        assert ctx.adherence_rate() == 1.0


# ── Engine (tool registration + execution, no LLM) ─────────────────────


class TestEngine:
    def test_register_tool(self):
        engine = Engine(ActionCatalog())
        engine.register_tool("calc", lambda expr: eval(expr), "Calculator")
        assert "calc" in engine._tools
        assert engine._tools["calc"].description == "Calculator"

    def test_exec_tool_success(self):
        engine = Engine(ActionCatalog())
        engine.register_tool("calc", lambda expr: eval(expr), "Calculator")
        result = engine._exec_tool("calc", {"expr": "2 + 3"})
        assert result == "5"

    def test_exec_tool_unknown(self):
        engine = Engine(ActionCatalog())
        result = engine._exec_tool("missing", {})
        assert "unknown tool" in result

    def test_exec_tool_error(self):
        def bad_tool(**kwargs):
            raise ValueError("boom")

        engine = Engine(ActionCatalog())
        engine.register_tool("bad", bad_tool, "Broken tool")
        result = engine._exec_tool("bad", {})
        assert "Error" in result
        assert "boom" in result

    def test_exec_tool_no_func(self):
        engine = Engine(ActionCatalog())
        engine._tools["empty"] = ToolDef(name="empty", description="no func")
        result = engine._exec_tool("empty", {})
        assert "no implementation" in result


# ── TokenTracker ────────────────────────────────────────────────────────


class TestTokenTracker:
    def test_record_single(self):
        tracker = TokenTracker()
        tracker.record("compute", TokenUsage(cost_usd=0.01, duration_ms=100))
        assert tracker.stats["compute"].calls == 1
        assert tracker.stats["compute"].cost_usd == 0.01
        assert tracker.stats["compute"].duration_ms == 100

    def test_record_multiple_same_type(self):
        tracker = TokenTracker()
        tracker.record("compute", TokenUsage(cost_usd=0.01, duration_ms=100))
        tracker.record("compute", TokenUsage(cost_usd=0.02, duration_ms=200))
        assert tracker.stats["compute"].calls == 2
        assert tracker.stats["compute"].cost_usd == pytest.approx(0.03)
        assert tracker.stats["compute"].duration_ms == 300

    def test_total_cost(self):
        tracker = TokenTracker()
        tracker.record("compute", TokenUsage(cost_usd=0.01))
        tracker.record("verify", TokenUsage(cost_usd=0.02))
        assert tracker.total_cost() == pytest.approx(0.03)

    def test_total_calls(self):
        tracker = TokenTracker()
        tracker.record("a", TokenUsage())
        tracker.record("b", TokenUsage())
        tracker.record("a", TokenUsage())
        assert tracker.total_calls() == 3

    def test_report(self):
        tracker = TokenTracker()
        tracker.record("compute", TokenUsage(cost_usd=0.05, duration_ms=500))
        report = tracker.report()
        assert "per_action_type" in report
        assert "compute" in report["per_action_type"]
        assert report["total_cost_usd"] == pytest.approx(0.05)
        assert report["total_calls"] == 1
