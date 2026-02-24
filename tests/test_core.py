"""Unit tests for chain-of-action core modules. No LLM calls."""
from __future__ import annotations

import pytest

from src.core.action_type import ActionCatalog, ActionType, default_salary_catalog
from src.core.advisor import ActionAdvisor
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
    def _make_advisor(self):
        catalog = default_salary_catalog()
        return ActionAdvisor(catalog)

    def test_build_system_prompt_contains_key_sections(self):
        advisor = self._make_advisor()
        prompt = advisor.build_system_prompt()
        assert "Action Types" in prompt
        assert "action_type" in prompt
        assert "ONE step per turn" in prompt
        assert "MCP tools" in prompt

    def test_build_plan_prompt(self):
        advisor = self._make_advisor()
        prompt = advisor.build_plan_prompt()
        assert "Action Types" in prompt
        assert "execution plan" in prompt
        assert "must end with" in prompt.lower()

    def test_build_plan_schema(self):
        advisor = self._make_advisor()
        schema = advisor.build_plan_schema()
        assert schema["type"] == "object"
        assert "thinking" in schema["properties"]
        assert "plan" in schema["properties"]
        assert set(schema["required"]) == {"thinking", "plan"}

    def test_build_plan_schema_plan_must_be_array_of_objects(self):
        advisor = self._make_advisor()
        schema = advisor.build_plan_schema()
        plan_prop = schema["properties"]["plan"]
        assert plan_prop["type"] == "array"
        assert plan_prop["items"]["type"] == "object"
        item_props = plan_prop["items"]["properties"]
        assert "action_type" in item_props
        assert "description" in item_props
        assert set(plan_prop["items"]["required"]) == {"action_type", "description"}

    def test_build_recommendation_with_suggestions(self):
        advisor = self._make_advisor()
        rec = advisor.build_recommendation("compute", [])
        assert "compute" in rec
        assert "verify" in rec

    def test_build_recommendation_no_suggestions(self):
        advisor = self._make_advisor()
        rec = advisor.build_recommendation("done", [])
        assert "done" in rec

    def test_build_recommendation_with_plan(self):
        advisor = self._make_advisor()
        plan = [
            {"action_type": "analyze", "description": "Understand the problem"},
            {"action_type": "compute", "description": "Calculate Alice's salary"},
            {"action_type": "verify", "description": "Cross-check results"},
            {"action_type": "done", "description": "Present final answer"},
        ]
        rec = advisor.build_recommendation("analyze", [], plan=plan, plan_cursor=1)
        assert "Step 2/4" in rec
        assert "compute" in rec
        assert "Calculate Alice" in rec
        assert "Execute ONLY" in rec
        assert "Do NOT" in rec

    def test_build_recommendation_with_plan_final_step(self):
        advisor = self._make_advisor()
        plan = [
            {"action_type": "compute", "description": "Calculate"},
            {"action_type": "done", "description": "Present final answer"},
        ]
        rec = advisor.build_recommendation("compute", [], plan=plan, plan_cursor=1)
        assert "Step 2/2" in rec
        assert "done" in rec
        # Final step should NOT have "Do NOT set is_done"
        assert "Do NOT" not in rec

    def test_build_recommendation_without_plan(self):
        advisor = self._make_advisor()
        rec = advisor.build_recommendation("compute", [], plan=None, plan_cursor=0)
        assert "last action was" in rec
        assert "verify" in rec

    def test_build_recommendation_repeated_actions(self):
        advisor = self._make_advisor()
        history = [
            ActionStep(action_type="compute"),
            ActionStep(action_type="compute"),
            ActionStep(action_type="compute"),
        ]
        rec = advisor.build_recommendation("compute", history)
        assert "consecutive" in rec

    def test_build_response_schema_no_tool_fields(self):
        advisor = self._make_advisor()
        schema = advisor.build_response_schema()
        assert schema["type"] == "object"
        props = schema["properties"]
        assert "action_type" in props
        assert "thinking" in props
        assert "response" in props
        assert "is_done" in props
        # MCP tools — no tool_name/tool_args in schema
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

    def test_plan_adherence_rate_full_match(self):
        ctx = ExecutionContext(task="test")
        ctx.plan = [
            {"action_type": "analyze", "description": "understand"},
            {"action_type": "compute", "description": "calculate"},
            {"action_type": "done", "description": "finish"},
        ]
        ctx.steps = [
            ActionStep(action_type="analyze"),
            ActionStep(action_type="compute"),
            ActionStep(action_type="done"),
        ]
        assert ctx.plan_adherence_rate() == 1.0

    def test_plan_adherence_rate_partial(self):
        ctx = ExecutionContext(task="test")
        ctx.plan = [
            {"action_type": "analyze", "description": "understand"},
            {"action_type": "compute", "description": "calculate"},
            {"action_type": "verify", "description": "check"},
            {"action_type": "done", "description": "finish"},
        ]
        ctx.steps = [
            ActionStep(action_type="analyze"),
            ActionStep(action_type="compute"),
            ActionStep(action_type="compute"),  # deviated from "verify"
            ActionStep(action_type="done"),
        ]
        assert ctx.plan_adherence_rate() == 0.75

    def test_plan_adherence_rate_no_plan(self):
        ctx = ExecutionContext(task="test")
        ctx.steps = [ActionStep(action_type="analyze")]
        assert ctx.plan_adherence_rate() == 0.0

    def test_plan_cursor_tracking(self):
        ctx = ExecutionContext(task="test")
        ctx.plan = [
            {"action_type": "analyze", "description": "understand"},
            {"action_type": "compute", "description": "calculate"},
            {"action_type": "done", "description": "finish"},
        ]
        assert ctx.plan_cursor == 0
        ctx.plan_cursor = 2
        assert ctx.plan_cursor == 2

    def test_action_step_planned_type(self):
        step = ActionStep(action_type="compute", planned_type="compute")
        assert step.planned_type == "compute"
        step2 = ActionStep(action_type="compute")
        assert step2.planned_type is None

    def test_action_step_tool_calls(self):
        step = ActionStep(
            action_type="compute",
            tool_calls=[
                {"name": "compound", "args": {"base": 95000, "rate": 0.08, "years": 4}, "result": "129145.55"},
                {"name": "compound", "args": {"base": 110000, "rate": 0.05, "years": 4}, "result": "133762.18"},
            ],
        )
        assert len(step.tool_calls) == 2
        assert step.tool_name == "compound"
        assert step.tool_args == {"base": 95000, "rate": 0.08, "years": 4}
        assert step.tool_result == "129145.55"

    def test_action_step_no_tool_calls(self):
        step = ActionStep(action_type="analyze")
        assert step.tool_calls == []
        assert step.tool_name is None
        assert step.tool_args is None
        assert step.tool_result is None


# ── Engine ──────────────────────────────────────────────────────────────


class TestEngine:
    def test_register_mcp_server(self):
        engine = Engine(ActionCatalog())
        engine.register_mcp_server("test-tools", ["python", "server.py"])
        assert engine._mcp_server_name == "test-tools"
        assert engine._mcp_command == ["python", "server.py"]


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


# ── ClaudeProvider stream-json parsing ──────────────────────────────────


class TestStreamJsonParsing:
    def test_parse_stream_events(self):
        from src.llm.claude import _parse_stream_events
        output = '{"type":"system"}\n{"type":"result","result":"hello"}\n'
        events = _parse_stream_events(output)
        assert len(events) == 2
        assert events[0]["type"] == "system"
        assert events[1]["type"] == "result"

    def test_extract_tool_calls(self):
        from src.llm.claude import _extract_tool_calls
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "mcp__server__calc", "input": {"expression": "2+3"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "5"},
            ]}},
        ]
        calls = _extract_tool_calls(events)
        assert len(calls) == 1
        assert calls[0]["name"] == "calc"
        assert calls[0]["full_name"] == "mcp__server__calc"
        assert calls[0]["args"] == {"expression": "2+3"}
        assert calls[0]["result"] == "5"

    def test_extract_tool_calls_empty(self):
        from src.llm.claude import _extract_tool_calls
        events = [{"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}}]
        assert _extract_tool_calls(events) == []

    def test_extract_result_text(self):
        from src.llm.claude import _extract_result_text
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "ignored"}]}},
            {"type": "result", "result": '{"action_type":"compute"}'},
        ]
        assert _extract_result_text(events) == '{"action_type":"compute"}'

    def test_extract_result_text_fallback(self):
        from src.llm.claude import _extract_result_text
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}},
        ]
        assert _extract_result_text(events) == "hello"

    def test_extract_usage(self):
        from src.llm.claude import _extract_usage
        events = [{"type": "result", "total_cost_usd": 0.05, "duration_ms": 1234}]
        cost, duration = _extract_usage(events)
        assert cost == 0.05
        assert duration == 1234
