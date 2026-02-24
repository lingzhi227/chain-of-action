"""Integration tests — requires real Claude CLI. Skip when inside Claude Code."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.action_type import default_salary_catalog, ActionCatalog, ActionType
from src.core.engine import Engine
from src.llm.claude import ClaudeProvider

# Skip all tests in this module when running inside Claude Code
pytestmark = pytest.mark.skipif(
    "CLAUDECODE" in os.environ,
    reason="Skipping integration tests inside Claude Code",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER = str(PROJECT_ROOT / "tools" / "mcp_server.py")


def _make_salary_engine() -> Engine:
    """Build an engine with the salary analysis catalog and MCP tools."""
    catalog = default_salary_catalog()
    engine = Engine(catalog)
    engine.register_mcp_server(
        "chain-tools",
        ["uv", "run", "--directory", str(PROJECT_ROOT), "python", MCP_SERVER],
    )
    return engine


@pytest.mark.integration
async def test_simple_qa():
    """Simple question that should: analyze -> synthesize -> done."""
    catalog = ActionCatalog()
    catalog.add(ActionType(name="analyze", description="Understand the question", suggested_next=["synthesize"]))
    catalog.add(ActionType(name="synthesize", description="Give the answer", suggested_next=["done"]))
    catalog.add(ActionType(name="done", description="Task complete", suggested_next=[]))

    engine = Engine(catalog)
    llm = ClaudeProvider(model="haiku")

    ctx = await engine.run("What is 2 + 2? Give just the number.", llm, max_turns=5)

    assert len(ctx.steps) >= 1
    assert ctx.steps[-1].action_type == "done" or any(
        s.response and "4" in s.response for s in ctx.steps
    )


@pytest.mark.integration
async def test_salary_analysis():
    """Full salary analysis with MCP tools — multiple compute/verify cycles expected."""
    engine = _make_salary_engine()
    llm = ClaudeProvider(model="haiku")

    task = (
        "Three engineers:\n"
        "- Alice: $95,000 base, 8% annual raise\n"
        "- Bob: $110,000 base, 5% annual raise\n"
        "- Charlie: $88,000 base, 10% annual raise\n\n"
        "Compute each engineer's salary after 4 years of compound raises. "
        "Then compute mean, median, and stdev of the final salaries."
    )

    ctx = await engine.run(task, llm, max_turns=15)

    # Should have used MCP tools
    tool_steps = [s for s in ctx.steps if s.tool_calls]
    assert len(tool_steps) >= 1, f"Expected MCP tool use, got {len(tool_steps)} tool steps"

    # Should reach done
    assert ctx.steps[-1].action_type == "done" or any(
        s.action_type == "done" for s in ctx.steps
    )

    # Action type counts should show variety
    counts = ctx.action_type_counts()
    assert len(counts) >= 2, f"Expected multiple action types, got: {counts}"

    # Plan should be generated and end with "done"
    assert ctx.plan, "Expected a non-empty plan"
    assert ctx.plan[-1].get("action_type") == "done", f"Plan should end with 'done', got: {ctx.plan}"

    # Plan adherence rate should be computable
    plan_rate = ctx.plan_adherence_rate()
    assert 0.0 <= plan_rate <= 1.0

    # Each step should have planned_type populated (if plan covers it)
    for i, step in enumerate(ctx.steps):
        if i < len(ctx.plan):
            assert step.planned_type is not None, f"Step {i} missing planned_type"


@pytest.mark.integration
async def test_cost_tracking():
    """Verify that cost stats are populated after a run."""
    engine = _make_salary_engine()
    llm = ClaudeProvider(model="haiku")

    task = (
        "Compute compound interest: $100,000 at 5% for 3 years. "
        "Then verify by computing year-by-year."
    )

    ctx = await engine.run(task, llm, max_turns=10)

    # Cost stats should be populated
    assert len(ctx.cost_stats) > 0
