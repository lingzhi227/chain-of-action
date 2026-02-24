"""Integration tests — requires real Claude CLI. Skip when inside Claude Code."""
from __future__ import annotations

import math
import os
import statistics

import pytest

from src.core.action_type import default_salary_catalog, ActionCatalog, ActionType
from src.core.engine import Engine
from src.llm.claude import ClaudeProvider

# Skip all tests in this module when running inside Claude Code
pytestmark = pytest.mark.skipif(
    "CLAUDECODE" in os.environ,
    reason="Skipping integration tests inside Claude Code",
)


def _make_salary_engine() -> Engine:
    """Build an engine with the salary analysis catalog and tools."""
    catalog = default_salary_catalog()
    engine = Engine(catalog)

    def calc(expression: str) -> str:
        allowed = set("0123456789+-*/.() ,")
        if not all(c in allowed for c in expression):
            return f"Error: invalid characters in expression"
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    def compound(base: float, rate: float, years: int) -> str:
        result = base * ((1 + rate) ** years)
        return f"{result:.2f}"

    def stats(values: list[float]) -> str:
        if not values:
            return "Error: empty list"
        return (
            f"mean={statistics.mean(values):.2f}, "
            f"median={statistics.median(values):.2f}, "
            f"stdev={statistics.stdev(values):.2f}" if len(values) > 1
            else f"mean={values[0]:.2f}, median={values[0]:.2f}, stdev=0.00"
        )

    engine.register_tool("calc", calc, "Evaluate arithmetic expression: calc(expression='2+3')")
    engine.register_tool("compound", compound, "Compound interest: compound(base=100000, rate=0.05, years=4)")
    engine.register_tool("stats", stats, "Statistics: stats(values=[1.0, 2.0, 3.0])")
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
    """Full salary analysis with tools — multiple compute/verify cycles expected."""
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

    # Should have used tools
    tool_steps = [s for s in ctx.steps if s.tool_name]
    assert len(tool_steps) >= 2, f"Expected tool use, got {len(tool_steps)} tool steps"

    # Should reach done
    assert ctx.steps[-1].action_type == "done" or any(
        s.action_type == "done" for s in ctx.steps
    )

    # Action type counts should show variety
    counts = ctx.action_type_counts()
    assert len(counts) >= 2, f"Expected multiple action types, got: {counts}"


@pytest.mark.integration
async def test_adherence_tracking():
    """Verify that recommendation/followed fields are populated."""
    engine = _make_salary_engine()
    llm = ClaudeProvider(model="haiku")

    task = (
        "Compute compound interest: $100,000 at 5% for 3 years. "
        "Then verify by computing year-by-year."
    )

    ctx = await engine.run(task, llm, max_turns=10)

    # After first step, all steps should have recommendations populated
    for step in ctx.steps[1:]:
        # recommendation might be empty if previous type has no suggestions
        assert isinstance(step.recommendation, list)
        assert isinstance(step.followed_recommendation, bool)

    # adherence_rate should be computable
    rate = ctx.adherence_rate()
    assert 0.0 <= rate <= 1.0

    # Cost stats should be populated
    assert len(ctx.cost_stats) > 0
