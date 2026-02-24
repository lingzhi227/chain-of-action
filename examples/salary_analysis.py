"""Salary analysis example — same problem as action-chain for comparison.

Outputs TRACE.md with full execution trace, transition matrix,
plan adherence rate, and cost per action type.

Usage:
    uv run python examples/salary_analysis.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_type import default_salary_catalog
from src.core.context import ExecutionContext
from src.core.engine import Engine
from src.llm.claude import ClaudeProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MCP_SERVER = str(PROJECT_ROOT / "tools" / "mcp_server.py")


def build_engine() -> Engine:
    """Build engine with salary analysis catalog and MCP tools."""
    catalog = default_salary_catalog()
    engine = Engine(catalog)
    engine.register_mcp_server(
        "chain-tools",
        ["uv", "run", "--directory", str(PROJECT_ROOT), "python", MCP_SERVER],
    )
    return engine


def generate_trace(ctx: ExecutionContext) -> str:
    """Generate TRACE.md content from execution context."""
    lines = [
        "# Chain-of-Action Execution Trace",
        "",
        f"**Task**: {ctx.task}",
        f"**Total turns**: {ctx.turn_count}",
        f"**Total steps**: {len(ctx.steps)}",
        "",
    ]

    # Generated plan
    if ctx.plan:
        lines.append("## Generated Plan")
        lines.append("")
        for i, step_info in enumerate(ctx.plan, 1):
            at = step_info.get("action_type", "?")
            desc = step_info.get("description", "")
            lines.append(f"{i}. **{at}**: {desc}")
        lines.append("")

    # Step-by-step trace
    lines.append("## Step-by-Step Trace")
    lines.append("")
    for i, step in enumerate(ctx.steps, 1):
        planned_str = f" (planned: `{step.planned_type}`)" if step.planned_type else ""
        lines.append(f"### Step {i}: [{step.action_type}]{planned_str}")
        lines.append("")
        lines.append(f"**Thinking**: {step.thinking}")
        lines.append("")
        lines.append(f"**Response**: {step.response[:200]}{'...' if len(step.response) > 200 else ''}")
        lines.append("")
        if step.tool_calls:
            lines.append(f"**Tool Calls** ({len(step.tool_calls)}):")
            lines.append("")
            for tc in step.tool_calls:
                lines.append(f"- `{tc['name']}({tc.get('args', {})})` → `{tc.get('result', 'N/A')}`")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Action type counts
    lines.append("## Action Type Distribution")
    lines.append("")
    lines.append("| Action Type | Count |")
    lines.append("|---|---|")
    for name, count in sorted(ctx.action_type_counts().items()):
        lines.append(f"| {name} | {count} |")
    lines.append("")

    # Transition matrix
    lines.append("## Transition Matrix")
    lines.append("")
    matrix = ctx.transition_matrix()
    if matrix:
        all_types = sorted(set(
            list(matrix.keys()) +
            [t for targets in matrix.values() for t in targets]
        ))
        header = "| From \\ To | " + " | ".join(all_types) + " |"
        sep = "|---|" + "|".join(["---"] * len(all_types)) + "|"
        lines.append(header)
        lines.append(sep)
        for src in all_types:
            row = f"| {src} |"
            for dst in all_types:
                count = matrix.get(src, {}).get(dst, 0)
                row += f" {count or '-'} |"
            lines.append(row)
    else:
        lines.append("No transitions recorded.")
    lines.append("")

    # Plan adherence rate
    if ctx.plan:
        lines.append("## Plan Adherence Rate")
        lines.append("")
        plan_rate = ctx.plan_adherence_rate()
        lines.append(f"**{plan_rate:.0%}** of steps matched the plan's expected action type.")
        lines.append("")

    # Cost per action type
    lines.append("## Cost per Action Type")
    lines.append("")
    lines.append("| Action Type | Calls | Cost (USD) | Duration (ms) |")
    lines.append("|---|---|---|---|")
    total_cost = 0.0
    total_duration = 0
    for name, stats in sorted(ctx.cost_stats.items()):
        lines.append(f"| {name} | {stats.calls} | ${stats.cost_usd:.4f} | {stats.duration_ms} |")
        total_cost += stats.cost_usd
        total_duration += stats.duration_ms
    lines.append(f"| **Total** | **{len(ctx.steps)}** | **${total_cost:.4f}** | **{total_duration}** |")
    lines.append("")

    # Comparison note
    lines.append("## Comparison vs Hard FSM (action-chain)")
    lines.append("")
    lines.append("| Aspect | action-chain (hard FSM) | chain-of-action (soft guidance) |")
    lines.append("|---|---|---|")
    lines.append("| State control | Enum-enforced transitions | LLM self-classifies freely |")
    lines.append("| Tool access | Whitelisted per state | MCP tools always available |")
    lines.append("| Flexibility | Rigid: must follow graph | Flexible: recommendations only |")
    lines.append(f"| Steps taken | (run action-chain for comparison) | {len(ctx.steps)} |")
    lines.append(f"| Total cost | (run action-chain for comparison) | ${total_cost:.4f} |")
    if ctx.plan:
        lines.append(f"| Plan adherence | N/A | {ctx.plan_adherence_rate():.0%} |")
    lines.append("")

    return "\n".join(lines)


async def main():
    engine = build_engine()
    llm = ClaudeProvider(model="haiku")

    task = (
        "Three software engineers have the following compensation:\n"
        "- Alice: $95,000 base salary, 8% annual raise\n"
        "- Bob: $110,000 base salary, 5% annual raise\n"
        "- Charlie: $88,000 base salary, 10% annual raise\n\n"
        "Compute each engineer's salary after 4 years of compound raises.\n"
        "Then compute the mean, median, and standard deviation of the three final salaries.\n"
        "Show all work."
    )

    print(f"Running chain-of-action salary analysis...")
    print(f"Task: {task[:80]}...")
    print()

    ctx = await engine.run(task, llm, max_turns=15)

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    if ctx.plan:
        plan_summary = " → ".join(s.get("action_type", "?") for s in ctx.plan)
        print(f"Plan: {plan_summary}")
    print(f"Total steps: {len(ctx.steps)}")
    print(f"Action types: {ctx.action_type_counts()}")
    if ctx.plan:
        print(f"Plan adherence: {ctx.plan_adherence_rate():.0%}")
    print()

    for i, step in enumerate(ctx.steps, 1):
        planned_str = f" [planned: {step.planned_type}]" if step.planned_type else ""
        tool_str = ""
        if step.tool_calls:
            names = [tc["name"] for tc in step.tool_calls]
            tool_str = f" → tools: {', '.join(names)}"
        print(f"  Step {i}: [{step.action_type}]{planned_str}{tool_str}")

    # Write TRACE.md
    trace_path = PROJECT_ROOT / "TRACE.md"
    trace_content = generate_trace(ctx)
    trace_path.write_text(trace_content)
    print(f"\nTrace written to: {trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
