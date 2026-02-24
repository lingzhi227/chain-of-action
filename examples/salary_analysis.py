"""Salary analysis example — same problem as action-chain for comparison.

Outputs TRACE.md with full execution trace, transition matrix,
adherence rate, and cost per action type.

Usage:
    uv run python examples/salary_analysis.py
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.action_type import default_salary_catalog
from src.core.context import ExecutionContext
from src.core.engine import Engine
from src.llm.claude import ClaudeProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_engine() -> Engine:
    """Build engine with salary analysis catalog and tools."""
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
        if len(values) == 1:
            return f"mean={values[0]:.2f}, median={values[0]:.2f}, stdev=0.00"
        return (
            f"mean={statistics.mean(values):.2f}, "
            f"median={statistics.median(values):.2f}, "
            f"stdev={statistics.stdev(values):.2f}"
        )

    engine.register_tool("calc", calc, "Evaluate arithmetic: calc(expression='2+3')")
    engine.register_tool("compound", compound, "Compound interest: compound(base=100000, rate=0.05, years=4)")
    engine.register_tool("stats", stats, "Statistics: stats(values=[1.0, 2.0, 3.0])")
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

    # Step-by-step trace
    lines.append("## Step-by-Step Trace")
    lines.append("")
    for i, step in enumerate(ctx.steps, 1):
        lines.append(f"### Step {i}: [{step.action_type}]")
        lines.append("")
        lines.append(f"**Thinking**: {step.thinking}")
        lines.append("")
        lines.append(f"**Response**: {step.response[:200]}{'...' if len(step.response) > 200 else ''}")
        lines.append("")
        if step.tool_name:
            lines.append(f"**Tool**: `{step.tool_name}({step.tool_args})`")
            lines.append(f"**Result**: `{step.tool_result}`")
            lines.append("")
        if step.recommendation:
            followed = "Yes" if step.followed_recommendation else "No"
            lines.append(f"**Recommendation**: {step.recommendation} | **Followed**: {followed}")
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

    # Adherence rate
    lines.append("## Adherence Rate")
    lines.append("")
    rate = ctx.adherence_rate()
    lines.append(f"**{rate:.0%}** of steps followed the recommended action type.")
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
    lines.append("| Tool access | Whitelisted per state | All tools always available |")
    lines.append("| Flexibility | Rigid: must follow graph | Flexible: recommendations only |")
    lines.append(f"| Steps taken | (run action-chain for comparison) | {len(ctx.steps)} |")
    lines.append(f"| Total cost | (run action-chain for comparison) | ${total_cost:.4f} |")
    lines.append(f"| Adherence rate | N/A (forced) | {rate:.0%} |")
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
    print(f"Total steps: {len(ctx.steps)}")
    print(f"Action types: {ctx.action_type_counts()}")
    print(f"Adherence rate: {ctx.adherence_rate():.0%}")
    print()

    for i, step in enumerate(ctx.steps, 1):
        rec_str = ""
        if step.recommendation:
            rec_str = f" (rec: {step.recommendation}, followed: {step.followed_recommendation})"
        tool_str = f" → {step.tool_name}({step.tool_args})" if step.tool_name else ""
        print(f"  Step {i}: [{step.action_type}]{tool_str}{rec_str}")

    # Write TRACE.md
    trace_path = Path(__file__).resolve().parent.parent / "TRACE.md"
    trace_content = generate_trace(ctx)
    trace_path.write_text(trace_content)
    print(f"\nTrace written to: {trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
