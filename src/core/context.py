from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.monitoring.tracker import TokenStats


@dataclass
class ActionStep:
    """A single step in the execution trace."""
    action_type: str                     # Self-declared by LLM
    thinking: str = ""
    response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{name, args, result}]
    planned_type: str | None = None                          # What the plan expected at this position
    cost: TokenStats | None = None

    @property
    def tool_name(self) -> str | None:
        """First tool called in this step (backward compat)."""
        return self.tool_calls[0]["name"] if self.tool_calls else None

    @property
    def tool_args(self) -> dict[str, Any] | None:
        """First tool's args (backward compat)."""
        return self.tool_calls[0].get("args") if self.tool_calls else None

    @property
    def tool_result(self) -> str | None:
        """First tool's result (backward compat)."""
        return self.tool_calls[0].get("result") if self.tool_calls else None


@dataclass
class ExecutionContext:
    """Tracks the full execution state of a chain-of-action run."""
    task: str
    plan: list[dict[str, str]] = field(default_factory=list)  # Generated plan: [{action_type, description}]
    plan_cursor: int = 0                                    # Current position in plan
    steps: list[ActionStep] = field(default_factory=list)
    turn_count: int = 0
    cost_stats: dict[str, TokenStats] = field(default_factory=dict)

    def action_type_counts(self) -> dict[str, int]:
        """Count how many times each action type was used."""
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.action_type] = counts.get(step.action_type, 0) + 1
        return counts

    def transition_matrix(self) -> dict[str, dict[str, int]]:
        """Build a transition matrix: type_a -> type_b: count."""
        matrix: dict[str, dict[str, int]] = {}
        for i in range(len(self.steps) - 1):
            src = self.steps[i].action_type
            dst = self.steps[i + 1].action_type
            if src not in matrix:
                matrix[src] = {}
            matrix[src][dst] = matrix[src].get(dst, 0) + 1
        return matrix

    def plan_adherence_rate(self) -> float:
        """Percentage of execution steps that matched the plan's expected action type."""
        if not self.plan or not self.steps:
            return 0.0
        matches = 0
        for i, step in enumerate(self.steps):
            if i < len(self.plan) and step.action_type == self.plan[i].get("action_type"):
                matches += 1
        return matches / len(self.steps)
