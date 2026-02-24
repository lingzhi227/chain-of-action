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
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    recommendation: list[str] = field(default_factory=list)  # What we suggested
    followed_recommendation: bool = True                     # Did LLM follow?
    cost: TokenStats | None = None


@dataclass
class ExecutionContext:
    """Tracks the full execution state of a chain-of-action run."""
    task: str
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

    def adherence_rate(self) -> float:
        """Percentage of steps that followed the recommendation."""
        if not self.steps:
            return 0.0
        # Skip first step (no prior recommendation)
        steps_with_rec = [s for s in self.steps[1:] if s.recommendation]
        if not steps_with_rec:
            return 1.0
        followed = sum(1 for s in steps_with_rec if s.followed_recommendation)
        return followed / len(steps_with_rec)
