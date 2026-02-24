from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.base import TokenUsage


@dataclass
class TokenStats:
    """Accumulated stats for a single action type."""
    cost_usd: float = 0.0
    duration_ms: int = 0
    calls: int = 0


class TokenTracker:
    """Tracks per-action-type usage across a run."""

    def __init__(self) -> None:
        self.stats: dict[str, TokenStats] = {}

    def record(self, action_type: str, usage: TokenUsage) -> None:
        """Record usage for an action type."""
        if action_type not in self.stats:
            self.stats[action_type] = TokenStats()
        s = self.stats[action_type]
        s.cost_usd += usage.cost_usd
        s.duration_ms += usage.duration_ms
        s.calls += 1

    def total_cost(self) -> float:
        """Total cost across all action types."""
        return sum(s.cost_usd for s in self.stats.values())

    def total_calls(self) -> int:
        """Total LLM calls across all action types."""
        return sum(s.calls for s in self.stats.values())

    def report(self) -> dict[str, Any]:
        """Generate a summary report."""
        per_type = {}
        for name, s in self.stats.items():
            per_type[name] = {
                "cost_usd": s.cost_usd,
                "duration_ms": s.duration_ms,
                "calls": s.calls,
            }
        return {
            "per_action_type": per_type,
            "total_cost_usd": self.total_cost(),
            "total_calls": self.total_calls(),
        }
