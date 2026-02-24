from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionType:
    """A single action type in the catalog."""
    name: str                          # "analyze", "compute", "verify", etc.
    description: str                   # What this action type means
    suggested_next: list[str] = field(default_factory=list)  # Recommended downstream types
    tools: list[str] = field(default_factory=list)           # Commonly used tools (informational)


class ActionCatalog:
    """Registry of action types with transition recommendations."""

    def __init__(self) -> None:
        self.types: dict[str, ActionType] = {}

    def add(self, action_type: ActionType) -> None:
        """Register an action type."""
        self.types[action_type.name] = action_type

    def get(self, name: str) -> ActionType | None:
        """Get an action type by name."""
        return self.types.get(name)

    def get_suggestions(self, current: str) -> list[str]:
        """Get recommended next action types for a given current type."""
        at = self.types.get(current)
        if at is None:
            return []
        return list(at.suggested_next)

    def type_names(self) -> list[str]:
        """All registered type names."""
        return list(self.types.keys())

    def to_prompt_section(self) -> str:
        """Render all types as a prompt section for the LLM."""
        lines = ["## Action Types", ""]
        for at in self.types.values():
            tools_str = f" (common tools: {', '.join(at.tools)})" if at.tools else ""
            next_str = f" → suggested next: [{', '.join(at.suggested_next)}]" if at.suggested_next else " (terminal)"
            lines.append(f"- **{at.name}**: {at.description}{tools_str}{next_str}")
        return "\n".join(lines)


def default_salary_catalog() -> ActionCatalog:
    """Default action catalog for the salary analysis example."""
    catalog = ActionCatalog()
    catalog.add(ActionType(
        name="analyze",
        description="Understand the problem, identify what data and computations are needed",
        suggested_next=["plan", "compute"],
    ))
    catalog.add(ActionType(
        name="plan",
        description="Break the task into concrete computation steps",
        suggested_next=["compute"],
    ))
    catalog.add(ActionType(
        name="compute",
        description="Perform a calculation using tools",
        suggested_next=["verify", "compute"],
        tools=["calc", "compound", "stats"],
    ))
    catalog.add(ActionType(
        name="verify",
        description="Check a previous computation for correctness",
        suggested_next=["compute", "synthesize"],
        tools=["calc", "compound"],
    ))
    catalog.add(ActionType(
        name="synthesize",
        description="Combine results into a final answer",
        suggested_next=["done"],
    ))
    catalog.add(ActionType(
        name="done",
        description="Task is complete, present final results",
        suggested_next=[],
    ))
    return catalog
