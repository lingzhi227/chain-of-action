from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.action_type import ActionCatalog
from src.core.context import ActionStep


@dataclass
class ToolDef:
    """Definition of a registered tool."""
    name: str
    description: str
    func: Any = None  # Callable, set at registration


class ActionAdvisor:
    """Builds system prompts and per-turn recommendations."""

    def __init__(self, catalog: ActionCatalog, tools: dict[str, ToolDef]) -> None:
        self._catalog = catalog
        self._tools = tools

    def build_system_prompt(self) -> str:
        """Build the initial system prompt describing action types, tools, and output format."""
        catalog_section = self._catalog.to_prompt_section()

        tool_lines = ["## Available Tools", ""]
        if self._tools:
            for t in self._tools.values():
                tool_lines.append(f"- **{t.name}**: {t.description}")
        else:
            tool_lines.append("No tools available.")
        tool_section = "\n".join(tool_lines)

        return (
            "You are solving a task step by step. At each step, you choose an action type "
            "that best describes what you're doing.\n\n"
            f"{catalog_section}\n\n"
            f"{tool_section}\n\n"
            "## How It Works\n"
            "1. You self-classify your action by setting `action_type` to any type from the catalog "
            "(or invent a new one if needed).\n"
            "2. All tools are always available — use whichever you need.\n"
            "3. Recommendations are suggestions only. You have full agency.\n"
            "4. Set `is_done: true` when the task is complete (or use action_type `done`).\n"
        )

    def build_recommendation(self, last_type: str, history: list[ActionStep]) -> str:
        """Build a per-turn recommendation based on last action type and history."""
        suggestions = self._catalog.get_suggestions(last_type)
        parts = []

        if suggestions:
            parts.append(
                f"Your last action was [{last_type}]. "
                f"Recommended next: [{', '.join(suggestions)}]."
            )
        else:
            parts.append(f"Your last action was [{last_type}].")

        # Count consecutive same-type actions
        if len(history) >= 3:
            recent_types = [s.action_type for s in history[-3:]]
            if len(set(recent_types)) == 1:
                parts.append(
                    f"You've done {len(recent_types)} consecutive [{recent_types[0]}] actions. "
                    f"Consider moving to a different action type."
                )

        return " ".join(parts)

    def build_response_schema(self) -> dict[str, Any]:
        """Build the JSON schema for structured LLM responses."""
        properties: dict[str, Any] = {
            "action_type": {
                "type": "string",
                "description": (
                    "The type of action you are performing. "
                    f"Known types: {', '.join(self._catalog.type_names())}. "
                    "You may also use a custom type if none fit."
                ),
            },
            "thinking": {
                "type": "string",
                "description": "Your internal reasoning about what to do.",
            },
            "response": {
                "type": "string",
                "description": "Your response text or the result of your work.",
            },
        }
        required = ["action_type", "thinking", "response"]

        if self._tools:
            tool_names = list(self._tools.keys()) + ["none"]
            properties["tool_name"] = {
                "type": "string",
                "enum": tool_names,
                "description": (
                    "Name of the tool to call, or 'none' if no tool is needed. "
                    f"Available: {', '.join(self._tools.keys())}"
                ),
            }
            properties["tool_args"] = {
                "type": "object",
                "description": "Arguments to pass to the selected tool. Empty if tool_name is 'none'.",
            }
            required.extend(["tool_name", "tool_args"])

        properties["is_done"] = {
            "type": "boolean",
            "description": "Set to true when the task is fully complete.",
        }
        required.append("is_done")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
