from __future__ import annotations

from typing import Any

from src.core.action_type import ActionCatalog
from src.core.context import ActionStep


class ActionAdvisor:
    """Builds system prompts and per-turn recommendations.

    Tools are handled natively by Claude CLI via MCP — the advisor
    only needs the action catalog for guidance prompts.
    """

    def __init__(self, catalog: ActionCatalog) -> None:
        self._catalog = catalog

    def build_system_prompt(self) -> str:
        """Build the initial system prompt describing action types and output format."""
        catalog_section = self._catalog.to_prompt_section()

        return (
            "You are solving a task step by step, following a plan. "
            "You must execute exactly ONE step per turn.\n\n"
            f"{catalog_section}\n\n"
            "## How It Works\n"
            "1. Each turn, you receive ONE step from the plan. Execute ONLY that step.\n"
            "2. Set `action_type` to match the plan's expected action type.\n"
            "3. Use MCP tools when you need to compute or verify.\n"
            "4. Do NOT skip ahead or combine multiple plan steps into one turn.\n"
            "5. Only set `is_done: true` when you reach the final `done` step of the plan.\n"
        )

    def build_plan_prompt(self) -> str:
        """Turn 0: ask LLM to generate an execution plan from the catalog."""
        catalog_section = self._catalog.to_prompt_section()

        return (
            "You are solving a task step by step. First, create an execution plan.\n\n"
            f"{catalog_section}\n\n"
            "## Your Task\n"
            "Analyze the task and create a plan: an ordered list of steps.\n"
            "Each step has an `action_type` from the catalog and a short `description` "
            "explaining what specifically to do in that step.\n"
            "You may repeat action types (e.g. multiple `compute` steps with different descriptions).\n"
            "The plan must end with a step whose action_type is `done`.\n"
        )

    def build_plan_schema(self) -> dict[str, Any]:
        """JSON schema for plan generation (turn 0)."""
        return {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string",
                    "description": "Your analysis of what steps are needed.",
                },
                "plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "description": (
                                    f"Action type. Available: {', '.join(self._catalog.type_names())}."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": "What specifically to do in this step.",
                            },
                        },
                        "required": ["action_type", "description"],
                    },
                    "description": (
                        "Ordered list of plan steps. "
                        "Must end with a step whose action_type is 'done'."
                    ),
                },
            },
            "required": ["thinking", "plan"],
        }

    def build_recommendation(
        self,
        last_type: str,
        history: list[ActionStep],
        plan: list[dict[str, str]] | None = None,
        plan_cursor: int = 0,
    ) -> str:
        """Build per-turn recommendation. If plan exists, show plan progress."""
        parts = []

        if plan:
            total = len(plan)
            current_idx = min(plan_cursor, total - 1)
            step_info = plan[current_idx]
            expected_type = step_info.get("action_type", "unknown")
            expected_desc = step_info.get("description", "")
            is_final = expected_type == "done"

            parts.append(f"Step {current_idx + 1}/{total}.")
            parts.append(f"Execute ONLY: [{expected_type}] — {expected_desc}." if expected_desc else f"Execute ONLY: [{expected_type}].")
            if not is_final:
                parts.append("Do NOT combine steps. Do NOT set is_done. One step only.")
        else:
            # Fallback to old behavior (no plan)
            suggestions = self._catalog.get_suggestions(last_type)
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
        """Build the JSON schema for structured LLM responses.

        Tools are NOT in the schema — Claude calls MCP tools natively.
        """
        return {
            "type": "object",
            "properties": {
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
                "is_done": {
                    "type": "boolean",
                    "description": "Set to true when the task is fully complete.",
                },
            },
            "required": ["action_type", "thinking", "response", "is_done"],
        }
