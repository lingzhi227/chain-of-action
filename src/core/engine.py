from __future__ import annotations

import logging
from typing import Any, Callable

from src.core.action_type import ActionCatalog
from src.core.advisor import ActionAdvisor, ToolDef
from src.core.context import ActionStep, ExecutionContext
from src.llm.base import LLMProvider
from src.monitoring.tracker import TokenTracker

logger = logging.getLogger(__name__)


class Engine:
    """Main execution loop for chain-of-action.

    Calls LLM -> classifies action -> executes tool -> recommends next -> repeat.
    All tools are always available. Recommendations are soft guidance.
    """

    def __init__(self, catalog: ActionCatalog) -> None:
        self._catalog = catalog
        self._tools: dict[str, ToolDef] = {}

    def register_tool(
        self,
        name: str,
        func: Callable[..., str],
        description: str,
    ) -> None:
        """Register a tool that the LLM can call."""
        self._tools[name] = ToolDef(name=name, description=description, func=func)

    async def run(
        self,
        task: str,
        llm: LLMProvider,
        max_turns: int = 20,
    ) -> ExecutionContext:
        """Execute the chain-of-action loop."""
        advisor = ActionAdvisor(self._catalog, self._tools)
        tracker = TokenTracker()
        llm.reset_session()

        ctx = ExecutionContext(task=task)
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        for turn in range(max_turns):
            ctx.turn_count = turn + 1

            # Build per-turn instructions
            if turn == 0:
                instructions = advisor.build_system_prompt()
            else:
                instructions = advisor.build_recommendation(
                    last_type=ctx.steps[-1].action_type,
                    history=ctx.steps,
                )

            schema = advisor.build_response_schema()
            response = await llm.call(messages, instructions, schema)

            # Parse structured output
            parsed = response.tool_input or {}
            action_type = parsed.get("action_type", "unknown")
            thinking = parsed.get("thinking", "")
            response_text = parsed.get("response", "")
            is_done = parsed.get("is_done", False) or action_type == "done"

            # Track cost
            tracker.record(action_type, response.usage)

            # Check if LLM followed recommendation
            if ctx.steps:
                prev_suggestions = self._catalog.get_suggestions(ctx.steps[-1].action_type)
                followed = action_type in prev_suggestions if prev_suggestions else True
            else:
                prev_suggestions = []
                followed = True

            # Execute tool if requested
            tool_name = parsed.get("tool_name")
            tool_args = parsed.get("tool_args", {})
            tool_result = None

            if tool_name and tool_name != "none":
                tool_result = self._exec_tool(tool_name, tool_args or {})
                messages.append({
                    "role": "user",
                    "content": f"[Tool: {tool_name}] Result: {tool_result}",
                })

            # Record step
            step = ActionStep(
                action_type=action_type,
                thinking=thinking,
                response=response_text,
                tool_name=tool_name if tool_name and tool_name != "none" else None,
                tool_args=tool_args if tool_name and tool_name != "none" else None,
                tool_result=tool_result,
                recommendation=prev_suggestions,
                followed_recommendation=followed,
            )
            ctx.steps.append(step)

            logger.info(
                "Turn %d: action_type=%s, tool=%s, followed=%s, done=%s",
                turn + 1, action_type, tool_name, followed, is_done,
            )

            if is_done:
                break

        # Attach cost stats
        ctx.cost_stats = tracker.stats
        return ctx

    def _exec_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a registered tool and return result as string."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        if tool.func is None:
            return f"Error: tool '{name}' has no implementation"
        try:
            result = tool.func(**args)
            return str(result)
        except Exception as e:
            return f"Error calling {name}: {e}"
