from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.action_type import ActionCatalog
from src.core.advisor import ActionAdvisor
from src.core.context import ActionStep, ExecutionContext
from src.llm.base import LLMProvider
from src.monitoring.tracker import TokenTracker

logger = logging.getLogger(__name__)

# Default MCP server path (relative to project root)
_DEFAULT_MCP_SERVER = Path(__file__).resolve().parent.parent.parent / "tools" / "mcp_server.py"


class Engine:
    """Main execution loop for chain-of-action.

    Two-phase loop: plan generation → guided execution.
    Tools are executed natively by Claude CLI via MCP servers.
    """

    def __init__(self, catalog: ActionCatalog) -> None:
        self._catalog = catalog
        self._mcp_server_name: str | None = None
        self._mcp_command: list[str] | None = None

    def register_mcp_server(self, name: str, command: list[str]) -> None:
        """Register an MCP server for tool execution.

        Args:
            name: Unique server name for this session.
            command: Command to start the server, e.g. ["uv", "run", "python", "tools/mcp_server.py"].
        """
        self._mcp_server_name = name
        self._mcp_command = command

    async def run(
        self,
        task: str,
        llm: LLMProvider,
        max_turns: int = 20,
    ) -> ExecutionContext:
        """Execute the two-phase chain-of-action loop.

        Phase 1 (turn 0): Generate an execution plan from the catalog.
        Phase 2 (turns 1+): Execute steps guided by the plan.
        Tools are called natively by Claude via MCP.
        """
        advisor = ActionAdvisor(self._catalog)
        tracker = TokenTracker()
        llm.reset_session()

        # Register MCP tools
        if self._mcp_server_name and self._mcp_command:
            await llm.setup_tools(self._mcp_server_name, self._mcp_command)

        try:
            return await self._run_loop(task, llm, advisor, tracker, max_turns)
        finally:
            await llm.cleanup_tools()

    async def _run_loop(
        self,
        task: str,
        llm: LLMProvider,
        advisor: ActionAdvisor,
        tracker: TokenTracker,
        max_turns: int,
    ) -> ExecutionContext:
        ctx = ExecutionContext(task=task)
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        # ── Phase 1: Plan generation (turn 0) ──
        plan_prompt = advisor.build_plan_prompt()
        plan_schema = advisor.build_plan_schema()
        plan_response = await llm.call(messages, plan_prompt, plan_schema)

        parsed_plan = plan_response.tool_input or {}
        ctx.plan = parsed_plan.get("plan", [])
        tracker.record("plan", plan_response.usage)
        ctx.turn_count = 1

        logger.info("Plan generated: %s", ctx.plan)

        # ── Phase 2: Execution (turns 1+) ──
        for turn in range(1, max_turns):
            ctx.turn_count = turn + 1

            # Build per-turn instructions
            if turn == 1 and not ctx.steps:
                instructions = advisor.build_system_prompt()
            else:
                instructions = advisor.build_recommendation(
                    last_type=ctx.steps[-1].action_type if ctx.steps else "plan",
                    history=ctx.steps,
                    plan=ctx.plan,
                    plan_cursor=ctx.plan_cursor,
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

            # Plan adherence: check if action matches plan expectation
            planned_type = None
            if ctx.plan and ctx.plan_cursor < len(ctx.plan):
                planned_type = ctx.plan[ctx.plan_cursor].get("action_type")

            # Advance plan cursor: always advance (plan is a guide, not a gate)
            if ctx.plan and ctx.plan_cursor < len(ctx.plan):
                ctx.plan_cursor += 1

            # Record step — tool_calls come from the LLM response (MCP native)
            step = ActionStep(
                action_type=action_type,
                thinking=thinking,
                response=response_text,
                tool_calls=response.tool_calls,
                planned_type=planned_type,
            )
            ctx.steps.append(step)

            logger.info(
                "Turn %d: action=%s, planned=%s, matched=%s, tools=%d, done=%s",
                turn + 1, action_type, planned_type,
                action_type == planned_type if planned_type else True,
                len(response.tool_calls), is_done,
            )

            if is_done:
                break

        # Attach cost stats
        ctx.cost_stats = tracker.stats
        return ctx
