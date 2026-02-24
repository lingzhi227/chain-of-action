from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "haiku"

GLOBAL_SYSTEM_PROMPT = (
    "You are a task-solving agent with soft action guidance. "
    "At each step you receive recommendations about action types and a JSON schema.\n"
    "You MUST respond with ONLY a valid JSON object matching the provided schema.\n"
    "No markdown, no code fences, no explanation outside the JSON.\n"
    "You self-classify your action type. Recommendations are suggestions, not constraints.\n"
    "You have MCP tools available — use them directly when you need to compute or verify."
)


class ClaudeProvider(LLMProvider):
    """Claude LLM provider using the Claude Code CLI with native MCP tool execution.

    Maintains a single session across all calls in one run.
    First call creates the session; subsequent calls use --resume.
    Tools are executed natively by Claude CLI via MCP servers.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._session_id: str | None = None
        self._mcp_server_name: str | None = None

    def reset_session(self) -> None:
        """Reset for a new run."""
        self._session_id = None

    async def setup_tools(self, server_name: str, command: list[str]) -> None:
        """Register an MCP server with Claude CLI."""
        self._mcp_server_name = server_name
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = [
            "claude", "mcp", "add", "--transport", "stdio", server_name, "--",
        ] + command
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Failed to register MCP server: %s", stderr.decode(errors="replace")[:500])
        else:
            logger.info("Registered MCP server: %s", server_name)

    async def cleanup_tools(self) -> None:
        """Remove the registered MCP server from Claude CLI."""
        if not self._mcp_server_name:
            return
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        cmd = ["claude", "mcp", "remove", self._mcp_server_name]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
        )
        await proc.communicate()
        logger.info("Removed MCP server: %s", self._mcp_server_name)
        self._mcp_server_name = None

    async def call(
        self,
        messages: list[dict[str, Any]],
        system: str,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        """Call Claude CLI within a persistent session. Tools execute natively via MCP."""
        state_prompt = _build_state_prompt(system, response_schema)
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        if self._session_id is None:
            # --- First call: create session ---
            self._session_id = str(uuid.uuid4())

            task = ""
            for m in messages:
                if m["role"] == "user":
                    task = str(m["content"])
                    break

            prompt = f"{task}\n\n---\n\n{state_prompt}"

            cmd = [
                "claude", "--print",
                "--model", self._model,
                "--session-id", self._session_id,
                "--output-format", "stream-json",
                "--verbose",
                "--system-prompt", GLOBAL_SYSTEM_PROMPT,
                "--dangerously-skip-permissions",
                prompt,
            ]
        else:
            # --- Subsequent calls: resume session ---
            prompt = state_prompt

            cmd = [
                "claude", "--print",
                "--resume", self._session_id,
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                prompt,
            ]

        logger.debug("Claude call (session=%s, resume=%s)", self._session_id, "--resume" in cmd)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace")
            logger.error("Claude CLI failed (rc=%d): %s", proc.returncode, stderr_text[:500])

        stdout_text = stdout_bytes.decode(errors="replace")

        # Parse stream-json events
        events = _parse_stream_events(stdout_text)
        tool_calls = _extract_tool_calls(events)
        result_text = _extract_result_text(events)
        cost_usd, duration_ms = _extract_usage(events)

        parsed = _parse_json_response(result_text)

        return LLMResponse(
            tool_input=parsed,
            usage=TokenUsage(cost_usd=cost_usd, duration_ms=duration_ms),
            tool_calls=tool_calls,
            raw_content=result_text,
            session_id=self._session_id or "",
        )


def _build_state_prompt(system: str, schema: dict[str, Any]) -> str:
    """Combine instructions with JSON schema requirement."""
    example = {}
    for key, prop in schema.get("properties", {}).items():
        if "enum" in prop:
            example[key] = prop["enum"][0]
        elif prop.get("type") == "object":
            example[key] = {}
        elif prop.get("type") == "boolean":
            example[key] = False
        elif prop.get("type") == "array":
            example[key] = []
        else:
            example[key] = f"<{key}>"

    return (
        f"[INSTRUCTIONS]\n{system}\n\n"
        f"[REQUIRED JSON SCHEMA]\n{json.dumps(schema, indent=2)}\n\n"
        f"[EXAMPLE FORMAT]\n{json.dumps(example, indent=2)}\n\n"
        f"Respond with ONLY the JSON object."
    )


def _parse_stream_events(output: str) -> list[dict[str, Any]]:
    """Parse NDJSON stream-json output into a list of events."""
    events = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON line: %s", line[:100])
    return events


def _extract_tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract MCP tool calls from stream events, matching tool_use with tool_result."""
    # Collect tool_use blocks
    pending: dict[str, dict[str, Any]] = {}  # tool_use_id -> {name, args}
    results: list[dict[str, Any]] = []

    for event in events:
        msg = event.get("message", {})
        content_blocks = msg.get("content", [])
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        name = block.get("name", "")
                        # Strip MCP prefix: mcp__server-name__tool -> tool
                        short_name = name.split("__")[-1] if "__" in name else name
                        pending[tool_id] = {
                            "name": short_name,
                            "full_name": name,
                            "args": block.get("input", {}),
                            "result": None,
                        }
                    elif block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id", "")
                        if tool_id in pending:
                            pending[tool_id]["result"] = block.get("content", "")

    # Return in order
    return list(pending.values())


def _extract_result_text(events: list[dict[str, Any]]) -> str:
    """Extract the final result text from the stream-json result event."""
    for event in events:
        if event.get("type") == "result":
            return event.get("result", "")
    # Fallback: collect text blocks from assistant messages
    texts = []
    for event in events:
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
    return "\n".join(texts)


def _extract_usage(events: list[dict[str, Any]]) -> tuple[float, int]:
    """Extract cost and duration from the result event."""
    for event in events:
        if event.get("type") == "result":
            return (
                event.get("total_cost_usd", 0.0),
                event.get("duration_ms", 0),
            )
    return 0.0, 0


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse JSON from response: %s", text[:200])
    return None
