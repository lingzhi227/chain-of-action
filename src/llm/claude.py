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
    "You self-classify your action type. Recommendations are suggestions, not constraints."
)


class ClaudeProvider(LLMProvider):
    """Claude LLM provider using the Claude Code CLI.

    Maintains a single session across all calls in one run.
    First call creates the session; subsequent calls use --resume.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._session_id: str | None = None
        self._msgs_at_last_call: int = 0

    def reset_session(self) -> None:
        """Reset for a new run."""
        self._session_id = None
        self._msgs_at_last_call = 0

    async def call(
        self,
        messages: list[dict[str, Any]],
        system: str,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        """Call Claude CLI within a persistent session."""
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
                "--output-format", "json",
                "--system-prompt", GLOBAL_SYSTEM_PROMPT,
                "--dangerously-skip-permissions",
                prompt,
            ]
        else:
            # --- Subsequent calls: resume session ---
            new_messages = messages[self._msgs_at_last_call:]
            new_user_parts = []
            for m in new_messages:
                if m["role"] == "user":
                    content = m["content"]
                    if isinstance(content, str):
                        new_user_parts.append(content)

            if new_user_parts:
                prompt = "\n\n".join(new_user_parts) + f"\n\n---\n\n{state_prompt}"
            else:
                prompt = state_prompt

            cmd = [
                "claude", "--print",
                "--resume", self._session_id,
                "--output-format", "json",
                "--dangerously-skip-permissions",
                prompt,
            ]

        self._msgs_at_last_call = len(messages)

        logger.debug(
            "Claude call (session=%s, resume=%s)",
            self._session_id,
            "--resume" in cmd,
        )

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

        result_text, cost_usd, duration_ms = _parse_json_output(
            stdout_bytes.decode(errors="replace")
        )

        parsed = _parse_json_response(result_text)

        return LLMResponse(
            tool_input=parsed,
            usage=TokenUsage(cost_usd=cost_usd, duration_ms=duration_ms),
            raw_content=result_text,
            session_id=self._session_id,
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
        else:
            example[key] = f"<{key}>"

    return (
        f"[INSTRUCTIONS]\n{system}\n\n"
        f"[REQUIRED JSON SCHEMA]\n{json.dumps(schema, indent=2)}\n\n"
        f"[EXAMPLE FORMAT]\n{json.dumps(example, indent=2)}\n\n"
        f"Respond with ONLY the JSON object."
    )


def _parse_json_output(output: str) -> tuple[str, float, int]:
    """Parse --output-format json envelope from claude CLI."""
    if not output.strip():
        return "", 0.0, 0
    try:
        data = json.loads(output)
        return (
            data.get("result", ""),
            data.get("cost_usd", 0.0),
            data.get("duration_ms", 0),
        )
    except json.JSONDecodeError:
        logger.warning("Could not parse CLI JSON envelope: %s", output[:200])
        return output.strip(), 0.0, 0


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
