from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Usage stats from a single LLM call."""
    cost_usd: float = 0.0
    duration_ms: int = 0


@dataclass
class LLMResponse:
    """Parsed response from an LLM call."""
    tool_input: dict[str, Any] | None  # Parsed structured output
    usage: TokenUsage
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{name, args, result}]
    raw_content: str = ""  # Raw result text
    session_id: str = ""


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def call(
        self,
        messages: list[dict[str, Any]],
        system: str,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        """Make an LLM call and return structured response."""
        ...

    def reset_session(self) -> None:
        """Reset provider state for a new run. Override if stateful."""
        pass

    async def setup_tools(self, server_name: str, command: list[str]) -> None:
        """Register MCP tools before a run. Override in providers that support MCP."""
        pass

    async def cleanup_tools(self) -> None:
        """Remove registered MCP tools after a run. Override in providers that support MCP."""
        pass
