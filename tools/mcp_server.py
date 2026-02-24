"""MCP server exposing calc, compound, and stats tools for chain-of-action."""
from __future__ import annotations

import statistics as stats_mod

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("chain-of-action-tools")


@mcp.tool()
def calc(expression: str) -> str:
    """Evaluate an arithmetic expression. Example: calc(expression='2 + 3 * 4')"""
    allowed = set("0123456789+-*/.() ,")
    if not all(c in allowed for c in expression):
        return f"Error: invalid characters in expression"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def compound(base: float, rate: float, years: int) -> str:
    """Calculate compound interest. Example: compound(base=100000, rate=0.05, years=4)"""
    result = base * ((1 + rate) ** years)
    return f"{result:.2f}"


@mcp.tool()
def stats(values: list[float]) -> str:
    """Calculate mean, median, stdev. Example: stats(values=[1.0, 2.0, 3.0])"""
    if not values:
        return "Error: empty list"
    if len(values) == 1:
        return f"mean={values[0]:.2f}, median={values[0]:.2f}, stdev=0.00"
    return (
        f"mean={stats_mod.mean(values):.2f}, "
        f"median={stats_mod.median(values):.2f}, "
        f"stdev={stats_mod.stdev(values):.2f}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
