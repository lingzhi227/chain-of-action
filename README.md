# chain-of-action

Soft action guidance framework for LLM agents. The LLM self-classifies its behavior into action types and receives recommendations for what to do next — but is never forced. All tools remain available at every step.

Counterpart to [finite-state-agent](https://github.com/lingzhi227/action-chain) (hard FSM). Same problem, opposite philosophy.

## Why

Hard FSM control (enum-enforced transitions, tool whitelists) is effective but rigid. It assumes you can predefine the optimal execution graph. chain-of-action tests an alternative:

- **LLM has full agency** — self-classifies action type as a free string, not an enum
- **Tools always available** — no whitelist filtering, trust the LLM to choose
- **Recommendations, not constraints** — suggested next actions injected into context each turn
- **Adherence tracking** — measures whether the LLM follows recommendations, enabling Bayesian action priors

Inspired by StateFlow (per-state prompts, -81% tokens), AFlow (operator type catalogs), and DEPO (asymmetric incentives).

## Architecture

```
src/
  core/
    action_type.py   ActionType dataclass + ActionCatalog (type registry)
    advisor.py        Builds system prompt + per-turn recommendations
    context.py        ActionStep trace + ExecutionContext (analytics)
    engine.py         Main loop: call LLM → classify → tool exec → recommend → repeat
  llm/
    base.py           LLMProvider ABC, LLMResponse, TokenUsage
    claude.py         ClaudeProvider (claude CLI, single-session --resume)
  monitoring/
    tracker.py        Per-action-type cost tracking
```

## How it works

```
         ┌──────────┐
 task -> │ analyze  │ ─ ─ recommended ─ ─ ┐
         └──────────┘                      v
              :                       ┌─────────┐
              : (LLM chooses)         │  plan   │
              v                       └─────────┘
         ┌──────────┐                      :
         │ compute  │ <── ── ── ── ── ── ─┘
         └──────────┘
              │ ─ ─ recommended ─ ─ ┐
              v                     v
         ┌──────────┐         ┌──────────┐
         │ compute  │         │  verify  │
         └──────────┘         └──────────┘
                                   │
                                   v
                             ┌────────────┐     ┌──────┐
                             │ synthesize │ --> │ done │
                             └────────────┘     └──────┘

  ─── = hard transition        ─ ─ = soft recommendation
```

Each turn:
1. Engine sends per-turn recommendation: "Your last action was [compute]. Recommended next: [verify, compute]."
2. LLM self-classifies its `action_type` (free string — can even invent new types)
3. LLM calls any tool it wants (all always available)
4. Engine records whether the LLM followed the recommendation
5. Repeat until `is_done: true` or action_type is `done`

## Setup

```bash
git clone https://github.com/lingzhi227/chain-of-action.git
cd chain-of-action
uv sync --extra dev
```

Requires: [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated.

## Usage

### Define an action catalog

```python
from src.core.action_type import ActionType, ActionCatalog
from src.core.engine import Engine
from src.llm.claude import ClaudeProvider

catalog = ActionCatalog()
catalog.add(ActionType(
    name="analyze",
    description="Understand the problem",
    suggested_next=["plan", "compute"],
))
catalog.add(ActionType(
    name="compute",
    description="Perform a calculation using tools",
    suggested_next=["verify", "compute"],
    tools=["calc"],  # informational, NOT filtering
))
catalog.add(ActionType(
    name="verify",
    description="Check a previous computation",
    suggested_next=["compute", "synthesize"],
))
catalog.add(ActionType(
    name="synthesize",
    description="Combine results into a final answer",
    suggested_next=["done"],
))
catalog.add(ActionType(
    name="done",
    description="Task complete",
))
```

### Register tools and run

```python
import asyncio

engine = Engine(catalog)
engine.register_tool("calc", lambda expression: str(eval(expression)), "Arithmetic")

async def main():
    llm = ClaudeProvider(model="haiku")
    ctx = await engine.run("What is 347 * 923?", llm, max_turns=10)

    print(ctx.action_type_counts())   # {'analyze': 1, 'compute': 1, 'done': 1}
    print(ctx.adherence_rate())       # 0.75
    print(ctx.transition_matrix())    # {'analyze': {'compute': 1}, ...}

asyncio.run(main())
```

### Analytics

`ExecutionContext` provides built-in analytics:

```python
ctx.action_type_counts()    # {type: count} — how often each action type was used
ctx.transition_matrix()     # {src: {dst: count}} — observed action flows
ctx.adherence_rate()        # float — % of steps that followed recommendation
ctx.cost_stats              # {type: TokenStats} — cost/duration per action type
```

## Examples

**Salary projection** — 3 engineers, compound raises, 4 years. 3 tools (`calc`, `compound`, `stats`):
```bash
uv run python examples/salary_analysis.py
```

Outputs `TRACE.md` with full execution trace, transition matrix, adherence rate, and cost breakdown.

## Tests

```bash
# Unit tests (no LLM calls, instant)
uv run pytest tests/test_core.py -v

# Integration tests (real Claude CLI calls)
uv run pytest tests/test_integration.py -v
```

35 unit tests cover: action types, catalog, advisor prompt/schema generation, execution context analytics, engine tool execution, token tracking.

3 integration tests cover: simple Q&A, multi-tool salary analysis, adherence tracking verification.

## chain-of-action vs finite-state-agent

| Aspect | finite-state-agent (hard FSM) | chain-of-action (soft guidance) |
|---|---|---|
| State control | Enum-enforced transitions | LLM self-classifies freely |
| Tool access | Whitelisted per state | All tools always available |
| Flexibility | Rigid: must follow graph | Flexible: recommendations only |
| Predictability | High — deterministic graph | Variable — LLM has agency |
| New behaviors | Impossible without graph change | LLM can invent new action types |
| Observability | State trace | Action trace + adherence rate |

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Action type selection | LLM self-classifies (free string) | Soft guidance — LLM retains full agency |
| Recommendations | Injected as instructions each turn | Context-based nudging, not schema constraint |
| Tool access | All tools always available | No whitelist filtering — trust the LLM |
| Session management | Single session + `--resume` | Full conversation context without re-sending history |
| Tracking | Per-action-type cost + adherence rate | Measures guidance effectiveness |

## License

MIT
