# agent-harness

A hackable agent runtime for people who want to *see* what their agent is doing.

Most agent frameworks optimize for demos: magic in, magic out. This one optimizes for the two things production agents actually need:

1. **Human-in-the-loop as a first-class mechanism** — every tool carries a `Risk` tier (`read` / `write` / `execute`); a policy decides which tiers need a human, and every human decision (approve / deny / *edit-the-arguments*) is recorded with its rationale.
2. **Observability you can grep** — every run is an OpenTelemetry trace: one span per step, LLM call, and tool call, exported to plain JSONL. Human decisions are span *events*, so "why did the agent do that?" is always answerable after the fact.

Plus the two extension points that matter in 2026: mount any **stdio MCP server** as tools (zero-SDK client included), and load **Agent Skills** (`SKILL.md`) with progressive disclosure.

```bash
pip install -e ".[dev]"        # or: uv pip install -e ".[dev]"
harness run "Summarize notes.md into SUMMARY.md"   # scripted demo, no API key needed
harness trace render traces/harness-*.jsonl
```

## Demo (real output, `--provider scripted --approve auto`)

```
trace f920798dd2558fcb84f9977736432a8b
======================================
agent.run       4.0ms
  agent.step       0.5ms
    llm.complete       0.0ms  [scripted · 92→9 tok]
    tool.call       0.2ms  [list_dir · risk=read]
  agent.step       0.3ms
    llm.complete       0.0ms  [scripted · 125→9 tok]
    tool.call       0.2ms  [write_file · risk=write]
      🔐 human: approve by auto — auto-approved (unattended run)
  agent.step       2.5ms
    llm.complete       0.0ms  [scripted · 132→9 tok]
    tool.call       2.3ms  [run_shell · risk=execute]
      🔐 human: approve by auto — auto-approved (unattended run)
total: 4.0ms across 15 spans
```

Full artifacts in [`examples/`](examples/demo_trace.jsonl): raw JSONL spans, run-store record, rendered timeline.

## Why these choices

| Decision | Rationale |
|---|---|
| Human decisions are **trace events**, not logs | Audit trail that survives the session; queryable by tools (see [mcp-trace](https://github.com/abhishekash/mcp-trace)) |
| Approver is a **protocol** (`auto` / `cli` / `callback` / `deny-all`) | Same loop runs unattended evals and interactive prod; `deny-all` exists to test graceful degradation |
| **EDIT is a decision type**, not just yes/no | Real HITL means the human can fix the call ("not `../etc`, `safe.txt`") and the agent proceeds with corrected args |
| JSONL span files as the **shared contract** | Traces are data: the MCP server, the evals repo, and the CLI renderer all read the same format |
| **ScriptedProvider** for tests/demos | Deterministic agentic loops offline; no VCR cassettes, no flaky API mocks |
| Zero-dep stdio **MCP client** (~200 LOC) | NDJSON + JSON-RPC is small enough to audit; you should be able to read your tool boundary |

## Human-in-the-loop

```python
from agent_harness.agent import Agent
from agent_harness.hitl import ApprovalPolicy, CLIApprover

policy = ApprovalPolicy.default()            # READ free, WRITE/EXECUTE gated
agent = Agent(provider, tools, policy=policy, approver=CLIApprover(policy=policy))
```

Interactive gate (stderr):

```
────────────────────────────────────────────
⚠  approval required — write_file [write]
   write 42 chars to SUMMARY.md
{ "path": "SUMMARY.md", "content": "# Summary\n…" }
[y]es / [n]o / [e]dit args / [a]lways this tool:
```

- `n` → the denial **and your reason** are fed back to the model ("respect the reason, pick another approach")
- `e` → supply replacement arguments as JSON; the call proceeds with *your* version
- `a` → pre-approve this tool for the rest of the session (you control how much control you want)

## Architecture

```
                 ┌────────────────────────── Agent.run(task) ──────────────────────────┐
                 │                                                                      │
   system prompt │  ┌──────────┐   tool_calls   ┌─────────────┐   result   ┌────────┐  │  spans:
   + skill index ┼─▶│ provider │───────────────▶│  HITL gate  │──────────▶│  tool  │  │  agent.run
                 │  │ (pluggable)│◀──────────────│ risk policy │◀──────────│ (sandbox)│  │  ├─ agent.step
                 │  └──────────┘   tool result   └──────┬──────┘            └────────┘  │  │   ├─ llm.complete (tokens, $)
                 │                                      │ approve/deny/edit              │  │   └─ tool.call (risk, duration)
                 │                              🔐 decision + rationale                 │  │      └─ event: hitl.decision
                 │                              → trace event                           │
                 └──────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                          traces/*.jsonl  (OTel spans, one per line)
                          runs.jsonl      (task, cost, approvals per run)
```

Providers: `scripted` (deterministic, offline), `anthropic` (optional extra). Adding one = implement `complete(messages, tools) -> AssistantMessage`.

## MCP: mount a server as tools

```bash
harness run "Why was run 3f2a slow?" --mcp "uvx --from git+https://github.com/abhishekash/mcp-trace.git mcp-trace --trace-dir ./traces" --approve cli
```

Remote tools are namespaced (`mcp-trace__slowest_spans`) and default to `Risk.EXECUTE` — i.e. **gated until you say otherwise**:

```python
client, tools = mount_server(
    ["uvx", "--from", "git+https://github.com/abhishekash/mcp-trace.git", "mcp-trace"],
    risk_overrides={"list_runs": Risk.READ},
)
```

## Skills

```bash
harness run "task" --skills ./my-skills   # each subdir has a SKILL.md
```

The agent's system prompt gets the *index* (name + description) only; bodies load on selection. Progressive disclosure keeps context cheap. See [agent-skills](https://github.com/abhishekash/agent-skills) for the format.

## CLI

| Command | What it does |
|---|---|
| `harness run TASK [--approve cli\|auto\|deny] [--mcp CMD] [--skills DIR] [--trace FILE]` | Run the agent |
| `harness trace render FILE [--trace-id ID]` | Timeline with tokens, cost, human decisions |
| `harness runs list [--runs-db FILE]` | Local run store: steps, cost, outcome |

## Testing philosophy

48 tests, zero network. The loop, HITL matrix, sandbox escapes, MCP protocol, and trace rendering are all tested against the `ScriptedProvider` and a hand-rolled NDJSON MCP fixture server (`tests/fixtures/echo_mcp_server.py`). If a test needs the network, the design is wrong.

## Honest limitations

- **Single-agent, single-threaded.** No sub-agent orchestration yet — the trace format is designed for it (parent spans), the loop isn't.
- Context management is naive: long runs grow the message list unbounded. Truncation/summarization is roadmap, not present.
- The Anthropic adapter is thin and *not* covered by the offline test-suite; the scripted provider is the reference implementation.
- MCP client is stdio-only (no HTTP/SSE transport yet) and ignores server-initiated requests.
- Shell tool allowlist is intentionally tiny; extend `RunShell.ALLOWLIST` knowingly.

## Roadmap

- [ ] Sub-agent spans (delegate task → child trace linked by parent)
- [ ] Context window budgeting + summarization middleware
- [ ] Webhook approver (Slack/HTTP) for async human gates
- [ ] MCP Streamable HTTP transport
- [ ] OTel OTLP export (ship spans to a real collector)

## License

MIT
