# agent-harness

[![CI](https://github.com/abhishekash/agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/abhishekash/agent-harness/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, inspectable runtime for experimenting with human approvals and traceable tool execution. The model loop, tool boundary, approval policy, and trace exporter are deliberately visible rather than hidden behind a large framework.

This is a reference and evaluation harness, not a blanket production-readiness claim. Its useful evidence is in the concrete boundaries it tests:

1. **Human-in-the-loop as a first-class mechanism** — every tool carries a `Risk` tier (`read` / `write` / `execute`); a policy decides which tiers need a human, and every human decision (approve / deny / *edit-the-arguments*) is recorded with its rationale.
2. **Observability you can grep** — every run is an OpenTelemetry trace: one span per step, LLM call, and tool call, exported to plain JSONL. Human decisions are span *events*, so "why did the agent do that?" is always answerable after the fact.

The runtime also supports mounting **stdio MCP servers** as tools (with a
small client), loading **Agent Skills** (`SKILL.md`) progressively, and
configuring budgets, retries, context compaction, and crash-safe checkpoints.

```bash
pip install -e ".[dev,opencode]" # add `,anthropic` for Anthropic
harness run "Summarize notes.md into SUMMARY.md"   # scripted demo, no API key needed
# OpenCode-compatible live run (keep the key in the environment)
# OPENCODE_API_KEY=... harness run "inspect the workspace" --provider opencode --approve cli
harness trace render traces/harness-*.jsonl
```

The default `scripted` provider is a deterministic fixture, not an intelligent
model. Use `--provider anthropic` or `--provider opencode` for a real
model-driven run.

## Mechanism demo (scripted, not model-quality evidence)

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

This run intentionally uses the deterministic `ScriptedProvider` and automatic
approval. It proves that the loop, tool calls, approval events, and JSONL
export fit together; it does **not** show a model making a good decision or
prove isolation under adversarial execution. Full artifacts in
[`examples/`](examples/demo_trace.jsonl) include raw JSONL spans, the run-store
record, the rendered timeline, and an [asciinema recording](examples/demo.cast).

## Live case-study recipe (requires a real provider)

There is no checked-in live-model result yet. The strongest scenario to run is
not the happy-path demo: ask a live model
to write to an unsafe destination, edit the proposed arguments at the approval
gate, then query the trace for the proposed path, edited path, decision, and
result. The checked-in `edited-write` task demonstrates the mechanism
reproducibly; run it against a real provider to make a model-quality claim:

```bash
export OPENCODE_API_KEY="..."       # keep credentials out of task files and commits
agent-evals run --only edited-write \
  --provider opencode --repeats 3 --out results/live-edited-write
```

Report the pass rate, each failed scorer, latency, cost, and trace ID. Do not
substitute the scripted result for this evidence; a live result needs a real
key and should be reviewed with its limitations.

## Minimal live run view

Interactive runs open a quiet terminal surface instead of spraying every model
turn into the shell. It keeps a short activity trail and a **two-line rolling
summary** of what has actually happened. After each tool action, the harness
asks the configured provider to rewrite that summary using the same model as
the run; summary calls are traced and included in token/cost accounting. The
summary is display state, not a replacement for the agent's full context.

```bash
harness run "inspect the service and fix the failing test"
harness run "task" --ui never       # plain/batch mode; no summary calls
```

The UI is automatic only on a TTY. `--ui always` forces it for a terminal that
is not detected as interactive.

## Why these choices

| Decision | Rationale |
|---|---|
| Human decisions are **trace events**, not logs | Audit trail that survives the session; queryable by tools (see [AgentHawk](https://github.com/abhishekash/agenthawk)) |
| Approver is a **protocol** (`auto` / `cli` / `callback` / `deny-all`) | Same loop runs unattended evals and interactive prod; `deny-all` exists to test graceful degradation |
| **EDIT is a decision type**, not just yes/no | Real HITL means the human can fix the call ("not `../etc`, `safe.txt`") and the agent proceeds with corrected args |
| JSONL span files as the **shared contract** | Traces are data: the MCP server, the evals repo, and the CLI renderer all read the same format |
| **ScriptedProvider** for tests/demos | Deterministic agentic loops offline; no VCR cassettes, no flaky API mocks |
| Zero-dep stdio **MCP client** (~200 LOC) | NDJSON + JSON-RPC is small enough to audit; you should be able to read your tool boundary |

## Reliability controls

Runs are bounded by default and can be configured for unattended experiments:

```bash
harness run "task" \
  --provider anthropic \
  --approve cli \
  --max-duration 300 \
  --max-input-tokens 120000 \
  --max-output-tokens 20000 \
  --max-cost 2.50
```

The harness writes an atomic, user-only checkpoint alongside the trace. A
completed run removes it; an interrupted or budget-limited run prints a resume
command. Unknown model pricing is reported as unknown rather than silently
counted as zero; a configured `--max-cost` refuses to run without pricing.
For a model not in the built-in table, provide both
`--input-price-per-million` and `--output-price-per-million`:

```bash
harness run "task" --provider opencode \
  --input-price-per-million 3 --output-price-per-million 15 \
  --max-cost 2.50
```

```bash
harness run --resume traces/harness-<timestamp>.jsonl.checkpoint.json \
  --provider anthropic --root . --approve cli
```

A checkpoint containing an in-flight tool call is refused rather than replaying
a possible side effect. Conversation context is compacted at an approximate
character budget (`--context-chars`) using the configured model.

## Human-in-the-loop

```python
from agent_harness.agent import Agent
from agent_harness.hitl import ApprovalPolicy, CLIApprover

policy = ApprovalPolicy.default()            # READ free, WRITE/EXECUTE gated
agent = Agent(provider, tools, policy=policy, approver=CLIApprover(policy=policy))
```

The Python API defaults to a deny-all approver when `approver` is omitted.
Pass `AutoApprover()` explicitly only for a controlled fixture or unattended
evaluation; the CLI defaults to the interactive gate.

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

Providers: `scripted` (deterministic, offline), `anthropic`, and OpenAI-compatible `opencode` (optional extras), plus a conservative retry wrapper for transient transport/rate-limit/server failures. Adding one = implement `complete(messages, tools) -> AssistantMessage`; optionally add `summarize(messages)` for same-model summaries and context handoffs.

## MCP: mount a server as tools

```bash
harness run "Why was run 3f2a slow?" --mcp "uvx agenthawk --trace-dir ./traces" --approve cli
```

Remote tools are namespaced (`agenthawk__slowest_spans`) and default to `Risk.EXECUTE` — i.e. **gated until you say otherwise**:

```python
client, tools = mount_server(
    ["uvx", "agenthawk"],
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
| `harness run TASK [--provider scripted|anthropic|opencode] [--approve ...] [--mcp CMD] [--skills DIR] [--trace FILE] [--ui ...] [--max-*] [--checkpoint/--resume]` | Run a bounded, observable agent |
| `harness trace render FILE [--trace-id ID]` | Timeline with tokens, cost, human decisions |
| `harness runs list [--runs-db FILE]` | Local run store: steps, cost, outcome |

## Testing philosophy

64 tests, zero network. The loop, HITL matrix, sandbox escapes, MCP protocol, retry behavior, context compaction, checkpoint/resume boundaries, trace rendering, tool-error observability, cost-budget handling, and two-line progress surface are all tested against deterministic providers and a hand-rolled NDJSON MCP fixture server (`tests/fixtures/echo_mcp_server.py`). Live-provider quality and the end-to-end shell-boundary debugging story are evaluated separately by [`agent-evals`](https://github.com/abhishekash/agent-evals).

## Design postmortem

Read the [architecture postmortem](docs/architecture-postmortem.md) for the decisions behind the shared trace contract, HITL model, MCP boundary, deterministic evals, and known limits.

## Honest limitations

- **Single-agent, single-threaded.** No sub-agent orchestration yet — the trace format is designed for it (parent spans), the loop isn't.
- Context compaction uses an approximate character budget rather than provider-native token counting; long-running production workloads should tune `--context-chars` and validate against their model.
- The Anthropic adapter is thin and *not* covered by the offline test-suite; live retries, provider semantics, and model quality need networked evaluation.
- MCP client is stdio-only (no HTTP/SSE transport yet), ignores server-initiated requests, and defaults remote tools to execute risk.
- Shell execution is allowlisted and path-checked, but it is not an OS-level sandbox; extend `RunShell.ALLOWLIST` knowingly.

## Roadmap

- [ ] Sub-agent spans (delegate task → child trace linked by parent)
- [ ] Provider-native token counting and streaming
- [ ] Webhook approver (Slack/HTTP) for async human gates
- [ ] OS/container sandbox for execute tools
- [ ] MCP Streamable HTTP transport
- [ ] OTel OTLP export (ship spans to a real collector)

## License

MIT
