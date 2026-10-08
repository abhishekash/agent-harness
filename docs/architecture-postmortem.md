# Building an Observable Agent Harness: Architecture Postmortem

Date: 2026-10-08

This is the design record for `agent-harness`, `mcp-trace`, `agent-skills`, and `agent-evals`. It documents what was built, what was deliberately not built, and what the first deterministic results actually prove.

## The problem

Agent demos usually show a final answer and hide the important questions:

- Which tool calls had side effects?
- Did a human approve them, deny them, or edit their arguments?
- Why did a run loop or become expensive?
- Can another agent inspect the failure without opening a dashboard?
- Did a prompt change improve behavior or merely improve one demo?

The portfolio is organized around one answer: **make execution, human judgment, observability, and measurement share a contract.**

```text
agent-harness  ──emits──▶  JSONL OTel spans  ──queried by──▶  mcp-trace
      │                                                          │
      └────────────── exercised and scored by agent-evals ◀──────┘
                              ▲
                    agent-skills teaches the workflow
```

## Decisions

### 1. A small provider-neutral loop before orchestration

The core loop is intentionally ordinary:

```text
messages → provider.complete → tool calls → tool results → repeat
```

Provider adapters translate into `AssistantMessage`, `ToolCall`, and `Usage`. The loop owns policy, tracing, and tool execution; providers do not. This makes it possible to test the important behavior without an API key or a network replay system.

**Not built yet:** sub-agents, planner/executor graphs, queues, or distributed state. Those features would add surface area before the single-agent contract is trustworthy.

### 2. Risk tiers are the HITL API

Every tool declares one of three risks:

- `read`: no side effect; free by default
- `write`: local mutation; gated by default
- `execute`: shell, network, or remote side effect; gated by default

An `ApprovalPolicy` maps risk to a gate. An `Approver` is a protocol with four implementations: interactive CLI, automatic approval for deterministic evals, deny-all for safety tests, and callbacks for embedding applications.

The important decision is that **edit is a first-class decision**, not a yes/no afterthought. A human can change `../outside.txt` to `safe.txt`, and the executed tool receives the changed arguments.

Denials are returned to the model as tool results with the human rationale and an explicit instruction not to retry the same call. This is both safer and more useful than silently dropping a call.

### 3. Human decisions are trace events

Every run has a root `agent.run` span. Each model turn, provider call, and tool call is a child span. A HITL decision is an event on the gated tool span:

```json
{
  "name": "hitl.decision",
  "attributes": {
    "hitl.decision": "deny",
    "hitl.approver": "cli",
    "hitl.rationale": "too broad",
    "hitl.edited": false
  }
}
```

This makes an approval auditable without requiring a second logging system. `mcp-trace` can answer "what did the human deny?" using the same data that `harness trace render` displays.

### 4. JSONL is the first trace transport

The exporter writes one JSON object per completed span. JSONL was chosen because it is:

- inspectable with standard tools;
- append-friendly;
- easy to attach to an eval result;
- independent of a hosted observability vendor;
- simple for an MCP server to query.

The format is shaped like OTel spans but is not a replacement for OTLP. OTLP export is a roadmap item once the event and attribute contract stabilizes.

### 5. MCP is the self-observability boundary

`mcp-trace` is a thin MCP 2.x server over pure query functions. It exposes:

- recent run summaries;
- nested span trees;
- slowest spans;
- approval audit logs;
- token and cost totals;
- substring search over span names and attributes.

Tool descriptions say **when to use** a tool, not merely what it returns. Results are compact by default and trace IDs accept prefixes because full IDs are poor model-facing identifiers.

The harness includes a small, auditable stdio MCP client. Remote tools default to `execute` risk until explicitly overridden. That is conservative by design.

### 6. Skills use progressive disclosure

The harness loads the name and description of each `SKILL.md` into the system context, but does not inject every skill body. The `agent-skills` repository treats descriptions as routing prompts and requires failure notes in every skill.

The format is intentionally plain Markdown so it can be consumed by different harnesses. Compatibility claims should be backed by tests as more harnesses are added.

### 7. Evals measure mechanisms before model quality

The first eval baseline uses a scripted provider. It does **not** claim that a model is intelligent. It verifies that the runtime mechanisms work:

- read → write execution;
- denied side effects;
- human-edited arguments;
- step budgets;
- unknown-tool recovery;
- token budgets;
- trace events;
- multi-step file pipelines.

The current run is 8/8 passing. A live-model runner should report model/version, sampling settings, repeats, pass rates, cost, latency, and failed-task traces separately.

## Evidence from the first build

| Artifact | Evidence |
|---|---|
| Harness | 48 offline tests; real demo trace with 15 spans and two approval events |
| MCP server | 19 tests; the harness MCP client successfully mounted all seven query tools |
| Skills | Four skills validated by a dependency-free frontmatter/routing checker |
| Evals | Eight deterministic tasks passing, including denial and edited-argument safety contracts |
| Operations | GitHub Actions green; MIT licenses; contributor guides; `v0.1.0` releases |

## What surprised us

1. **The trace contract became the integration surface.** Once spans had stable names and attributes, the MCP server and eval runner were straightforward. Without that contract, each project would invent its own logging.
2. **Human edit is more valuable than human approval.** Approval prevents an unsafe call; edit lets a human redirect a basically-correct call without spending another model turn.
3. **Deterministic providers expose runtime bugs quickly.** They make it possible to assert the exact number of tool calls, the exact side effect, and the exact approval event before introducing model nondeterminism.
4. **Install truth matters.** A README must not say `uvx mcp-trace` until the package is actually on PyPI. The repository now builds clean distributions, includes the PyPI ownership marker and registry manifest, and uses a trusted-publishing workflow that will be enabled after PyPI ownership is configured.

## Known limits

- The loop is single-agent and single-threaded.
- Context grows without summarization or compaction.
- MCP support is stdio-only and does not handle server-initiated requests.
- The CLI approver is synchronous; webhook/Slack approval is not implemented.
- The eval suite is small and scripted; it is a mechanism baseline, not a model leaderboard.
- JSONL is local observability, not a production collector.

## Next experiments

1. Configure PyPI trusted publishing and publish `mcp-trace` 0.1.0.
2. Publish the validated `server.json` through `mcp-publisher`.
3. Add an OTLP exporter and compare local JSONL with a real collector.
4. Add context-budget middleware and an eval that catches context growth.
5. Add a live-provider runner with repeated trials and cost/latency reporting.
6. Contribute a focused fix or documentation improvement to a core MCP/pi/skills repository.
