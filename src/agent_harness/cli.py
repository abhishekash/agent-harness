"""`harness` command line.

    harness run "task"        run the agent (scripted demo provider or anthropic)
    harness trace render F    render a JSONL trace file as a timeline
    harness runs list         show the local run store
"""
from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from agent_harness.agent import Agent
from agent_harness.hitl import AutoApprover, CLIApprover, DenyAllApprover
from agent_harness.providers import call_tool, say, scripted_run
from agent_harness.skills import discover_skills
from agent_harness.store import list_runs, record_run
from agent_harness.tools import default_tools
from agent_harness.tracing import init_tracing, render_file
from agent_harness.types import Risk

DEMO_TASK = "Summarize the workspace's notes into SUMMARY.md, then verify it with ls"


def _demo_provider():
    """A scripted run that exercises read (auto), write (gate), shell (gate)."""
    return scripted_run(
        call_tool("list_dir", {}),
        call_tool("read_file", {"path": "notes.md"}),
        call_tool(
            "write_file",
            {"path": "SUMMARY.md", "content": "# Summary\n\nConsolidated from notes.md.\n"},
        ),
        call_tool("run_shell", {"command": "ls -la"}),
        say("Done: read notes.md, wrote SUMMARY.md, verified with ls."),
    )


def _build_provider(args):
    if args.provider == "scripted":
        return _demo_provider()
    if args.provider == "anthropic":
        from agent_harness.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=args.model)
    raise SystemExit(f"unknown provider: {args.provider}")


def _build_approver(name: str, policy):
    if name == "auto":
        return AutoApprover()
    if name == "deny":
        return DenyAllApprover()
    return CLIApprover(policy=policy)


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    tools = default_tools(root)
    mcp_clients = []
    for spec in args.mcp or []:
        from agent_harness.mcp import mount_server

        client, mcp_tools = mount_server(shlex.split(spec))
        mcp_clients.append(client)
        tools.extend(mcp_tools)
        print(f"mounted {len(mcp_tools)} tools from MCP server: {spec}", file=sys.stderr)

    skills = discover_skills(args.skills or [])
    if skills:
        print(f"loaded skill index: {', '.join(s.name for s in skills)}", file=sys.stderr)

    from agent_harness.hitl import ApprovalPolicy

    policy = ApprovalPolicy.default()
    trace_path = Path(args.trace or f"traces/harness-{int(time.time())}.jsonl")
    tracer_provider = init_tracing(trace_path)
    provider = _build_provider(args)
    agent = Agent(
        provider,
        tools,
        approver=_build_approver(args.approve, policy),
        policy=policy,
        skills=skills,
        tracer_provider=tracer_provider,
        max_steps=args.max_steps,
    )
    try:
        result = agent.run(args.task)
    finally:
        tracer_provider.force_flush()
        for c in mcp_clients:
            c.close()

    record = record_run(args.runs_db, result)
    print(f"\n{result.answer}\n")
    print(
        f"run {result.trace_id[:8]} · {result.steps} steps · "
        f"{result.usage.total} tokens · ${result.cost_usd:.4f} · "
        f"{len(result.approvals)} approvals · {result.stopped_reason}",
        file=sys.stderr,
    )
    print(f"trace: {trace_path}  (render: `harness trace render {trace_path}`)", file=sys.stderr)
    return 0 if result.stopped_reason == "completed" else 1


def cmd_trace_render(args: argparse.Namespace) -> int:
    print(render_file(args.file, trace_id=args.trace_id))
    return 0


def cmd_runs_list(args: argparse.Namespace) -> int:
    runs = list_runs(args.runs_db, limit=args.limit)
    if not runs:
        print("(no runs recorded yet)")
        return 0
    for r in runs:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["finished_unix"]))
        print(
            f"{r['trace_id'][:8]}  {ts}  {r['steps']:>2} steps  "
            f"${r['cost_usd']:.4f}  {r['stopped_reason']:<10}  {r['task'][:60]}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent on a task")
    run.add_argument("task", nargs="?", default=DEMO_TASK)
    run.add_argument("--provider", choices=["scripted", "anthropic"], default="scripted")
    run.add_argument("--model", default="claude-sonnet-4-5-20250929")
    run.add_argument("--root", default=".", help="workspace root the tools are sandboxed to")
    run.add_argument("--approve", choices=["cli", "auto", "deny"], default="cli")
    run.add_argument("--trace", help="trace output path (default: traces/harness-<ts>.jsonl)")
    run.add_argument("--skills", action="append", help="directory containing <name>/SKILL.md skills")
    run.add_argument("--mcp", action="append", help="MCP server command to mount, e.g. 'uvx mcp-trace'")
    run.add_argument("--max-steps", type=int, default=12)
    run.add_argument("--runs-db", default="runs.jsonl")
    run.set_defaults(fn=cmd_run)

    tr = sub.add_parser("trace", help="trace utilities")
    trsub = tr.add_subparsers(dest="trace_command", required=True)
    render = trsub.add_parser("render", help="render a JSONL trace as a timeline")
    render.add_argument("file")
    render.add_argument("--trace-id")
    render.set_defaults(fn=cmd_trace_render)

    runs = sub.add_parser("runs", help="run store utilities")
    runsub = runs.add_subparsers(dest="runs_command", required=True)
    ls = runsub.add_parser("list")
    ls.add_argument("--runs-db", default="runs.jsonl")
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(fn=cmd_runs_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
