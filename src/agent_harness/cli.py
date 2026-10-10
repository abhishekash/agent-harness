"""`harness` command line.

    harness run "task"        run the agent (scripted, Anthropic, or OpenCode)
    harness trace render F    render a JSONL trace file as a timeline
    harness runs list         show the local run store
"""
from __future__ import annotations

import argparse
import os
import signal
import shlex
import sys
import time
from pathlib import Path

from agent_harness.agent import Agent
from agent_harness.checkpoint import CheckpointError, CheckpointStore
from agent_harness.context import ContextManager, ContextPolicy
from agent_harness.hitl import AutoApprover, CLIApprover, DenyAllApprover
from agent_harness.providers import (
    ResilientProvider,
    RetryPolicy,
    call_tool,
    say,
    scripted_run,
)
from agent_harness.skills import discover_skills
from agent_harness.store import list_runs, record_run
from agent_harness.summary import RollingSummary
from agent_harness.tools import default_tools
from agent_harness.tracing import init_tracing, render_file
from agent_harness.types import CancellationToken, Risk, RunLimits
from agent_harness.ui import MinimalTerminalUI

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

        model = args.model or "claude-sonnet-4-5-20250929"
        kwargs = {"timeout": args.provider_timeout} if args.provider_timeout else {}
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        return AnthropicProvider(model=model, **kwargs)
    if args.provider == "opencode":
        from agent_harness.providers.opencode import OpenCodeProvider

        return OpenCodeProvider(
            model=args.model or os.getenv("OPENCODE_MODEL", "big-pickle"),
            base_url=args.base_url,
            api_key=os.getenv(args.api_key_env),
            temperature=args.temperature,
            timeout=args.provider_timeout,
        )
    raise SystemExit(f"unknown provider: {args.provider}")


def _build_approver(name: str, policy):
    if name == "auto":
        return AutoApprover()
    if name == "deny":
        return DenyAllApprover()
    return CLIApprover(policy=policy)


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    task = args.task or DEMO_TASK
    tools = default_tools(root)
    mcp_clients = []
    try:
        for spec in args.mcp or []:
            from agent_harness.mcp import mount_server

            client, mcp_tools = mount_server(shlex.split(spec), timeout=args.mcp_timeout)
            mcp_clients.append(client)
            tools.extend(mcp_tools)
            print(f"mounted {len(mcp_tools)} tools from MCP server: {spec}", file=sys.stderr)
    except Exception as exc:
        for client in mcp_clients:
            client.close()
        print(f"error: could not mount MCP server: {exc}", file=sys.stderr)
        return 2

    skills = discover_skills(args.skills or [])
    if skills:
        print(f"loaded skill index: {', '.join(s.name for s in skills)}", file=sys.stderr)

    from agent_harness.hitl import ApprovalPolicy

    policy = ApprovalPolicy.default()
    trace_path = Path(args.trace or f"traces/harness-{int(time.time())}.jsonl")
    checkpoint_path = Path(
        args.resume
        or args.checkpoint
        or f"{trace_path}.checkpoint.json"
    )
    checkpoint_store = CheckpointStore(checkpoint_path)
    if args.resume and args.task is None:
        try:
            task = checkpoint_store.load().task
        except CheckpointError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    tracer_provider = init_tracing(trace_path)
    cancellation = CancellationToken()
    provider = ResilientProvider(
        _build_provider(args),
        policy=RetryPolicy(max_attempts=args.retries + 1),
        cancellation=cancellation,
    )
    ui = MinimalTerminalUI(sys.stderr, mode=args.ui)
    progress = RollingSummary(provider) if ui.enabled else None
    initial_summary = progress.begin(task) if progress is not None else ""
    ui.start(task, getattr(provider, "model", "unknown"), initial_summary)
    context_manager = (
        ContextManager(provider, ContextPolicy(max_chars=args.context_chars))
        if args.context_chars > 0
        else None
    )
    agent = Agent(
        provider,
        tools,
        approver=_build_approver(args.approve, policy),
        policy=policy,
        skills=skills,
        tracer_provider=tracer_provider,
        max_steps=args.max_steps,
        event_handler=ui.handle_event if ui.enabled else None,
        progress_summarizer=progress,
        limits=RunLimits(
            max_steps=args.max_steps,
            max_duration_s=args.max_duration,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            max_cost_usd=args.max_cost,
        ),
        cancellation=cancellation,
        context_manager=context_manager,
        checkpoint_store=checkpoint_store,
        resume=bool(args.resume),
    )
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _cancel(_signum, _frame):
        cancellation.cancel("interrupted")
        print("\ninterrupt received; stopping after the current safe boundary...", file=sys.stderr)

    signal.signal(signal.SIGINT, _cancel)
    result = None
    checkpoint_error = None
    try:
        result = agent.run(task)
    except CheckpointError as exc:
        checkpoint_error = str(exc)
        print(f"error: {checkpoint_error}", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        tracer_provider.force_flush()
        for c in mcp_clients:
            c.close()
        if result is not None:
            ui.finish(result.summary, result.stopped_reason)
        else:
            ui.close()

    if result is None:
        return 2 if checkpoint_error else 1

    record = record_run(args.runs_db, result)
    print(f"\n{result.answer}\n")
    print(
        f"run {result.trace_id[:8]} · {result.steps} steps · "
        f"{result.usage.total} tokens · ${result.cost_usd:.4f} · "
        f"{len(result.approvals)} approvals · {result.stopped_reason}",
        file=sys.stderr,
    )
    print(f"trace: {trace_path}  (render: `harness trace render {trace_path}`)", file=sys.stderr)
    if result.stopped_reason != "completed":
        print(f"checkpoint: {checkpoint_path}  (resume with `--resume {checkpoint_path}`)", file=sys.stderr)
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
    run.add_argument("task", nargs="?", default=None)
    run.add_argument("--provider", choices=["scripted", "anthropic", "opencode"], default="scripted")
    run.add_argument("--model", help="provider model (defaults to the selected provider's model)")
    run.add_argument("--base-url", help="OpenCode-compatible API base URL")
    run.add_argument("--api-key-env", default="OPENCODE_API_KEY", help="environment variable containing the OpenCode key")
    run.add_argument("--temperature", type=float, default=None)
    run.add_argument("--provider-timeout", type=float, default=60.0, help="provider request timeout in seconds")
    run.add_argument("--retries", type=int, default=2, help="transient provider retries (default: 2)")
    run.add_argument("--root", default=".", help="workspace root the tools are sandboxed to")
    run.add_argument("--approve", choices=["cli", "auto", "deny"], default="cli")
    run.add_argument("--trace", help="trace output path (default: traces/harness-<ts>.jsonl)")
    run.add_argument("--skills", action="append", help="directory containing <name>/SKILL.md skills")
    run.add_argument("--mcp", action="append", help="MCP server command to mount, e.g. 'uvx mcp-trace'")
    run.add_argument("--mcp-timeout", type=float, default=30.0, help="MCP request timeout in seconds")
    run.add_argument(
        "--ui",
        choices=["auto", "always", "never"],
        default="auto",
        help="minimal live terminal UI (default: auto; enabled on a TTY)",
    )
    run.add_argument("--max-steps", type=int, default=12)
    run.add_argument("--max-duration", type=float, help="hard run duration limit in seconds")
    run.add_argument("--max-input-tokens", type=int, help="hard input-token budget")
    run.add_argument("--max-output-tokens", type=int, help="hard output-token budget")
    run.add_argument("--max-cost", type=float, help="hard estimated USD cost budget")
    run.add_argument(
        "--context-chars",
        type=int,
        default=80_000,
        help="approximate conversation budget before model compaction; 0 disables",
    )
    run.add_argument("--checkpoint", help="checkpoint path (default: alongside trace)")
    run.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="resume a safe checkpoint; in-flight side effects are never replayed",
    )
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
