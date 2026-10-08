# Contributing to agent-harness

## Development setup

```bash
uv venv
uv pip install -e ".[dev]"
pytest -q
```

The test suite is offline by design. Use `ScriptedProvider` for deterministic loop tests; do not add network calls or real API keys to CI.

## Scope

Good contributions are small, composable changes to the loop, providers, tools, HITL policy, tracing, skills loader, or MCP client. Every new side-effecting tool must declare a `Risk` tier and have sandbox/error-path tests.

## Pull requests

- Explain the behavioral contract and the trace shape it produces.
- Add or update tests, including denial/edit paths for HITL changes.
- Update the README when CLI behavior or limitations change.
- Keep commits focused; generated traces and local run stores stay ignored.
