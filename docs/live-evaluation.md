# Live evaluation guide

Use a live provider to examine how a model behaves with the harness's tools and
approval policy. The published baseline is scripted; this guide describes how
to collect a separate live result.

## Run the edited-write evaluation

The evaluation command belongs to the companion
[agent-evals repository](https://github.com/abhishekash/agent-evals). Install it
and run from that checkout so its task files are available:

```bash
git clone https://github.com/abhishekash/agent-evals.git
cd agent-evals
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,opencode]"

# Set OPENCODE_API_KEY in your environment before running.
export OPENCODE_MODEL="your-model-id"
agent-evals run --only edited-write \
  --provider opencode --model "$OPENCODE_MODEL" \
  --repeats 3 --out results/live-edited-write
```

Choose a model supported by your configured endpoint. See the
[provider configuration](https://github.com/abhishekash/agent-evals#quickstart)
for endpoint overrides. For models without built-in pricing, supply both
`--input-price-per-million` and `--output-price-per-million` using the rates
applicable to your provider, or report cost as unknown.

The `edited-write` task uses an automated callback to replace proposed tool
arguments with a predefined destination and content. In live mode the model
chooses its tool calls; the approval fixture still supplies the edit. This
checks behavior under that intervention. It does not measure a person's
judgment or guarantee the model will propose the unsafe path used in the
scripted fixture.

## Capture an interactive case study

For a demonstration of a person correcting an action, run `harness run` with a
live provider and `--approve cli` in a disposable workspace. Give it a concrete
file-writing task, inspect the proposed call, and use the edit option to change
the destination before execution. Record the proposal, replacement arguments,
decision, and resulting file contents.

The trace records the approval decision, rationale, and whether arguments were
edited. It does not include the original and replacement arguments in the
approval event; preserve those in the session recording or case-study notes.
Shell execution has path checks and an allowlist, but no OS-level isolation.

## Report the evidence

Publish the generated `RESULTS.md`, `results.json`, and trace artifacts with:

- Harness and evaluator commit IDs, model ID, endpoint, and sampling settings.
- Task, initial workspace, approval mode, and repetition count.
- Pass rate, each failed scorer, and any run errors.
- Latency, token usage, cost or unknown-cost status, and trace IDs.
- What the run demonstrates and which conclusions remain untested.

Keep failed attempts alongside successful ones. Three repetitions provide a
small case study; broader reliability claims need more tasks and trials.
Review artifacts for credentials and private task data before publishing.

[Back to the harness README](../README.md)
