import json

from agent_harness.cli import main


def test_cli_run_scripted(tmp_path, capsys):
    notes = tmp_path / "notes.md"
    notes.write_text("# notes\ncli smoke test\n")
    code = main(
        [
            "run",
            "--provider",
            "scripted",
            "--approve",
            "auto",
            "--root",
            str(tmp_path),
            "--trace",
            str(tmp_path / "t.jsonl"),
            "--runs-db",
            str(tmp_path / "runs.jsonl"),
        ]
    )
    assert code == 0
    assert (tmp_path / "SUMMARY.md").exists()
    err = capsys.readouterr().err
    assert "approvals" in err

    runs = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 1
    assert runs[0]["stopped_reason"] == "completed"


def test_cli_trace_render(tmp_path, capsys):
    (tmp_path / "notes.md").write_text("x\n")
    main(
        [
            "run",
            "--provider",
            "scripted",
            "--approve",
            "auto",
            "--root",
            str(tmp_path),
            "--trace",
            str(tmp_path / "t.jsonl"),
            "--runs-db",
            str(tmp_path / "runs.jsonl"),
        ]
    )
    capsys.readouterr()
    assert main(["trace", "render", str(tmp_path / "t.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "agent.run" in out
    assert "🔐 human" in out


def test_cli_runs_list_empty(tmp_path, capsys):
    assert main(["runs", "list", "--runs-db", str(tmp_path / "none.jsonl")]) == 0
    assert "no runs" in capsys.readouterr().out
