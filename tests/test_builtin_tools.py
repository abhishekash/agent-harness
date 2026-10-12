import pytest

from agent_harness.tools import ListDir, ReadFile, RunShell, WriteFile, default_tools


def test_read_file(workspace):
    assert ReadFile(workspace).run(path="notes.md") == "# notes\nship the harness\n"


def test_read_missing_file(workspace):
    assert "error" in ReadFile(workspace).run(path="nope.md")


def test_read_file_blocks_escape(workspace):
    out = ReadFile(workspace).run(path="../outside.txt")
    assert "escapes workspace root" in out


def test_list_dir_recursive_skips_dotfiles(workspace):
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("x")
    out = ListDir(workspace).run()
    assert "src/main.py" in out
    assert ".git" not in out


def test_write_file_creates_parents(workspace):
    WriteFile(workspace).run(path="deep/nested/out.txt", content="data")
    assert (workspace / "deep" / "nested" / "out.txt").read_text() == "data"


def test_write_file_blocks_escape(workspace):
    out = WriteFile(workspace).run(path="../escape.txt", content="x")
    assert "escapes workspace root" in out


def test_shell_allowlist(workspace):
    out = RunShell(workspace).run(command="cat notes.md")
    assert "ship the harness" in out
    blocked = RunShell(workspace).run(command="rm -rf .")
    assert "not allowlisted" in blocked


def test_shell_timeout_and_parsing(workspace):
    assert "error" in RunShell(workspace).run(command='echo "unterminated')


def test_shell_blocks_paths_outside_workspace_and_find_exec(workspace):
    outside = RunShell(workspace).run(command="cat ../outside.txt")
    dangerous = RunShell(workspace).run(command="find . -exec cat {} \\;")
    assert "escapes workspace root" in outside
    assert "recursive find execution" in dangerous


def test_shell_rejects_symlink_escape_and_find_output_action(workspace, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside secret")
    (workspace / "linked.txt").symlink_to(outside)
    shell = RunShell(workspace)

    assert "escapes workspace root" in shell.run(command="cat linked.txt")
    assert "outside secret" not in shell.run(command="grep secret linked.txt")
    assert "unsupported find argument" in shell.run(command="find . -fprint output.txt")
    assert not (workspace / "output.txt").exists()
    assert "ship the harness" in shell.run(command="grep -rn ship .")


def test_shell_uses_system_binary_even_when_path_points_at_workspace(workspace, monkeypatch):
    fake_cat = workspace / "cat"
    fake_cat.write_text("#!/bin/sh\necho hijacked > marker.txt\n")
    fake_cat.chmod(0o755)
    monkeypatch.setenv("PATH", str(workspace))

    assert "ship the harness" in RunShell(workspace).run(command="cat notes.md")
    assert not (workspace / "marker.txt").exists()


def test_default_tools_cover_risk_tiers(workspace):
    risks = {t.name: t.risk.value for t in default_tools(workspace)}
    assert risks == {
        "read_file": "read",
        "list_dir": "read",
        "write_file": "write",
        "run_shell": "execute",
    }
