"""Skills: progressive-disclosure loading of SKILL.md directories.

A skill is a directory containing SKILL.md with frontmatter:

    ---
    name: pdf-processing
    description: Extract and transform PDFs. Use when the task mentions PDFs.
    ---

    # Body: instructions loaded only when the skill is selected.

Progressive disclosure:
1. The agent always sees the *index* (name + description) — cheap.
2. The full body is injected only when the agent (or user) selects the skill.

Frontmatter parsing is a deliberately small subset of YAML (flat `key: value`
pairs) to keep the harness dependency-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path

    def load_body(self) -> str:
        _, body = parse_skill_md((self.path / "SKILL.md").read_text(encoding="utf-8"))
        return body


def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with a '---' frontmatter block")
    meta: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i]
        if line.strip() and ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
        i += 1
    if i >= len(lines):
        raise ValueError("unterminated frontmatter block")
    return meta, "\n".join(lines[i + 1 :]).strip()


def discover_skills(dirs: Iterable[str | Path]) -> list[Skill]:
    """Find every ``<dir>/<skill-name>/SKILL.md`` across the given directories."""
    skills: list[Skill] = []
    seen: set[str] = set()
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            continue
        for md in sorted(root.glob("*/SKILL.md")):
            try:
                meta, _ = parse_skill_md(md.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            name = meta.get("name", md.parent.name)
            if name in seen:
                continue  # first directory wins (like PATH precedence)
            seen.add(name)
            skills.append(Skill(name=name, description=meta.get("description", ""), path=md.parent))
    return skills


def render_index(skills: list[Skill]) -> str:
    """The always-visible index injected into the system prompt."""
    if not skills:
        return ""
    lines = ["Available skills (call load_skill_body-equivalent by asking to 'use <name>'):"]
    for s in skills:
        lines.append(f"- {s.name}: {s.description}")
    return "\n".join(lines)
