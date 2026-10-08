import pytest

from agent_harness.skills import discover_skills, parse_skill_md, render_index

SKILL = """---
name: pdf-processing
description: Extract and transform PDFs.
---

# PDF processing

Use pypdf. Never shell out to pdftk.
"""


def test_parse_frontmatter():
    meta, body = parse_skill_md(SKILL)
    assert meta == {"name": "pdf-processing", "description": "Extract and transform PDFs."}
    assert "Never shell out" in body


def test_parse_requires_frontmatter():
    with pytest.raises(ValueError):
        parse_skill_md("# no frontmatter")


def test_discovery_and_precedence(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d, desc in ((a, "first wins"), (b, "second loses")):
        (d / "pdf-processing").mkdir(parents=True)
        (d / "pdf-processing" / "SKILL.md").write_text(
            SKILL.replace("Extract and transform PDFs.", desc)
        )
    skills = discover_skills([a, b])
    assert len(skills) == 1
    assert skills[0].description == "first wins"
    assert "Use pypdf" in skills[0].load_body()


def test_render_index(tmp_path):
    d = tmp_path / "skills"
    (d / "pdf-processing").mkdir(parents=True)
    (d / "pdf-processing" / "SKILL.md").write_text(SKILL)
    index = render_index(discover_skills([d]))
    assert "pdf-processing: Extract and transform PDFs." in index
    assert render_index([]) == ""
