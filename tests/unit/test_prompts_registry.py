"""Prompt registry: content-hash versions, placeholder rendering, no hidden state."""

import pytest

from dossier.config import REPO_ROOT
from dossier.obs.prompts import PromptRegistry, registry

EXPECTED = {
    "system_analyst", "section_business", "section_financial", "section_competitive",
    "section_risks", "section_questions", "synthesizer", "entity_extraction",
    "input_guard", "judge",
}


def test_every_prompt_the_code_references_exists():
    assert EXPECTED.issubset(set(registry().names()))


def test_version_is_a_short_content_hash(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("hello")
    reg = PromptRegistry(tmp_path)
    _, version = reg.get("a")
    assert len(version) == 8 and all(c in "0123456789abcdef" for c in version)


def test_editing_a_prompt_changes_its_version(tmp_path):
    (tmp_path / "a.md").write_text("v1")
    assert PromptRegistry(tmp_path).version("a") != (
        (tmp_path / "a.md").write_text("v2") or PromptRegistry(tmp_path).version("a")
    )


def test_identical_content_hashes_identically(tmp_path):
    (tmp_path / "a.md").write_text("same")
    (tmp_path / "b.md").write_text("same")
    reg = PromptRegistry(tmp_path)
    assert reg.version("a") == reg.version("b")


def test_missing_prompt_raises_rather_than_returning_empty():
    with pytest.raises(FileNotFoundError):
        registry().get("no_such_prompt")


def test_render_substitutes_placeholders(tmp_path):
    (tmp_path / "s.md").write_text("Draft for {{target}} against {{peers}}.")
    text, _ = PromptRegistry(tmp_path).render("s", target="3M", peers="Amcor, Corning")
    assert text == "Draft for 3M against Amcor, Corning."


def test_section_prompts_declare_the_placeholders_they_use():
    reg = registry()
    for name in ("section_business", "section_financial", "section_competitive", "section_risks", "section_questions"):
        assert "{{target}}" in reg.text(name), f"{name} must be parameterised on the target"


def test_all_prompt_files_are_committed_under_prompts():
    assert (REPO_ROOT / "prompts").is_dir()
    assert len(list((REPO_ROOT / "prompts").glob("*.md"))) >= len(EXPECTED)
