"""Prompt registry.

Prompts live in `prompts/*.md` as plain files, and their version is the first 8 hex chars
of the sha256 of their content. That means:

  * every llm span records `prompt_name@version`, so a metric can be attributed to the
    exact text that produced it;
  * `dossier eval compare` can run the same tier under two prompt directories or two git
    refs and diff the results;
  * git history is the version history -- there is no second store to keep in sync.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from ..config import get_config


class PromptRegistry:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory) if directory else get_config().paths.prompts
        self._cache: dict[str, tuple[str, str]] = {}

    def path_for(self, name: str) -> Path:
        return self.dir / (name if name.endswith(".md") else f"{name}.md")

    def get(self, name: str) -> tuple[str, str]:
        """Return `(text, version)`."""
        if name in self._cache:
            return self._cache[name]
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"prompt {name!r} not found at {path}")
        text = path.read_text()
        version = hashlib.sha256(text.encode()).hexdigest()[:8]
        self._cache[name] = (text, version)
        return text, version

    def text(self, name: str) -> str:
        return self.get(name)[0]

    def version(self, name: str) -> str:
        return self.get(name)[1]

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.md"))

    def versions(self) -> dict[str, str]:
        return {n: self.version(n) for n in self.names()}

    def render(self, name: str, **kwargs) -> tuple[str, str]:
        """Substitute `{{placeholders}}`. Deliberately not Jinja: prompts must stay
        readable as prose and diffable as prose."""
        text, version = self.get(name)
        for key, value in kwargs.items():
            text = text.replace("{{" + key + "}}", "" if value is None else str(value))
        return text, version


@lru_cache(maxsize=8)
def registry(directory: str | None = None) -> PromptRegistry:
    return PromptRegistry(Path(directory) if directory else None)
