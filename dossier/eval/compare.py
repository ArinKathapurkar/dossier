"""`dossier eval compare` -- run a tier under two prompt sets and diff the metrics.

This is the prompt-iteration loop. Prompts are content-hashed files (obs/prompts.py) and
git history is their version history, so "compare v1 against v2" means "run the same tier
with the prompt directory pointed at two different git refs".

A ref is materialised into a temp directory with `git show <ref>:prompts/<file>`, so the
comparison works against any commit without checking anything out. A plain directory path
also works, which is how an uncommitted draft gets measured before it is committed.

Tier 1 does not read prompts at all -- retrieval has no prompt -- so comparing it is a
useful control: any difference it shows is noise, and that tells you how much of a Tier 3
difference is signal.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, get_config
from ..obs.prompts import PromptRegistry, registry


def materialize_prompts(ref_or_dir: str) -> tuple[Path, bool]:
    """Return `(directory, is_temporary)` holding the prompt set for `ref_or_dir`."""
    candidate = Path(ref_or_dir)
    if candidate.is_dir():
        return candidate, False
    tmp = Path(tempfile.mkdtemp(prefix="dossier_prompts_"))
    listing = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{ref_or_dir}:prompts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"{ref_or_dir!r} is neither a directory nor a git ref with a prompts/ tree")
    for name in listing.stdout.split():
        blob = subprocess.run(
            ["git", "show", f"{ref_or_dir}:prompts/{name}"], cwd=REPO_ROOT, capture_output=True, text=True
        )
        if blob.returncode == 0:
            (tmp / name).write_text(blob.stdout)
    return tmp, True


def _with_prompts(directory: Path, fn, *args, **kwargs):
    """Run `fn` with the prompt registry pointed at `directory`."""
    import dossier.obs.prompts as prompts_mod

    original = prompts_mod.registry
    reg = PromptRegistry(directory)

    def patched(_directory: str | None = None) -> PromptRegistry:
        return reg

    prompts_mod.registry = patched
    # Modules that imported `registry` by name need rebinding too.
    rebound = []
    for mod_name in (
        "dossier.agent.loop",
        "dossier.agent.subagents",
        "dossier.guard.input_guard",
        "dossier.index.extract_entities",
        "dossier.eval.tier3_judge",
    ):
        import importlib

        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "registry"):
            rebound.append((mod, mod.registry))
            mod.registry = patched
    try:
        return fn(*args, **kwargs)
    finally:
        prompts_mod.registry = original
        for mod, old in rebound:
            mod.registry = old


def _metrics_for(tier: int, report: dict) -> dict[str, float]:
    if tier == 1:
        out: dict[str, float] = {"evidence_match_rate": report["evidence_match_rate"]}
        for name, m in report["configs"].items():
            for metric in ("recall@5", "recall@10", "ndcg@10", "mrr"):
                out[f"{name} {metric}"] = m[metric]
        return out
    if tier == 2:
        return {k: v for k, v in report["metrics"].items() if v is not None}
    return {
        "correct": report["overall"]["correct"],
        "partially_correct": report["overall"]["partially_correct"],
        "incorrect": report["overall"]["incorrect"],
        "abstained": report["overall"]["abstained"],
        "revise_rate": report["revise_rate"],
        "hitl_rate": report["hitl_rate"],
        "mean_cost_usd": report["mean_cost_usd"],
        "mean_latency_s": report["mean_latency_s"],
    }


def _run_tier(tier: int, limit: int | None, mini: bool) -> dict:
    if tier == 1:
        from .tier1_retrieval import run_tier1

        return run_tier1(limit=limit, mini=mini)
    if tier == 2:
        from .tier2_grounding import run_tier2

        return run_tier2()
    from .tier3_judge import run_tier3

    return run_tier3(limit=limit or 20)


def run_compare(tier: int, a: str, b: str, limit: int | None = None, mini: bool = False) -> dict:
    results: dict[str, Any] = {}
    versions: dict[str, dict] = {}
    temps: list[Path] = []
    try:
        for label, ref in (("a", a), ("b", b)):
            directory, is_temp = materialize_prompts(ref)
            if is_temp:
                temps.append(directory)
            versions[label] = PromptRegistry(directory).versions()
            results[label] = _with_prompts(directory, _run_tier, tier, limit, mini)
    finally:
        for t in temps:
            shutil.rmtree(t, ignore_errors=True)

    ma, mb = _metrics_for(tier, results["a"]), _metrics_for(tier, results["b"])
    rows = []
    for metric in sorted(set(ma) | set(mb)):
        va, vb = ma.get(metric), mb.get(metric)
        if va is None or vb is None:
            continue
        rows.append({"metric": metric, "a": round(va, 4), "b": round(vb, 4), "delta": round(vb - va, 4)})
    changed_prompts = sorted(
        name for name in set(versions["a"]) | set(versions["b"]) if versions["a"].get(name) != versions["b"].get(name)
    )
    return {
        "tier": tier,
        "a": a,
        "b": b,
        "limit": limit,
        "prompt_versions": versions,
        "changed_prompts": changed_prompts,
        "rows": rows,
        "reports": {"a": results["a"], "b": results["b"]},
    }


def current_versions() -> dict[str, str]:
    _ = get_config()
    return registry().versions()
