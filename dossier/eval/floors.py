"""The CI regression gate.

`python -m dossier.eval.floors` runs Tier 1 on the committed mini corpus and fails the
build if `rrf(vector+bm25)` nDCG@10 drops below the committed floor in
`tests/fixtures/floors.json`.

The floor is set a little below the measured value at the time it was committed, so normal
variation in a chunking or fusion tweak does not trip it but a real regression does. It is
the only quality number CI enforces, on purpose: it is the one that is fully deterministic
and fully offline, so a red build means something specific rather than "a model felt
different today".

Raise the floor deliberately, in the same commit as the improvement that earns it.
"""

from __future__ import annotations

import json
import sys

from ..config import REPO_ROOT

FLOORS = REPO_ROOT / "tests" / "fixtures" / "floors.json"


def load_floors() -> dict:
    return json.loads(FLOORS.read_text()) if FLOORS.exists() else {}


def write_floors(payload: dict) -> None:
    FLOORS.parent.mkdir(parents=True, exist_ok=True)
    FLOORS.write_text(json.dumps(payload, indent=2))


def check() -> tuple[bool, list[str]]:
    from .tier1_retrieval import CONFIGS, run_tier1

    floors = load_floors()
    if not floors:
        return True, ["no floors committed yet -- gate is a no-op"]
    wanted = set(floors.get("configs", {}))
    rep = run_tier1(mini=True, configs=[c for c in CONFIGS if c[0] in wanted])
    messages: list[str] = []
    ok = True
    for name, thresholds in floors.get("configs", {}).items():
        measured = rep["configs"].get(name)
        if measured is None:
            messages.append(f"FAIL {name}: not measured on the mini corpus")
            ok = False
            continue
        for metric, floor in thresholds.items():
            value = measured.get(metric)
            if value is None:
                messages.append(f"FAIL {name} {metric}: not reported")
                ok = False
            elif value < floor:
                messages.append(f"FAIL {name} {metric}: {value:.4f} < floor {floor:.4f}")
                ok = False
            else:
                messages.append(f"ok   {name} {metric}: {value:.4f} >= floor {floor:.4f}")
    emr_floor = floors.get("evidence_match_rate")
    if emr_floor is not None:
        value = rep["evidence_match_rate"]
        status = "ok  " if value >= emr_floor else "FAIL"
        ok = ok and value >= emr_floor
        messages.append(f"{status} evidence_match_rate: {value:.4f} >= floor {emr_floor:.4f}")
    return ok, messages


def main() -> None:
    ok, messages = check()
    for m in messages:
        print(m)
    if not ok:
        print("\nTier 1 regression gate FAILED. If this drop is intentional, update tests/fixtures/floors.json in the same commit.")
        sys.exit(1)
    print("\nTier 1 regression gate passed.")


if __name__ == "__main__":
    main()
