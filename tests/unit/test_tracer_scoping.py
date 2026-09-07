"""Span-stack scoping, and the failure mode that took the memo run down.

The tracer keeps its run binding and its parent stack in thread-local storage so the five
parallel section sub-agents do not interleave their trees. That part worked. What did not
was the interaction between `bind` and an open span on the *same* thread: `bind` resets the
stack to empty, and the span's exit used to pop it unconditionally. A `bind` issued inside
an open span therefore raised `IndexError: pop from empty list` on the way out -- and the
one place that did it was `subagents.memo`, which re-bound the parent run after the
sub-agents returned, inside the `run` span wrapping the whole memo.

Nothing caught it because no test and no cassette ran the full five-section assembly; the
recorded cassettes exercise a single `memo_section` at a time, where no such re-bind
happens. The call is gone now, but observability must never be able to fail the work it is
observing, so the tracer is hardened too and these tests pin both halves.
"""

from __future__ import annotations

import threading

from dossier.obs.tracer import Tracer


def test_a_bind_inside_an_open_span_does_not_fail_the_traced_work(tmp_path):
    """The regression. A mis-scoped bind may distort the trace; it may not raise."""
    tracer = Tracer(tmp_path / "t.sqlite")
    tracer.bind("run_a")

    with tracer.span("run", "outer"):
        tracer.bind("run_a")  # what subagents.memo used to do
    # reaching here at all is the assertion

    kinds = {s["kind"] for s in tracer.spans_for("run_a")}
    assert "run" in kinds, "the span was lost as well as the exception"


def test_a_span_still_records_when_the_stack_is_reset_under_it(tmp_path):
    tracer = Tracer(tmp_path / "t.sqlite")
    tracer.bind("run_b")

    with tracer.span("run", "outer") as outer:
        with tracer.span("turn", "inner"):
            tracer.bind("run_b")
        outer.attrs["survived"] = True

    names = {s["name"] for s in tracer.spans_for("run_b")}
    assert {"outer", "inner"} <= names


def test_nesting_is_restored_after_a_reset_rather_than_left_corrupt(tmp_path):
    """After the disturbance, a fresh span must not inherit a stale parent."""
    tracer = Tracer(tmp_path / "t.sqlite")
    tracer.bind("run_c")

    with tracer.span("run", "outer"):
        tracer.bind("run_c")
    with tracer.span("run", "after"):
        pass

    after = [s for s in tracer.spans_for("run_c") if s["name"] == "after"][0]
    assert after["parent_id"] is None, "a later span inherited a parent from the reset stack"


def test_parallel_threads_keep_separate_trees(tmp_path):
    """The property the thread-local stack exists for, pinned so it cannot regress."""
    tracer = Tracer(tmp_path / "t.sqlite")
    barrier = threading.Barrier(3)

    def worker(run_id: str) -> None:
        tracer.bind(run_id)
        with tracer.span("run", f"{run_id}_root"):
            barrier.wait()  # force the spans to overlap in time
            with tracer.span("turn", f"{run_id}_child"):
                pass

    threads = [threading.Thread(target=worker, args=(f"run_{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(3):
        spans = {s["name"]: s for s in tracer.spans_for(f"run_{i}")}
        assert set(spans) == {f"run_{i}_root", f"run_{i}_child"}
        assert spans[f"run_{i}_root"]["parent_id"] is None
        assert spans[f"run_{i}_child"]["parent_id"] == spans[f"run_{i}_root"]["span_id"]
