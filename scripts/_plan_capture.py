"""Capture one representative run for genai.plan.PLAN_DEMO.

We want the transcript instance that tells the three-act story cleanly: an initial
TOC valid under the six shown constraints, an in-place update that folds in the
held keystone but regresses on exactly one shown edge (the whack-a-mole), and a
checker pass that returns the broken edge and lands a fully valid plan. Hits live
gemma4, so it is nondeterministic; run until it prints a clean candidate, then
paste the dict into genai/plan.py.

Run: .venv/bin/python scripts/_plan_capture.py
"""
import sys
sys.path.insert(0, "chapters")
from genai.planning import (make_toc, update_toc, check_loop, parse_order,
                        violations, CHAPTERS, SHOWN, HELD, CONSTRAINTS, KEY)


def _complete(o):
    """All chapters present, none dropped in the reorder."""
    return sorted(o) == sorted(CHAPTERS)


def _sane(o):
    """A believable foundation order, so the draft reads like a real TOC and the
    only flaw on show is the constraint regression."""
    p = {c: i for i, c in enumerate(o)}
    return p["Foundations"] <= 1 and p["Tokens"] <= 3 and p["Semantics"] <= 4


def trial():
    init = parse_order(make_toc())
    if not (_complete(init) and _sane(init) and not violations(init, SHOWN)):
        return None
    upd = parse_order(update_toc("\n".join(init)))
    held_ok = KEY not in violations(upd, HELD)        # folded in the new constraint
    broke = violations(upd, SHOWN)                     # but broke ones it had
    if not (_complete(upd) and held_ok and len(broke) == 1):
        return None
    fixed, rounds = check_loop(upd)
    if not (_complete(fixed) and not violations(fixed, CONSTRAINTS)):
        return None
    return init, upd, broke[0], fixed, rounds


if __name__ == "__main__":
    for i in range(40):
        got = trial()
        print(f"trial {i+1}: {'candidate!' if got else 'skip'}", flush=True)
        if got:
            init, upd, broke, fixed, rounds = got
            print("\nPLAN_DEMO = {")
            print(f'    "initial": {init},')
            print(f'    "updated": {upd},')
            print(f'    "broke":   {broke},')
            print(f'    "fixed":   {fixed},')
            print(f"}}   # checker fixed in {rounds} round(s)")
            break
