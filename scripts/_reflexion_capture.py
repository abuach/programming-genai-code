"""Reproducer for the two REAL lfm2 attempts baked into show_reflexion_trace
(genai.agent._TRY1 / _TRY2) and the deterministic lesson between them.

The agentic.ipynb Reflexion trace (cell age-reflexion-trace) shows one
fail -> lesson -> pass cycle on the ``nightly_pace`` task. lfm2 plays Sophia here
because the 14-model bake-off (scripts/_reflexion_bakeoff.py) made it the biggest
gainer from reflection. The two attempts are real lfm2:24b output captured here and
baked verbatim (the way the Voyager skill code is baked), so the demo obeys the
model-output-honesty rule: a speaker-labelled row must carry the model's real
output, never the author's paraphrase.

What this script documents:
  1. lfm2 GENUINELY fails ``nightly_pace`` on the first try (sum over length, no
     empty-list guard -> ZeroDivisionError) under the reflexion prompt, about two
     thirds of the time, so the manufactured failure is faithful, not fictional.
  2. After its own reflection it writes a passing fix that guards the empty list.
  3. The lesson is deterministic at temperature 0, so the live REFLECT line in the
     frozen cell reproduces exactly on a re-run.

(round_rating was also evaluated as a substrate -- lfm2 fails its half-up rounding
6/6 -- but its over-eager fix reads as a ternary that muddies the trace, so
nightly_pace wins on legibility. See scripts/_reflexion_lfm2.py for that screen.)

Run: .venv/bin/python scripts/_reflexion_capture.py
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import (reflexion, code_test, REFLEXION_MODEL, _CODER,
                         _extract_code, REFLEXION_TASK, _TRY1, _TRY2)
from genai.llm import ask as _ask

NAME, SPEC, CASES = REFLEXION_TASK
CHECK = code_test(NAME, CASES)


def temp0_lesson(detail):
    """The deterministic lesson show_reflexion_trace computes live (temperature 0,
    max_tokens 60, first sentence only)."""
    raw = _ask(f"Task: {SPEC}\nThis attempt failed: {detail}\n"
               "In one short sentence, what will you change next time?",
               model=REFLEXION_MODEL, system=_CODER, max_tokens=60,
               options={"temperature": 0}).strip()
    return raw.replace("\n", " ").split(". ")[0].rstrip(".") + "."


def first_try_failure_rate(n=8):
    """lfm2 really fails nightly_pace on the first try -- show the rate and a sample
    of the verbatim first attempts (the obvious sum-over-length)."""
    print(f"=== first-attempt behaviour on '{NAME}' (n={n}) ===")
    fails = 0
    for i in range(n):
        reply = _ask(f"{SPEC} Reply with only the function in one code block.",
                     model=REFLEXION_MODEL, system=_CODER)
        ok, detail = CHECK(_extract_code(reply))
        fails += not ok
        print(f"  [{i}] {'PASS' if ok else 'FAIL'}  {detail}")
    print(f"  >> failed first try {fails}/{n}\n")


def capture_fail_fix(tries=15):
    """Capture a real fail -> fix trajectory matching the baked pair: try1 the
    obvious mean that divides by zero, try2 the guarded fix that passes."""
    print(f"=== real fail->fix trajectories (up to {tries}) ===")
    for r in range(tries):
        trace = reflexion(SPEC, CHECK, model=REFLEXION_MODEL, max_trials=2)
        if len(trace) < 2:
            continue
        (c1, ok1, d1, _), (c2, ok2, _, _) = trace
        if (not ok1) and ok2:
            print("  TRY1 (fails):", repr(c1))
            print("  TRY2 (passes):", repr(c2))
            print("  temp0 lesson:", temp0_lesson(d1), "\n")
            return


def verify_baked():
    """The exact strings baked in genai.agent must still fail / pass as labelled."""
    print("=== verify baked _TRY1 / _TRY2 against code_test ===")
    print("  _TRY1:", CHECK(_TRY1))
    print("  _TRY2:", CHECK(_TRY2))


if __name__ == "__main__":
    verify_baked()
    print()
    first_try_failure_rate()
    capture_fail_fix()
