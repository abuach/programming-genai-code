"""Pick + capture the lfm2 trace for show_reflexion_trace (lfm2 plays Sophia after
the 14-model bake-off made it the biggest gainer). For each candidate gotcha task
it reports lfm2's first-try failure rate and captures one real fail -> fix
trajectory plus the deterministic temperature-0 lesson, so we can choose the task
with the most legible fix and a coherent lesson, then bake its two attempts.

Run: .venv/bin/python scripts/_reflexion_lfm2.py
"""
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "chapters")
from genai.agent import reflexion, code_test, _CODER, _extract_code
from genai.llm import ask as _ask
import _reflexion_proto as P
import _reflexion_gotcha  # noqa: F401  (sets P.SUITE = GOTCHA_SUITE)

MODEL = "lfm2:24b"
SUITE = {name: (spec, cases) for name, spec, cases in P.SUITE}
WANT = ["nightly_pace", "round_rating", "rounded_rating", "shelve_order"]


def temp0_lesson(spec, detail):
    raw = _ask(f"Task: {spec}\nThis attempt failed: {detail}\n"
               "In one short sentence, what will you change next time?",
               model=MODEL, system=_CODER, max_tokens=60,
               options={"temperature": 0}).strip()
    return raw.replace("\n", " ").split(". ")[0].rstrip(".") + "."


for name in WANT:
    spec, cases = SUITE[name]
    check = code_test(name, cases)
    print(f"\n==== {name} ====", flush=True)
    fails = 0
    for i in range(6):
        reply = _ask(f"{spec} Reply with only the function in one code block.",
                     model=MODEL, system=_CODER)
        ok, _ = check(_extract_code(reply))
        fails += not ok
    print(f"  first-try failed {fails}/6", flush=True)
    for r in range(12):
        trace = reflexion(spec, check, model=MODEL, max_trials=2)
        if len(trace) < 2:
            continue
        (c1, ok1, d1, _), (c2, ok2, _, _) = trace
        if (not ok1) and ok2:
            print("  TRY1 (fails):\n" + c1)
            print("  detail:", d1)
            print("  TRY2 (passes):\n" + c2)
            print("  temp0 lesson:", temp0_lesson(spec, d1), flush=True)
            break
    else:
        print("  (no clean fail->fix captured in 12 tries)", flush=True)
