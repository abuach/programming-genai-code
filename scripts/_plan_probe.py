"""THROWAWAY probe: a simple planning/replanning illustration on the book's TOC.

The plan IS the table of contents. Give Sophia ~80% of the chapter dependencies
and ask for a valid TOC. Then reveal the missing ~20% (one constraint we "forgot")
and ask her to update the TOC accordingly. The question is whether she folds the
new constraint in correctly -- without breaking the ones she already had.

Ground truth is the real dependency set (the strong backward cross-references the
chapters actually make). Held back: the keystone Augmentation -> Agentic
(search_book and Sophia's corpus are built in Augmentation), the one edge the
other six leave under-determined.

We measure, per run:
  - initial TOC valid under the 80% shown?      (the easy part)
  - did the initial TOC already satisfy the held edge, or does it need a fix?
  - updated TOC valid under ALL constraints?     (folded the new one in)
  - did the update BREAK any of the 80% it already had?  (whack-a-mole)

Run: .venv/bin/python scripts/_plan_probe.py [M] [model ...]
"""
import sys
sys.path.insert(0, "chapters")
from genai.llm import ask

# Presented to the model in neutral (alphabetical) order so the ordering isn't leaked.
CHAPTERS = sorted([
    "Foundations", "Prompting", "Tokens", "Semantics", "Metacoding",
    "Augmentation", "Agentic", "Autonomy", "Thinking", "Multimodal",
    "Design", "Efficiency", "Responsible",
])

# (prereq, dependent, why): prereq MUST come before dependent. The real strong
# backward references in the book. CONSTRAINTS[0] is the keystone, held back as
# the 20%; the rest are the 80% shown up front.
CONSTRAINTS = [
    ("Augmentation", "Agentic",     "Agentic uses the search_book tool and corpus built in Augmentation"),
    ("Agentic",      "Autonomy",    "Autonomy is the deployment half that continues Agentic"),
    ("Semantics",    "Augmentation","Augmentation's retrieval needs the embeddings from Semantics"),
    ("Semantics",    "Metacoding",  "Metacoding builds on the embeddings from Semantics"),
    ("Augmentation", "Design",      "Design reuses Augmentation's retrieval"),
    ("Agentic",      "Responsible", "Responsible stress-tests the agent from Agentic"),
    ("Autonomy",     "Responsible", "Responsible stress-tests the deployment from Autonomy"),
]
HELD  = CONSTRAINTS[:1]   # the 20% revealed at update time (the keystone)
SHOWN = CONSTRAINTS[1:]   # the 80% given up front (6 of 7 edges)

KEY = (HELD[0][0], HELD[0][1])  # ("Augmentation", "Agentic")


def block(edges):
    return "\n".join(f"- {p} must come before {d}  ({why})" for p, d, why in edges)


def parse_order(text):
    """Recover the ordering: text after the last 'ORDER:' marker, chapters by
    first appearance."""
    tail = text.rsplit("ORDER:", 1)[-1].lower()
    hits = [(tail.find(c.lower()), c) for c in CHAPTERS if c.lower() in tail]
    return [c for _, c in sorted(hits)]


def violations(order, edges):
    pos = {c: i for i, c in enumerate(order)}
    return [(p, d) for p, d, _ in edges
            if p in pos and d in pos and pos[p] > pos[d]]


CATALOG = "\n".join(f"- {c}" for c in CHAPTERS)
TAIL = ("\n\nEnd with a line 'ORDER:' followed by every chapter name in order, "
        "one per line, using exactly the names above.")
SOPHIA = ("You are Sophia, the companion for a generative-AI textbook. You plan "
          "the book's table of contents so every chapter comes after the chapters "
          "it builds on.")


def make_toc(model, temp):
    p = (f"The book has these chapters:\n{CATALOG}\n\n"
         f"Ordering constraints (each chapter must come after what it builds on):\n"
         f"{block(SHOWN)}\n\n"
         f"Produce a valid table of contents with every constraint satisfied.{TAIL}")
    return ask(p, model=model, system=SOPHIA, max_tokens=400,
               options={"temperature": temp})


def update_toc(prev, model, temp):
    p = (f"Here is the current table of contents:\n{prev}\n\n"
         f"One more constraint we left out earlier:\n{block(HELD)}\n\n"
         f"Update the table of contents so it respects this new constraint too, "
         f"while keeping every earlier constraint satisfied and changing as little "
         f"as possible.{TAIL}")
    return ask(p, model=model, system=SOPHIA, max_tokens=400,
               options={"temperature": temp})


def grade(label, text, edges):
    order = parse_order(text)
    v = violations(order, edges)
    tag = "clean" if not v else "broken: " + ", ".join(f"{p}>{d}" for p, d in v)
    print(f"  {label:14s} viol={len(v)}  {tag}")
    return order, v


def triage(model, temp=0.0):
    print(f"\n===== {model}  (triage, temp={temp}) =====", flush=True)
    toc = make_toc(model, temp)
    o0, _ = grade("initial/shown", toc, SHOWN)
    pre = "already satisfied" if KEY not in violations(o0, HELD) else "VIOLATED -> needs fix"
    print(f"  held edge {KEY[0]}->{KEY[1]} in initial TOC: {pre}")
    print("  initial TOC:", o0)
    upd = update_toc("\n".join(o0), model, temp)
    o1, _ = grade("updated/ALL", upd, CONSTRAINTS)
    print("  updated TOC:", o1)
    print("  update reply:", repr(upd[:500]))


def bake(model, M, temp=0.7):
    print(f"\n===== {model}  (M={M}, temp={temp}) =====", flush=True)
    needed = fixed = broke_shown = full_valid = 0
    for _ in range(M):
        o0, _ = grade("initial/shown", make_toc(model, temp), SHOWN)
        need = KEY in violations(o0, HELD)          # did initial violate the held edge?
        needed += need
        o1, vall = grade("updated/ALL", update_toc("\n".join(o0), model, temp), CONSTRAINTS)
        full_valid += not vall                       # updated TOC respects everything
        if KEY not in violations(o1, HELD):          # held edge satisfied after update
            fixed += 1
        if violations(o1, SHOWN):                    # update broke something it already had
            broke_shown += 1
    print(f"\n  initial-needed-fix={needed}/{M}   held-satisfied-after={fixed}/{M}"
          f"   broke-a-shown-edge={broke_shown}/{M}   fully-valid-after={full_valid}/{M}")


# ── Act 2: the two fixes for the in-place edit ────────────────────────────────

def make_toc_full(model, temp):
    """Re-derive from scratch: a fresh TOC built from ALL constraints at once,
    rather than editing a draft made under partial information."""
    p = (f"The book has these chapters:\n{CATALOG}\n\n"
         f"Ordering constraints (each chapter must come after what it builds on):\n"
         f"{block(CONSTRAINTS)}\n\n"
         f"Produce a valid table of contents with every constraint satisfied.{TAIL}")
    return ask(p, model=model, system=SOPHIA, max_tokens=400,
               options={"temperature": temp})


def fix_toc(prev, broken, model, temp):
    """Hand the checker's specific complaint back to the model and ask for a
    corrected TOC that also keeps every other dependency."""
    probs = "\n".join(f"- {p} must come before {d}, but right now it does not."
                      for p, d in broken)
    p = (f"This table of contents breaks some ordering constraints:\n{prev}\n\n"
         f"A checker found these problems:\n{probs}\n\nProduce a corrected table of "
         f"contents that fixes these AND keeps every dependency below satisfied:\n"
         f"{block(CONSTRAINTS)}{TAIL}")
    return ask(p, model=model, system=SOPHIA, max_tokens=400,
               options={"temperature": temp})


def check_loop(order, model, temp, K=3):
    """Re-check the WHOLE plan after the edit; bounce each violation back to the
    model until it is clean or we run out of rounds. Returns (order, rounds)."""
    rounds = 0
    while rounds < K:
        v = violations(order, CONSTRAINTS)
        if not v:
            break
        rounds += 1
        order = parse_order(fix_toc("\n".join(order), v, model, temp))
    return order, rounds


def compare(model, M, temp=0.7):
    """Three ways to land at a TOC that honors all constraints, graded under ALL:
    edit-in-place (Act 1), re-derive from scratch, and edit + checker-in-the-loop."""
    print(f"\n===== {model}  (compare, M={M}, temp={temp}) =====", flush=True)
    edit_ok = rederive_ok = check_ok = 0
    rounds_used = []
    for i in range(M):
        init = parse_order(make_toc(model, temp))               # plan from 80%
        edited = parse_order(update_toc("\n".join(init), model, temp))  # +20%, in place
        rederived = parse_order(make_toc_full(model, temp))     # fresh from 100%
        checked, r = check_loop(edited, model, temp)            # edit + validator
        e, d, c = (not violations(edited, CONSTRAINTS),
                   not violations(rederived, CONSTRAINTS),
                   not violations(checked, CONSTRAINTS))
        edit_ok += e; rederive_ok += d; check_ok += c; rounds_used.append(r)
        print(f"  run {i+1:2d}: edit={'ok ' if e else 'BAD'}  "
              f"rederive={'ok ' if d else 'BAD'}  "
              f"checked={'ok ' if c else 'BAD'} ({r} round{'s' if r != 1 else ''})",
              flush=True)
    avg = sum(rounds_used) / len(rounds_used)
    print(f"\n  edit-in-place fully-valid = {edit_ok}/{M}")
    print(f"  re-derive     fully-valid = {rederive_ok}/{M}")
    print(f"  checker-loop  fully-valid = {check_ok}/{M}   (avg {avg:.1f} fix rounds)")


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    models = sys.argv[2:] or ["gemma4:latest"]
    for mdl in models:
        triage(mdl) if M == 1 else compare(mdl, M)
