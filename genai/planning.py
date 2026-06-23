"""Planning helpers for the Agentic chapter: plan, then keep the plan valid.

The plan here is the book's own table of contents. We give Sophia most of the
dependencies between chapters and ask her to order them, then hand her the one
constraint we held back and ask her to update the plan. The honest finding is
that the in-place edit is unreliable: she always folds the new constraint in, but
about half the time she breaks one she already had, because she patches locally
without re-checking the whole plan. Two disciplines fix it, to different degrees:
re-deriving the plan from the full constraint set, and wrapping the planner in a
checker that re-validates the whole thing and bounces each violation back.

The calls here mirror scripts/_plan_probe.py exactly (same SOPHIA system prompt,
think=False, the same prompts and token budgets), so PLAN_STUDY below is a
faithful record of what these functions do on gemma4:latest, and PLAN_DEMO is one
real captured run (via scripts/_plan_capture.py) rendered deterministically.
"""
from genai.llm import ask, DEFAULT_MODEL
from genai.agent import show_turn

# The book's chapters, presented to the model in neutral (alphabetical) order so
# the answer ordering is never leaked in the prompt.
CHAPTERS = sorted([
    "Foundations", "Prompting", "Tokens", "Semantics", "Metacoding",
    "Augmentation", "Agentic", "Autonomy", "Thinking", "Multimodal",
    "Design", "Efficiency", "Responsible",
])

# (prereq, dependent, why): prereq MUST come before dependent. These are the real
# strong backward cross-references the chapters make to each other. CONSTRAINTS[0]
# is the keystone, held back as the constraint we "forgot"; the rest are shown up
# front. Augmentation -> Agentic is the one the other six leave under-determined.
CONSTRAINTS = [
    ("Augmentation", "Agentic",      "Agentic uses the search_book tool and corpus built in Augmentation"),
    ("Agentic",      "Autonomy",     "Autonomy is the deployment half that continues Agentic"),
    ("Semantics",    "Augmentation", "Augmentation's retrieval needs the embeddings from Semantics"),
    ("Semantics",    "Metacoding",   "Metacoding builds on the embeddings from Semantics"),
    ("Augmentation", "Design",       "Design reuses Augmentation's retrieval"),
    ("Agentic",      "Responsible",  "Responsible stress-tests the agent from Agentic"),
    ("Autonomy",     "Responsible",  "Responsible stress-tests the deployment from Autonomy"),
]
HELD  = CONSTRAINTS[:1]   # revealed only at update time (the keystone)
SHOWN = CONSTRAINTS[1:]   # the six dependencies given up front
KEY   = (HELD[0][0], HELD[0][1])

SOPHIA = ("You are Sophia, the companion for a generative-AI textbook. You plan "
          "the book's table of contents so every chapter comes after the chapters "
          "it builds on.")
CATALOG = "\n".join(f"- {c}" for c in CHAPTERS)
TAIL = ("\n\nEnd with a line 'ORDER:' followed by every chapter name in order, one "
        "per line, using exactly the names above.")


def block(edges):
    """The constraint list as plain English, one dependency per line."""
    return "\n".join(f"- {p} must come before {d}  ({why})" for p, d, why in edges)


def parse_order(text):
    """Recover the ordering from a reply: the text after the last 'ORDER:' marker,
    chapters ranked by first appearance."""
    tail = text.rsplit("ORDER:", 1)[-1].lower()
    hits = [(tail.find(c.lower()), c) for c in CHAPTERS if c.lower() in tail]
    return [c for _, c in sorted(hits)]


def violations(order, edges):
    """Every dependency the ordering breaks: a prerequisite that lands after the
    chapter depending on it. An empty list means the plan is valid."""
    pos = {c: i for i, c in enumerate(order)}
    return [(p, d) for p, d, _ in edges
            if p in pos and d in pos and pos[p] > pos[d]]


# ── The four model calls (mirror scripts/_plan_probe.py) ──────────────────────

def make_toc(shown=SHOWN, model=DEFAULT_MODEL, temp=0.7):
    """Plan a table of contents from the dependencies in ``shown``."""
    p = (f"The book has these chapters:\n{CATALOG}\n\nOrdering constraints (each "
         f"chapter must come after what it builds on):\n{block(shown)}\n\nProduce a "
         f"valid table of contents with every constraint satisfied.{TAIL}")
    return ask(p, model=model, system=SOPHIA, max_tokens=400, options={"temperature": temp})


def update_toc(prev, held=HELD, model=DEFAULT_MODEL, temp=0.7):
    """Edit an existing TOC in place to honor one newly revealed constraint."""
    p = (f"Here is the current table of contents:\n{prev}\n\nOne more constraint we "
         f"left out earlier:\n{block(held)}\n\nUpdate the table of contents so it "
         f"respects this new constraint too, while keeping every earlier constraint "
         f"satisfied and changing as little as possible.{TAIL}")
    return ask(p, model=model, system=SOPHIA, max_tokens=400, options={"temperature": temp})


def fix_toc(prev, broken, model=DEFAULT_MODEL, temp=0.7):
    """Hand the checker's specific complaint back and ask for a corrected TOC."""
    probs = "\n".join(f"- {p} must come before {d}, but right now it does not."
                      for p, d in broken)
    p = (f"This table of contents breaks some ordering constraints:\n{prev}\n\nA "
         f"checker found these problems:\n{probs}\n\nProduce a corrected table of "
         f"contents that fixes these AND keeps every dependency below satisfied:\n"
         f"{block(CONSTRAINTS)}{TAIL}")
    return ask(p, model=model, system=SOPHIA, max_tokens=400, options={"temperature": temp})


def check_loop(order, model=DEFAULT_MODEL, temp=0.7, rounds=3):
    """Re-check the whole plan and bounce each violation back until it is clean or
    we run out of rounds. Returns ``(order, rounds_used)``."""
    used = 0
    while used < rounds:
        bad = violations(order, CONSTRAINTS)
        if not bad:
            break
        used += 1
        order = parse_order(fix_toc("\n".join(order), bad, model, temp))
    return order, used


# ── Measured on gemma4:latest, M=15 at temperature 0.7, via _plan_probe.py ─────
# Three ways to land at a TOC that honors all seven dependencies, each graded
# under the full set. `valid` is the share fully valid: editing the draft in place
# satisfies the new constraint every time but regresses on an old one about half
# the time; re-deriving from the full set is far better but still fumbles a fresh
# sort one time in five; only the checker-in-the-loop, which re-validates the whole
# plan and returns each broken edge, reaches every-time validity (in a single fix
# round whenever the edit broke something).
PLAN_STUDY = {
    "samples": 15,
    "labels": ["edit in place", "re-derive", "checker loop"],
    "valid":  [7 / 15, 12 / 15, 15 / 15],
}

# One real captured run (scripts/_plan_capture.py), rendered deterministically by
# the show_* helpers below. The transcript is a single representative instance of
# the half-the-time regression PLAN_STUDY measures; the rates, not this one run,
# are the evidence. FILLED FROM CAPTURE.
PLAN_DEMO = {
    # Sane plan, valid under the six shown constraints, but Agentic sits ahead of
    # Augmentation (the hidden keystone).
    "initial": ["Foundations", "Tokens", "Semantics", "Metacoding", "Prompting",
                "Thinking", "Multimodal", "Agentic", "Autonomy", "Responsible",
                "Augmentation", "Design", "Efficiency"],
    # Folding in the keystone, she pulls Augmentation and Agentic together but lets
    # Design slip ahead of Augmentation.
    "updated": ["Foundations", "Tokens", "Semantics", "Prompting", "Thinking",
                "Metacoding", "Design", "Efficiency", "Augmentation", "Agentic",
                "Multimodal", "Autonomy", "Responsible"],
    "broke":   ("Augmentation", "Design"),
    # After the checker returns the broken edge, every dependency holds.
    "fixed":   ["Foundations", "Tokens", "Semantics", "Metacoding", "Prompting",
                "Thinking", "Efficiency", "Augmentation", "Design", "Agentic",
                "Multimodal", "Autonomy", "Responsible"],
}


def _toc(order):
    return ", ".join(order)


def show_draft():
    """The six dependencies Sophia is shown, and the table of contents she plans
    from them. Real captured output, rendered deterministically."""
    show_turn("RULES", "; ".join(f"{p} before {d}" for p, d, _ in SHOWN))
    show_turn("SOPHIA", _toc(PLAN_DEMO["initial"]))
    show_turn("CHECK", "valid: every dependency shown holds")


def show_update():
    """Reveal the constraint we held back. Sophia folds it in, but the in-place
    edit breaks one she already had."""
    p, d, why = HELD[0]
    show_turn("NEW", f"{p} before {d}  ({why})")
    show_turn("SOPHIA", _toc(PLAN_DEMO["updated"]))
    bp, bd = PLAN_DEMO["broke"]
    show_turn("CHECK", f"broken: {bp} must come before {bd}, but now it doesn't")


def show_fix():
    """The checker returns the broken edge; the corrected plan is clean again."""
    bp, bd = PLAN_DEMO["broke"]
    show_turn("FIX", f"checker returns: {bp} must come before {bd}")
    show_turn("SOPHIA", _toc(PLAN_DEMO["fixed"]))
    show_turn("CHECK", "valid: all seven dependencies hold")
