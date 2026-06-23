"""THROWAWAY probe: de-risk the Voyager 'growing skill library' demo.

Two things can break this demo, and this probe measures both cheaply:

  (a) SKILL WRITING -- can the coding model author a SMALL, GENERAL, reusable
      skill that passes a hidden-contract test (empty input, etc.) on the first
      try or two? A skill only compounds if it was written generally.
  (b) RETRIEVAL -- does description-similarity retrieval reliably surface the
      RIGHT earlier skill for a related-but-different new task, beating the
      distractors? And are embeddings deterministic, so a scripted demo's
      similarity numbers are stable enough to bake?

Run: .venv/bin/python scripts/_voyager_probe.py
"""
import sys, time
sys.path.insert(0, "chapters")
import numpy as np
from genai.llm import CODING_MODEL, ask as _ask
from genai.embed import embed, similarity
from genai.agent import code_test, _extract_code

_CODER = "You are a careful Python programmer."

# Terse specs (like a hurried request); the TEST holds the full contract -- the
# short/empty/edge cases a first attempt rarely anticipates. A general skill handles
# them; a narrow one fails and would not transfer. reading_time is the skill Sophia
# writes live in the book demo.
_W = lambda n: "w " * n  # an n-word passage
SKILL_TASKS = [
    ("reading_time", "Write reading_time(text): estimate how many whole minutes the "
                     "passage takes to read at 200 words per minute, rounding up so "
                     "even a short passage takes at least a minute.",
     [((_W(200),), 1), ((_W(250),), 2), ((_W(30),), 1), (("",), 1), ((_W(600),), 3)]),
    ("count_words", "Write count_words(text): count the number of words in the text.",
     [(("hello world",), 2), (("",), 0), (("   ",), 0), (("  a  b c ",), 3)]),
    ("slugify", "Write slugify(title): turn a section title into a url slug -- "
                "lowercase, with runs of non-alphanumeric characters collapsed to "
                "single hyphens and no leading or trailing hyphen.",
     [(("Word Arithmetic",), "word-arithmetic"), (("Intro",), "intro"),
      (("A/B Test!",), "a-b-test")]),
]

# (a) skill writing -------------------------------------------------------------
def probe_writing(model=CODING_MODEL, tries=2):
    print(f"\n===== (a) SKILL WRITING  model={model} =====", flush=True)
    for name, spec, cases in SKILL_TASKS:
        check = code_test(name, cases)
        t0 = time.perf_counter()
        for k in range(1, tries + 1):
            code = _extract_code(_ask(f"{spec} Reply with only the function in one "
                                      "code block.", model=model, system=_CODER))
            ok, detail = check(code)
            if ok:
                break
        dt = time.perf_counter() - t0
        crux = " | ".join(l.strip() for l in code.splitlines() if l.strip())[:88]
        print(f"{name:12s} try{k} {'PASS' if ok else 'FAIL':4s} {dt:4.1f}s  {detail}",
              flush=True)
        print(f"             code: {crux}", flush=True)


# Library of skill DESCRIPTIONS (what each skill is filed under) for a book
# assistant. reading_time is the Act-1 skill Sophia writes; the rest are distractors
# the library already holds.
LIB = {
    "reading_time":     "estimate how many minutes a passage takes to read",
    "count_words":      "count the words in a passage",
    "slugify":          "turn a section title into a url slug",
    "extract_headings": "pull the section headings out of a chapter",
    "summarize":        "write a short summary of a passage",
    "preview":          "trim a passage down to a short preview",
}

# Related-but-different NEW tasks; each names the skill it SHOULD retrieve. The last
# is the trap: "trim ... down" pulls toward preview even though the intent is a
# summary.
QUERIES = [
    ("how long will this section take to read",   "reading_time"),
    ("how many words are in this answer",         "count_words"),
    ("make a url anchor from this section title", "slugify"),
    ("list the section headings in this chapter", "extract_headings"),
    ("trim this section down to its main idea",   "summarize"),
]


# (b) retrieval + determinism ---------------------------------------------------
def probe_retrieval():
    print(f"\n===== (b) RETRIEVAL over {len(LIB)} skill descriptions =====", flush=True)
    vecs = {name: embed(desc) for name, desc in LIB.items()}
    # determinism: re-embed one description and compare
    again = embed(LIB["reading_time"])
    drift = float(np.max(np.abs(np.array(again) - np.array(vecs["reading_time"]))))
    print(f"embedding determinism: max abs drift on re-embed = {drift:.2e}", flush=True)

    hits = 0
    for query, want in QUERIES:
        q = embed(query)
        scored = sorted(((name, similarity(q, v)) for name, v in vecs.items()),
                        key=lambda x: x[1], reverse=True)
        top, top_s = scored[0]
        second, second_s = scored[1]
        margin = top_s - second_s
        ok = top == want
        hits += ok
        flag = "OK " if ok else "XX "
        print(f"{flag}{query[:42]:42s} -> {top:12s} {top_s:.3f} "
              f"(margin {margin:+.3f} over {second}) want={want}", flush=True)
    print(f"retrieval: {hits}/{len(QUERIES)} correct top-1", flush=True)


if __name__ == "__main__":
    probe_writing()
    probe_retrieval()
