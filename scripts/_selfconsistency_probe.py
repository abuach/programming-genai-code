"""THROWAWAY probe: find an honest substrate for a Self-Consistency demo.

Self-consistency = sample several CoT reasoning chains at a real temperature,
extract each final answer, and take a majority vote {cite}`wang2023selfconsistency`.
It should BEAT a single sample when the model is right MORE OFTEN THAN WRONG but
wobbles (p_single in ~0.5-0.85, diverse chains), and it CANNOT help when the model
has a stable wrong prior (one confident wrong answer dominates every sample) --
the exact mirror of the Reflexion gotcha (retry fixes wobble, reflection fixes a
stuck prior). This probe measures, per (model, problem):
  - p_single : fraction of individual CoT samples that are correct
  - hist     : the answer histogram (is a single wrong value dominating?)
  - vote@N   : bootstrap accuracy of majority-vote-of-N over the M samples

Run: .venv/bin/python scripts/_selfconsistency_probe.py [M] [model ...]
"""
import sys, re, time, random
from collections import Counter
sys.path.insert(0, "chapters")
import ollama
from genai.llm import SERVER
from genai.thinking import NOVEL_PROBLEMS, _grade

_client = ollama.Client(host=SERVER)
random.seed(0)

# System-1 "lure" traps with uncommon numbers (dodging the memorized originals).
# Each has an intuitive WRONG answer the model tends to commit to on every sample
# (a stable wrong prior), so voting amplifies the lure instead of fixing it -- the
# honest ceiling of self-consistency, and the place a thinking model earns its keep.
TRAP_PROBLEMS = [
    ("Pen",   "A textbook and a pen cost 240 dollars together. The textbook costs "
              "200 dollars more than the pen. How many dollars does the pen cost?",
     ["20"]),                                            # lure: 40
    ("Cooks", "If 6 cooks bake 6 cakes in 6 hours, how many hours do 18 cooks need "
              "to bake 18 cakes?", ["6"]),               # lure: 18
    ("Algae", "A patch of algae doubles in size every day and covers a lake in 60 "
              "days. On what day was the lake half covered?", ["59"]),  # lure: 30
    ("Widgets", "If 7 printers print 7 pages in 7 seconds, how many seconds do 28 "
                "printers need to print 28 pages?", ["7"]),  # lure: 28
]

# CoT prompt: reason in the open, then commit to a parseable final line. The
# diversity across samples (driven by temperature) is what makes voting work.
_COT = (" Think step by step, then end with a line exactly like 'Answer: <number>'.")


def cot_sample(question: str, model: str, temp: float = 0.8,
               max_tokens: int = 350) -> str:
    """One sampled chain-of-thought; return the extracted final integer (as str)."""
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": question + _COT}],
                        options={"num_predict": max_tokens, "temperature": temp})
    text = (resp["message"]["content"] or "")
    m = re.findall(r"answer\s*[:=]\s*\$?(-?\d+)", text, re.I)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d+", text)          # fallback: last integer in the reply
    return nums[-1] if nums else ""


def vote_at_n(samples: list, gold: list, n: int, trials: int = 400) -> float:
    """Bootstrap: draw n samples, take the mode, score it; average over trials."""
    hits = 0
    for _ in range(trials):
        draw = [random.choice(samples) for _ in range(n)]
        mode = Counter(draw).most_common(1)[0][0]
        hits += _grade(mode, gold)
    return hits / trials


def probe(model: str, M: int, problems=NOVEL_PROBLEMS):
    print(f"\n===== {model}  (M={M} samples/problem, temp=0.8) =====", flush=True)
    for label, q, gold in problems:
        t0 = time.perf_counter()
        samples = [cot_sample(q, model) for _ in range(M)]
        dt = time.perf_counter() - t0
        p1 = sum(_grade(s, gold) for s in samples) / M
        hist = Counter(samples).most_common(4)
        v3, v5, v7 = (vote_at_n(samples, gold, n) for n in (3, 5, 7))
        star = "  <-- gold " + str(gold)
        print(f"{label:11s} p1={p1:4.0%}  vote@3={v3:4.0%} @5={v5:4.0%} "
              f"@7={v7:4.0%}  {dt:4.1f}s  hist={hist}{star}", flush=True)


if __name__ == "__main__":
    # usage: _selfconsistency_probe.py [M] [novel|traps] [model ...]
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    rest = sys.argv[2:]
    problems = NOVEL_PROBLEMS
    if rest and rest[0] in ("novel", "traps"):
        problems = TRAP_PROBLEMS if rest[0] == "traps" else NOVEL_PROBLEMS
        rest = rest[1:]
    models = rest or ["qwen2.5-coder:latest"]
    for mdl in models:
        probe(mdl, M, problems)
