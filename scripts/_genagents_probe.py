"""THROWAWAY probe: de-risk the Generative Agents (Smallville) memory demo.

Park et al. 2023 give an agent a *memory stream* of timestamped natural-language
observations and retrieve from it by a weighted blend of three factors:
RECENCY (exponential decay since last access), IMPORTANCE (the model rates each
memory 1-10 when it is created), and RELEVANCE (embedding similarity to the
current situation). Periodically the agent REFLECTS: it distils recent
observations into a higher-level insight and stores that back in the stream.

Three things can break the demo; this probe measures all of them cheaply:

  (a) IMPORTANCE -- does gemma4 give DIFFERENTIATED 1-10 scores (mundane low,
      poignant high), or does it flatten everything to a 5? If it flattens, the
      whole weighting is pointless, so adjust the prompt until it spreads.
  (b) REFLECTION -- does synthesising over the observations produce a GENUINE,
      non-vacuous higher-level insight about the student (not a restatement)?
  (c) RETRIEVAL REORDER -- with a curated observation set + a finals-prep query,
      does the 3-factor weighting visibly REORDER the top-k vs relevance-only?
      (If weighted == relevance-only, the demo teaches nothing.) Also checks that
      nomic embeddings are DETERMINISTIC, so a scripted demo's numbers are stable.

Run: .venv/bin/python scripts/_genagents_probe.py
"""
import re
import sys
import time

sys.path.insert(0, "chapters")
import numpy as np
from genai.llm import DEFAULT_MODEL, ask as _ask
from genai.embed import embed, similarity

# Sophia tutors one student, Marcus, and logs what she notices over ~10 days.
# The set is curated so importance and relevance pull in DIFFERENT directions:
# the exam-shaped facts (revision question, panic email) score high on
# relevance to the exam query, but the memory a companion most needs to act on
# (the missed reading-plan deadline) is recent + important yet NOT lexically
# exam-shaped. Mirrors agent.SOPHIA_OBSERVATIONS (the book-universe set).
NOW = 10.0
OBSERVATIONS = [
    (1, "Marcus asked which chapters to revise for the final exam."),
    (2, "Marcus finished the Prompting chapter exercises on time."),
    (4, "Marcus aced the end-of-chapter quiz on attention."),
    (8, "Marcus emailed at 2am, panicking about the upcoming final exam."),
    (9, "Marcus missed the reading-plan deadline he set for himself."),
]
QUERY = "How should I help Marcus prepare for his final exam?"

# Faithful to the paper's importance prompt (1 = mundane like brushing teeth,
# 10 = poignant like a breakup / college acceptance).
_IMP_PROMPT = (
    "On a scale of 1 to 10, where 1 is purely mundane (brushing teeth, making "
    "the bed) and 10 is extremely poignant (a breakup, a college acceptance), "
    "rate how significant this memory about a student is. "
    "Memory: {text}\nReply with a single integer, nothing else.")


def rate_importance(text, model=DEFAULT_MODEL):
    raw = _ask(_IMP_PROMPT.format(text=text), model=model, max_tokens=6).strip()
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else -1


def probe_importance(model=DEFAULT_MODEL):
    print(f"\n===== (a) IMPORTANCE  model={model} =====", flush=True)
    scores = []
    for day, text in OBSERVATIONS:
        t0 = time.perf_counter()
        s = rate_importance(text, model)
        scores.append(s)
        print(f"  imp={s:>2}  ({time.perf_counter()-t0:3.1f}s)  {text}", flush=True)
    print(f"  spread: min={min(scores)} max={max(scores)} distinct={len(set(scores))}",
          flush=True)
    return scores


def synthesize_insight(observations, model=DEFAULT_MODEL):
    joined = "\n".join(f"- {o}" for o in observations)
    return _ask(
        "Here are recent observations about a student:\n" + joined +
        "\n\nStep back. What is the single most important higher-level insight "
        "you can conclude about this student? Reply with one sentence.",
        model=model, max_tokens=60).strip()


def probe_reflection(model=DEFAULT_MODEL):
    print(f"\n===== (b) REFLECTION  model={model} =====", flush=True)
    texts = [t for _, t in OBSERVATIONS]
    for k in range(3):                      # a few rolls to see if it is stable
        t0 = time.perf_counter()
        insight = synthesize_insight(texts, model)
        print(f"  roll {k+1} ({time.perf_counter()-t0:3.1f}s): {insight}", flush=True)


def _recency(day, decay):
    return decay ** (NOW - day)


def probe_retrieval(scores, decay=0.85):
    print(f"\n===== (c) RETRIEVAL REORDER  decay={decay} =====", flush=True)
    qv = embed(QUERY)
    rows = []
    for (day, text), imp in zip(OBSERVATIONS, scores):
        rec = _recency(day, decay)
        rel = similarity(qv, embed(text))
        rows.append({"day": day, "text": text, "imp": imp, "rec": rec, "rel": rel})

    # min-max normalise each factor across the candidate set (the paper's move),
    # then equal-weight sum. Also keep the raw-normalised version for comparison.
    def minmax(vals):
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]
    recs = minmax([r["rec"] for r in rows])
    imps = minmax([r["imp"] for r in rows])
    rels = minmax([r["rel"] for r in rows])
    for r, rc, ip, rl in zip(rows, recs, imps, rels):
        r["nrec"], r["nimp"], r["nrel"] = rc, ip, rl
        r["total"] = rc + ip + rl                       # equal weights
        r["raw_total"] = r["rec"] + r["imp"]/10 + r["rel"]  # un-normalised blend

    print("  day imp  rec   rel  | nrec nimp nrel  TOTAL | rawT | text")
    for r in sorted(rows, key=lambda r: -r["total"]):
        print(f"  {r['day']:>3} {r['imp']:>3} {r['rec']:.2f} {r['rel']:.2f} | "
              f"{r['nrec']:.2f} {r['nimp']:.2f} {r['nrel']:.2f}  {r['total']:.2f} | "
              f"{r['raw_total']:.2f} | {r['text'][:42]}", flush=True)

    def topk(key, k=2):
        return [r["text"][:34] for r in sorted(rows, key=lambda r: -r[key])[:k]]
    print("\n  WEIGHTED  top-2:", topk("total"))
    print("  RAW-BLEND top-2:", topk("raw_total"))
    print("  RELEVANCE top-2:", topk("rel"))
    reorder = topk("total") != topk("rel")
    print(f"\n  --> weighted reorders vs relevance-only? {reorder}")
    return rows


def probe_determinism():
    print(f"\n===== (d) EMBEDDING DETERMINISM =====", flush=True)
    drift = 0.0
    for _, text in OBSERVATIONS + [(0, QUERY)]:
        a, b = embed(text), embed(text)
        drift = max(drift, float(np.abs(a - b).max()))
    print(f"  max re-embed drift = {drift:.2e}  (0 => safe to bake, no freeze)",
          flush=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    scores = probe_importance()
    probe_reflection()
    probe_retrieval(scores)
    probe_determinism()
    print(f"\n[probe done in {time.perf_counter()-t0:.0f}s]")
