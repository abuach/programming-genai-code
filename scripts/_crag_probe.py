"""Reproducer / probe for the Corrective RAG (CRAG) demo in augmentation.ipynb.

CRAG {cite}`yan2024crag` grades retrieved docs and routes: CORRECT -> refine local
docs, INCORRECT -> fall back to a web search, AMBIGUOUS -> do both. This probe builds
the chapter's wiki_store and, per curated query, reports the retrieval-evaluator score
and verdict, whether STANDARD RAG (no correction) lands the known fact, and whether
CRAG lands it. It is how genai.crag.CRAG_STUDY was baked.

  M=1 (triage): prints the grade, both answers, and for traps the web snippet, so we
                can eyeball where the honest gap lives before baking.
  M>1 (bake):   the share of standard vs CRAG answers that contain the gold keyword.

WARNING: this hits the LIVE WEB through ddgs on the trap/ambiguous queries, so it is
slow, rate-limited, and its output drifts over time. The bake caches one web call per
query (the fact is stable; the variance we care about is the model's, not the web's),
which keeps it to a few calls total per the minimal-measurement rule.

Run from chapters/:  PYTHONPATH=. ../.venv/bin/python ../scripts/_crag_probe.py [M]
"""
import sys
import time
from genai.rag import build_wiki_store, wiki_retrieve_dense
from genai.crag import (grade_retrieval, standard_rag, crag,
                        answer_hit, web_search, CRAG_QUERIES)

# A FALSE CORRECT lure kept alongside the charted queries: it is an ML question, so it
# grades CORRECT (high similarity), but the corpus does not actually hold the mechanism,
# so CRAG trusts the bad docs and misses exactly as standard RAG does. Not charted; cited
# in the read-out as the evaluator's blind spot. (label, question, gold).
LURE = ("Learn", "How does a neural network learn from its errors during training?",
        {"gradient", "weight", "loss", "descent", "backprop", "adjust"})

# Subset baked into genai.crag.CRAG_STUDY: three local (CORRECT, no regression) and two
# traps (INCORRECT, recovered by the web). The grouped-bar chart is these five.
BAKE = ["CNN", "BERT", "Embedding", "Nobel '24", "Olympics"]


def _cached_web(store):
    """One live web call per query, reused across the M samples: the fact is stable,
    so re-searching M times would only burn the rate limit. The model's generation is
    what varies, and that we sample for real."""
    cache = {}
    def web_fn(query, k=3):
        if query not in cache:
            cache[query] = web_search(query, k)
        return cache[query]
    return web_fn


def triage(store):
    print("\n===== CRAG triage (M=1) =====", flush=True)
    for label, q, gold in list(CRAG_QUERIES) + [LURE]:
        chunks = [h["text"] for h in wiki_retrieve_dense(q, store, 3)]
        score, verdict = grade_retrieval(q, chunks)
        std = standard_rag(q, store)
        r = crag(q, store)
        tag = "TRAP" if label in ("Nobel '24", "Olympics") else (
              "LURE" if label == "Learn" else "local")
        print(f"\n## {label:11s} [{tag}]  score={score:.3f}  verdict={verdict}")
        print(f"   standard  hit={answer_hit(std, gold)!s:5s} | {std[:64]}")
        print(f"   crag      hit={answer_hit(r['answer'], gold)!s:5s} | {r['answer'][:64]}")
        if r["web"]:
            print(f"   web[0]: {r['web'][0]['title'][:40]} | {r['web'][0]['snippet'][:50]}")


def bake(store, M):
    print(f"\n===== CRAG bake (M={M}/query) =====", flush=True)
    web_fn = _cached_web(store)
    std_acc, crag_acc = [], []
    for label, q, gold in CRAG_QUERIES:
        if label not in BAKE:
            continue
        t0 = time.perf_counter()
        s_hits = sum(answer_hit(standard_rag(q, store), gold) for _ in range(M))
        c_hits = sum(answer_hit(crag(q, store, web_fn=web_fn)["answer"], gold)
                     for _ in range(M))
        dt = time.perf_counter() - t0
        std_acc.append(s_hits / M); crag_acc.append(c_hits / M)
        print(f"{label:11s} standard {s_hits/M:4.0%}  crag {c_hits/M:4.0%}   {dt:5.1f}s",
              flush=True)
    print(f'\n  "labels":   {BAKE}')
    print(f'  "standard": {[round(a, 3) for a in std_acc]}')
    print(f'  "crag":     {[round(a, 3) for a in crag_acc]}')


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    store, _ = build_wiki_store()
    if M == 1:
        triage(store)
    else:
        bake(store, M)
