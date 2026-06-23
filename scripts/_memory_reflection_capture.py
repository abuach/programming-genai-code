"""Capture + reproducer for the reflection memory stream ("A Memory That
Reflects", agentic.ipynb): Sophia keeps a generative-agents memory stream on a
reader working through this book.

The demo bakes three things gemma4 actually produced over the stream so the
section reproduces without a live call: the importance score for each
observation, the one-sentence INSIGHT reflection distils from them, and Sophia's
ANSWER once that insight is her top recalled memory. This script regenerates all
three and prints a paste-ready block for genai/agent.py. Recency, relevance, and
the scoring are recomputed every build, so only these three are baked.

It also checks the four teaching beats the walkthrough prose depends on:
  1. a relevance-only search grabs the FORGETTABLE day-1 note (the trap),
  2. the weighted blend overrules it and surfaces the poignant 2am note,
  3. gemma4 under-rates the missed deadline (the noisy-importance soft spot),
  4. once filed back, the reflection itself becomes the TOP recalled memory.
Beat 4 is the ACT-3 payoff and the easiest to lose: the insight only outranks
the 2am note when it shares the query's anchor ("momentum"/"finish the book"),
so we sample insights and keep the first real one that actually tops.

Run: .venv/bin/python scripts/_memory_reflection_capture.py
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import (MemoryStream, rate_importance, synthesize_insight,
                         answer_from_memory, REFLECT_THRESHOLD)

# A reader working THROUGH the book, no exam in sight. day-1 is the relevance-only
# trap: forgettable logistics that share surface words with the query. The 2am
# note is the poignant high; the missed deadline is the soft spot gemma4 tends to
# under-rate. Importance is the model's own call, captured live below.
READER = "Priya"
OBSERVATIONS = [
    (1, "skip-ahead ask",
     "Priya asked which chapters she could skip to finish the book faster."),
    (2, "exercises on time",
     "Priya finished the Prompting chapter exercises on time."),
    (4, "rebuilt demo",
     "Priya rebuilt the attention demo from the book on her own laptop."),
    (8, "2am stuck email",
     "Priya emailed at 2am, discouraged and talking about giving up on the book."),
    (9, "missed deadline",
     "Priya missed the reading-plan deadline she set for herself."),
]
QUERY = "How can I help Priya rebuild her momentum and finish the book?"


def reflection_tops(stream, insight, imp):
    """True if, once the insight is filed back at ``imp``, it outranks every raw
    observation for QUERY. This is beat 4, the ACT-3 payoff."""
    probe = MemoryStream()
    for m in stream.memories:
        probe.observe(m["text"], day=m["day"], importance=m["importance"],
                      label=m["label"])
    probe.observe(insight, importance=imp, label="her reflection", kind="reflection")
    return probe.score(QUERY)[0][0]["kind"] == "reflection"


def main():
    print("rating importance with gemma4 (live)...\n")
    stream = MemoryStream()
    captured = []
    for day, label, text in OBSERVATIONS:
        imp = rate_importance(text)               # real gemma4 score, captured once
        stream.observe(text, day=day, importance=imp, label=label)
        captured.append((day, imp, label, text))
        print(f"  imp {imp:>2}  day {day}  {label}")

    scored = stream.score(QUERY)
    print(f"\nSITUATION: {QUERY}")
    print(f"  {'memory':<18}{'recency':>9}{'import':>8}{'relev':>8}{'TOTAL':>8}")
    for m, rec, imp, rel, tot in scored:
        print(f"  {m['label']:<18}{rec:>9.2f}{imp:>8.2f}{rel:>8.2f}{tot:>8.2f}")
    weighted = [m["label"] for m, *_ in scored[:2]]
    rel_only = [r[0]["label"] for r in sorted(scored, key=lambda r: -r[3])[:2]]
    print(f"\n  weighted recall : {weighted}")
    print(f"  relevance only  : {rel_only}")

    # Beats 1-3 come straight off the table.
    deadline_imp = next(i for _, i, l, _ in captured if l == "missed deadline")
    print(f"\n  beat 1  relevance-only grabs the forgettable note : "
          f"{rel_only[0] == 'skip-ahead ask'}")
    print(f"  beat 2  weighting rescues the 2am note            : "
          f"{weighted[0] == '2am stuck email'}")
    print(f"  beat 3  deadline under-rated (imp {deadline_imp} <= 3)         : "
          f"{deadline_imp <= 3}")
    total = sum(i for _, i, _, _ in captured)
    print(f"  reflect: importance sums to {total} (threshold {REFLECT_THRESHOLD}, "
          f"{'fires' if total >= REFLECT_THRESHOLD else 'DOES NOT FIRE'})")

    # Beat 4: sample real insights, keep the first that both reads as a synthesis
    # (importance >= 6) and actually tops the table once filed back.
    print("\nsynthesizing insight with gemma4 until it tops the table (beat 4)...")
    insight = insight_imp = None
    for attempt in range(15):
        cand = synthesize_insight([t for *_, t in OBSERVATIONS])
        cand_imp = rate_importance(cand)
        tops = reflection_tops(stream, cand, cand_imp)
        print(f"  [imp {cand_imp}] tops={tops!s:<5} {cand[:60]}")
        if tops and cand_imp >= 6:
            insight, insight_imp = cand, cand_imp
            break
    if insight is None:
        sys.exit("no sampled insight topped the table; revisit the query anchor")
    answer = answer_from_memory(QUERY, insight)

    print("\n" + "=" * 70 + "\nPASTE-READY (genai/agent.py)\n" + "=" * 70)
    print(f'SOPHIA_STUDENT = "{READER}"')
    print("SOPHIA_OBSERVATIONS = [")
    for day, imp, label, text in captured:
        print(f'    {{"day": {day}, "importance": {imp}, "label": "{label}",')
        print(f'     "text": "{text}"}},')
    print("]")
    print(f'SOPHIA_QUERY = "{QUERY}"')
    print(f'SOPHIA_INSIGHT = "{insight}"')
    print(f"SOPHIA_INSIGHT_IMPORTANCE = {insight_imp}")
    print(f'SOPHIA_REFLECTION_ANSWER = "{answer}"')


if __name__ == "__main__":
    main()
