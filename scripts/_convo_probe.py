"""Reproducer: the "Why Conversations Slow Down" demo in efficiency.ipynb.

Holds a six-turn study-session chat with llama3.2 and times every turn twice:

  warm : the natural ongoing conversation. The Ollama server still holds the
         KV cache from the previous turn, so each request only has to read
         the newest question (prompt_ms stays flat).
  cold : the same transcript snapshots replayed via time_chat(fresh=True),
         which runs `ollama stop` first. The model arrives with no memory of
         the chat and pays the full reading bill for the whole transcript
         (prompt_ms climbs with transcript length).

Timing pitfall this probe is built around (see memory: efficiency-padding-demo):
re-sending text the server has cached still reports the full prompt_eval_count
but prompt_eval_duration collapses. Token counts are robust, read-times are
not, so the warm run must happen exactly once, in order, on a freshly
re-warmed model.

Writes chapters/_convo_results.json (warm rows + replies, cold rows) and
prints a per-turn table. The frozen notebook cells embed one real run of
this; digits drift run to run but the shape (flat warm, climbing cold) is
structural.

Run: .venv/bin/python scripts/_convo_probe.py
"""
import json
import subprocess
import sys

sys.path.insert(0, "chapters")
from genai.perf import time_chat, time_call

MODEL = "llama3.2:latest"

QUESTIONS = [
    "What's a token in a language model?",
    "How do embeddings relate to tokens?",
    "What is retrieval-augmented generation?",
    "How does an AI agent use tools?",
    "Why do language models hallucinate?",
    "Summarize everything we've covered so far.",
]


def main():
    # Flush whatever conversation cache the server holds, then re-warm with an
    # unrelated prompt so turn 1 pays a real (but load-free) reading cost.
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)
    time_call("Say hello.", model=MODEL)

    chat = [{"role": "system", "content": "Answer in one short paragraph."}]
    warm, prefixes, replies = [], [], []
    for q in QUESTIONS:
        chat.append({"role": "user", "content": q})
        prefixes.append(list(chat))
        reply, m = time_chat(chat, model=MODEL)
        chat.append({"role": "assistant", "content": reply})
        warm.append(m)
        replies.append(reply)

    cold = [time_chat(p, model=MODEL, fresh=True)[1] for p in prefixes]

    print(f"{'turn':>4} {'tok(warm)':>10} {'tok(cold)':>10} "
          f"{'warm ms':>8} {'cold ms':>8} {'warm t/s':>9} {'cold t/s':>9}")
    for i, (w, c) in enumerate(zip(warm, cold), 1):
        print(f"{i:>4} {w['prompt_tokens']:>10} {c['prompt_tokens']:>10} "
              f"{w['prompt_ms']:>8.0f} {c['prompt_ms']:>8.0f} "
              f"{w['tokens_per_sec']:>9.1f} {c['tokens_per_sec']:>9.1f}")

    out = {"model": MODEL, "questions": QUESTIONS,
           "warm": warm, "cold": cold, "replies": replies}
    with open("chapters/_convo_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote chapters/_convo_results.json")

    print("\n--- transcript ---")
    for q, r in zip(QUESTIONS, replies):
        print(f"\nUSER: {q}\nMODEL: {r[:200]}{'...' if len(r) > 200 else ''}")


if __name__ == "__main__":
    main()
