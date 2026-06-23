"""Probe: does generation speed sag measurably as a conversation grows?

Extends the "Why Conversations Slow Down" demo (scripts/_convo_probe.py).
Theory: every generated token attends over the whole KV cache, so per-token
work grows linearly with context. At 460 tokens we measured only 42.0 -> 40.4
tok/s; this probe keeps the session going to ~6k tokens to see whether the
sag becomes chart-worthy.

Design notes:
  - num_ctx is pinned to 8192 on EVERY call. Ollama's default window (4096)
    silently truncates long transcripts, which would flatten the curve.
    Pinning it once for all points also keeps the allocation constant, so
    the only thing varying is how full the context is.
  - The transcript grows via "Tell me more." turns under the same
    one-short-paragraph system message as the book demo.
  - Truncation check: reported prompt_tokens must keep growing monotonically
    past 4096.
  - Thermal control: after the marathon, a fresh one-question chat at the
    same num_ctx. If speed bounces back near the start value, the slope is
    context-driven, not heat.

Writes chapters/_convo_sag_results.json.
Run: .venv/bin/python scripts/_convo_sag_probe.py
"""
import json
import subprocess
import sys

sys.path.insert(0, "chapters")
from genai.perf import time_chat

MODEL = "llama3.2:latest"
OPTS = {"num_ctx": 8192}
TARGET_TOKENS = 5800
MAX_EXTRA_TURNS = 75

QUESTIONS = [
    "What's a token in a language model?",
    "How do embeddings relate to tokens?",
    "What is retrieval-augmented generation?",
    "How does an AI agent use tools?",
    "Why do language models hallucinate?",
    "Summarize everything we've covered so far.",
]


def main():
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)
    time_chat([{"role": "user", "content": "Say hello."}],
              model=MODEL, options=OPTS)  # load + allocate once

    chat = [{"role": "system", "content": "Answer in one short paragraph."}]
    rows = []
    turns = QUESTIONS + ["Tell me more."] * MAX_EXTRA_TURNS
    for i, q in enumerate(turns, 1):
        chat.append({"role": "user", "content": q})
        reply, m = time_chat(chat, model=MODEL, options=OPTS)
        chat.append({"role": "assistant", "content": reply})
        rows.append(m)
        if i % 5 == 0 or i <= 6:
            print(f"turn {i:3d}  ctx {m['prompt_tokens']:5d}  "
                  f"{m['tokens_per_sec']:5.1f} tok/s  "
                  f"gen {m['gen_tokens']:3d}  read {m['prompt_ms']:6.0f} ms",
                  flush=True)
        if m["prompt_tokens"] >= TARGET_TOKENS:
            break

    # Truncation check
    toks = [r["prompt_tokens"] for r in rows]
    mono = all(b > a for a, b in zip(toks, toks[1:]))
    print(f"\nprompt_tokens strictly increasing: {mono} (max {max(toks)})")

    # Thermal control: tiny fresh chat, same allocation
    ctrl_chat = [{"role": "system", "content": "Answer in one short paragraph."},
                 {"role": "user", "content": QUESTIONS[0]}]
    _, ctrl = time_chat(ctrl_chat, model=MODEL, fresh=True, options=OPTS)
    # fresh=True evicted the model; re-warm happened inside that call.
    _, ctrl2 = time_chat(ctrl_chat, model=MODEL, options=OPTS)
    print(f"thermal control (small ctx after marathon): "
          f"{ctrl['tokens_per_sec']:.1f} then {ctrl2['tokens_per_sec']:.1f} tok/s "
          f"vs first-turn {rows[0]['tokens_per_sec']:.1f}")

    out = {"model": MODEL, "options": OPTS, "rows": rows,
           "control": [ctrl, ctrl2]}
    with open("chapters/_convo_sag_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote chapters/_convo_sag_results.json")


if __name__ == "__main__":
    main()
