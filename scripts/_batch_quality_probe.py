"""Probe: does unbatching (one question per call) beat batching on QUALITY?

Replicates the exact setup of the "Batching Queries with Shared Context"
demo in efficiency.ipynb (same ctx, same qs, same llama3.2:latest, same
prompt construction), but captures the answers instead of timing them.

Finding (M=8, hand-graded 2026-06-11): batched 20/24 facts vs unbatched
16/24 — batching does NOT cost quality here; it helps. The unbatched
"What was it trained on?" went 0/8: a lone claim+question reads like a
fact-check request, and llama3.2's (wrong) parametric prior fights the
context ("GPT-3 was actually released in 2021"). Bundled questions read
like a comprehension quiz and flip the model into extract-from-passage
mode. Batching's real cost is CORRELATED failure: one bad batched call
(trial 2 derailed into "this is GPT-2") loses all three answers at once.

GRADING CAVEAT: do not string-match these transcripts — denials quote
the numbers ("not ~300 billion tokens" matches "300"). Hand-grade from
the JSON dump this script writes.
"""
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"
M = 8  # trials per mode
OUT = "/tmp/batch_quality_transcripts.json"

ctx = ("GPT-3 was released by OpenAI in 2020. It has 175 billion "
       "parameters and was trained on ~300 billion tokens of text.")
qs = ["Who released it?", "How many parameters?", "What was it trained on?"]


def gen(prompt: str) -> str:
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt,
                                        "stream": False})
    return r.json()["response"]


log = {"unbatched": [], "batched": []}
for trial in range(M):
    log["unbatched"].append(
        [{"q": q, "a": gen(ctx + "\n" + q)} for q in qs])
    log["batched"].append(gen(ctx + "\n" + " ".join(qs)))

with open(OUT, "w") as f:
    json.dump(log, f, indent=1)
print(f"saved {M * len(qs)} unbatched + {M} batched answers to {OUT}")
print("hand-grade each answer against the context facts: "
      "OpenAI / 175 billion / ~300 billion tokens")
