"""Capture the phi4 vs phi4-reasoning accuracy contrast for the Thinking chapter.

The point of the demo: the existing snap-vs-think chart compares two DIFFERENT
models (qwen2.5-coder vs qwen3:4b), so the gap could be model quality rather than
thinking. Phi-4 controls for that: phi4:14b is Microsoft's base model, and
phi4-reasoning:14b is the SAME base fine-tuned on reasoning traces. Holding the
base fixed isolates what the reasoning post-training buys.

Grading note: phi4-reasoning writes its reasoning as <think>...</think> INSIDE the
content (Ollama does not split it into the thinking field), so a naive substring
grader matches stray digits from the trace and inflates the score. We strip the
think block and grade only the final answer.

Run: uv run python scripts/_phi4_reasoning_capture.py [runs]
"""
import re, sys, time, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
import ollama
from genai.thinking import NOVEL_PROBLEMS

client = ollama.Client(host="http://localhost:11434")
BASE, REASON = "phi4:14b", "phi4-reasoning:14b"
SNAP = " Reply with ONLY the final number, no explanation, no units."
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def last_int(s):
    nums = re.findall(r"-?\d+", s)
    return nums[-1] if nums else ""


def base_answer(q):
    """phi4 base, a non-thinking model: a bare number, no think param."""
    r = client.chat(model=BASE,
                    messages=[{"role": "user", "content": q + SNAP}],
                    options={"num_predict": 60})
    return (r["message"]["content"] or "").strip()


def reason_answer(q, budget=3500):
    """phi4-reasoning: returns (final_answer, think_closed, raw_tail).

    Ollama rejects think=True for this model; its chat template emits
    <think>...</think> inline in the content, which we strip before grading.
    """
    r = client.chat(model=REASON,
                    messages=[{"role": "user", "content": q + SNAP}],
                    options={"num_predict": 80 + budget})
    content = r["message"]["content"] or ""
    closed = "</think>" in content.lower()
    tail = re.split(r"</think>", content, flags=re.I)[-1] if closed else content
    return last_int(tail), closed, tail.strip()[-90:]


labels, base_acc, reason_acc = [], [], []
for label, q, gold in NOVEL_PROBLEMS:
    bhits = rhits = 0
    for r in range(RUNS):
        b = last_int(base_answer(q))
        rf, closed, tail = reason_answer(q)
        bhits += (b in gold)
        rhits += (rf in gold)
        print(f"{label:11} run{r+1}  base={b:>4} {'OK' if b in gold else 'X':<2}  "
              f"reason={rf:>4} {'OK' if rf in gold else 'X':<2} "
              f"closed={closed}  gold={gold[0]}  tail={tail!r}", flush=True)
    labels.append(label); base_acc.append(bhits / RUNS); reason_acc.append(rhits / RUNS)
    subprocess.run(["ollama", "stop", REASON], capture_output=True)

print("\nPHI4_REASONING_STUDY = {")
print(f'    "labels": {labels},')
print(f'    "base": {[round(a, 3) for a in base_acc]},')
print(f'    "reasoning": {[round(a, 3) for a in reason_acc]},')
print(f'    "runs": {RUNS},')
print("}")
