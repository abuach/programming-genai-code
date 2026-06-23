"""Reproducer for the three reworked efficiency.ipynb demos.

1. "Matching Model to Task" (cell 81d2ca51): small (1B) vs large (3B) model
   of the same family across an easy->hard ladder of GPU-themed questions.
   We report generation SPEED (tokens/sec): a stable model property, flat
   across difficulty, with the 1B model the faster tier on every rung. The
   notebook capture (printed speeds + plot_task_ladder figure) is embedded
   by scripts/_ladder_capture.py.
2. "Quantization" answers (cell 37): the three precisions of Llama 3.2 1B
   should land on the same physics for a simple factual question.
3. "Specialized Models" (cell 26): a code specialist (qwen2.5-coder, 7.6B) vs
   a generalist (mistral, 7.2B) of the SAME size on a coding task. The stable
   finding is the tie in tokens/sec: at equal size, specialization is invisible
   to the clock. How many tokens each writes is sampling noise (it flips run to
   run), so the prose leans on the speed tie, not the token count.

Timing drifts run to run; the embedded notebook output is one frozen capture.
The qualitative shape (clean speed tiers, flat across difficulty; answers
agree; equal-size speed tie) is what the prose claims, and that is what this
probe checks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))

from genai import ask, time_call

SMALL, LARGE = "llama3.2:1b", "llama3.2:latest"
ladder = [("simple",   "What does GPU stand for?"),
          ("moderate", "Explain in one sentence why AI runs on GPUs."),
          ("complex",  "Compare running AI locally versus in the cloud.")]

# Warm both models so the first timed call isn't paying a load cost.
for m in (SMALL, LARGE):
    time_call("hi", model=m)

print("=== (1) Matching model to task: flat speed tiers up the ladder ===")
print(f"{'task':9s} {'1B t/s':>7} {'3B t/s':>7}")
for task, prompt in ladder:
    s = time_call(prompt, model=SMALL)
    b = time_call(prompt, model=LARGE)
    print(f"{task:9s} {s['tokens_per_sec']:7.0f} {b['tokens_per_sec']:7.0f}")

print()
print("=== (2) Quantization: do the three precisions agree? ===")
variants = ["llama3.2:1b-instruct-q4_K_M",
            "llama3.2:1b-instruct-q8_0",
            "llama3.2:1b-instruct-fp16"]
q = "Explain gravity in one sentence."
for v in variants:
    a = ask(q, model=v, max_tokens=40, options={"temperature": 0})
    print(f"{v.split('-')[-1]:8s} -> {a}")

print()
print("=== (3) Specialized models: specialist vs generalist at the same size ===")
coding = "Write a Python function to find prime numbers up to n."
pair = ["qwen2.5-coder:latest", "mistral:latest"]  # 7.6B specialist, 7.2B generalist
for m in pair:
    time_call("hi", model=m)  # warm
for m in pair:
    r = time_call(coding, model=m)
    print(f"{m:24s} {r['tokens_per_sec']:5.1f} t/s  {r['gen_tokens']:3d} tok")
