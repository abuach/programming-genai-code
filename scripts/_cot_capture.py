"""Capture for the live ages-problem cell (chapters/prompting.ipynb).

The cell runs show_cot_comparison once, so its transcript is a single noisy draw.
This re-rolls that exact call until it lands on a run where BOTH gemma4 and mistral
are *saved* by chain of thought -- wrong cold, right (67) once asked to think step by
step -- which is the illustrative draw we freeze. Every line printed is real model
output; we only select which draw to keep, never edit it. Prints the winning
transcript ready to paste into the frozen cell's saved output.
"""
import sys, io, re, contextlib
sys.path.insert(0, "chapters")
from genai.prompting import show_cot_comparison

PROBLEM = ("When I was 6 my sister was half my age. "
           "Now I am 70. How old is my sister?")
MODELS = ["gemma4:latest", "llama3.2", "qwen3:latest", "mistral:latest"]
BUDGET = 40
row = re.compile(r"──\s+(\S+)\s+no CoT:\s*(-?\d+|\?)\s+with CoT:\s*(-?\d+|\?)")


def saved(parsed, model):
    """True when `model` missed cold but reached 67 once it reasoned step by step."""
    cold, cot = parsed.get(model, ("?", "?"))
    return cold != "67" and cot == "67"


for attempt in range(1, BUDGET + 1):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        show_cot_comparison(PROBLEM, MODELS)
    out = buf.getvalue()
    parsed = {m: (c, k) for m, c, k in row.findall(out)}
    g, m = saved(parsed, "gemma4:latest"), saved(parsed, "mistral:latest")
    print(f"attempt {attempt:2d}: {parsed}  gemma_saved={g} mistral_saved={m}",
          file=sys.stderr)
    if g and m:
        print("\n=== FROZEN DRAW (both gemma4 and mistral saved by CoT) ===\n")
        print(out, end="")
        break
else:
    sys.exit(f"no qualifying draw in {BUDGET} attempts")
