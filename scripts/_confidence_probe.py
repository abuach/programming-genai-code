"""Reproducer for CONFIDENCE_STUDY (Self-Refine / calibration, agentic.ipynb).

For each contestant, mean self-confidence on answerable vs impossible questions.
The gap (answerable minus impossible) is calibration: a big gap means the model
knows when it is guessing. Prints a paste-ready CONFIDENCE_STUDY and a board
ranked by that gap.

Run: .venv/bin/python scripts/_confidence_probe.py [model ...]
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import AGENT_BENCH_MODELS, score_confidence

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"probing calibration on {len(models)} models...\n")
study = score_confidence(models)

print("CONFIDENCE_STUDY = {")
for m, r in study.items():
    key = f'"{m}":'.ljust(max(len(x) for x in study) + 3)
    print(f'    {key} {{"answerable": {r["answerable"]:.2f}, '
          f'"impossible": {r["impossible"]:.2f}}},')
print("}\n")

print("ranked by calibration gap (answerable - impossible):")
for m, r in sorted(study.items(), key=lambda kv: -(kv[1]["answerable"] - kv[1]["impossible"])):
    gap = r["answerable"] - r["impossible"]
    print(f"  {m:<22} answerable {r['answerable']:.2f}  "
          f"impossible {r['impossible']:.2f}  (gap {gap:+.2f})")
