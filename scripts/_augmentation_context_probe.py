"""Reproducer for CONTEXT_STUDY (lost-in-the-middle, augmentation.ipynb).

Scores each model's recall of a fact planted at the EDGES vs the MIDDLE of a long
context, averaged over several facts. The gap is the lost-in-the-middle effect.
Prints a paste-ready CONTEXT_STUDY and a ranked board (by middle recall).

Run: .venv/bin/python scripts/_augmentation_context_probe.py [model ...]
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import AGENT_BENCH_MODELS
from genai.rag import score_context

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"scoring lost-in-the-middle recall on {len(models)} models...\n")
study = score_context(models)

print("CONTEXT_STUDY = {")
for m, r in sorted(study.items(), key=lambda kv: -kv[1]["middle"]):
    key = f'"{m}":'.ljust(max(len(x) for x in study) + 3)
    print(f'    {key} {{"edges": {r["edges"]:5}, "middle": {r["middle"]:5}}},')
print("}\n")
print("ranked by middle recall:")
for m, r in sorted(study.items(), key=lambda kv: -kv[1]["middle"]):
    print(f"  {m:<22} edges {r['edges']:5.1f}%  middle {r['middle']:5.1f}%")
