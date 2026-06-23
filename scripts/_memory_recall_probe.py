"""Reproducer for MEMORY_RECALL_STUDY (Memory and World State, agentic.ipynb).

Scores every contestant as a semantic librarian: given a fixed pile of student
notes and a set of paraphrased queries (each pointing at one note in different
words), what percent does the model recall correctly by meaning? Prints a
paste-ready MEMORY_RECALL_STUDY and a ranked board.

Run: .venv/bin/python scripts/_memory_recall_probe.py [model ...]
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import AGENT_BENCH_MODELS, score_memory_recall

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"scoring {len(models)} models as semantic librarians...\n")
study = score_memory_recall(models)

print("MEMORY_RECALL_STUDY = {")
for m, r in sorted(study.items(), key=lambda kv: -kv[1]["recall_pct"]):
    key = f'"{m}":'.ljust(max(len(x) for x in study) + 3)
    print(f'    {key} {{"recall_pct": {r["recall_pct"]:5}}},')
print("}\n")

print("ranked:")
for m, r in sorted(study.items(), key=lambda kv: -kv[1]["recall_pct"]):
    bar = "#" * round(r["recall_pct"] / 4)
    print(f"  {m:<22} {r['recall_pct']:5.1f}%  {bar}")
