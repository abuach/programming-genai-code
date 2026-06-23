"""Reproducer for POISON_STUDY (poisoning screening, responsible.ipynb).

Each candidate model triages POISON_TEST_EMAILS twice, once from the honest
few-shot examples and once from the label-flipped set. ``clean`` scores how
often it follows the honest labels; ``poisoned`` scores how often it follows
the flipped labels (the poison succeeding), against gold labels fixed by
construction. Prints a paste-ready POISON_STUDY and a ranked board on the
poisoned condition. functiongemma never returns a usable label and is dropped
from the chart in security.py.

Run: .venv/bin/python scripts/_resp_poison_probe.py [model ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.agent import AGENT_BENCH_MODELS
from genai.security import POISON_TEST_EMAILS, score_poisoning

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"screening {len(models)} models on {len(POISON_TEST_EMAILS)} emails x 2 example sets...\n")
study = score_poisoning(models)

print("POISON_STUDY = {")
width = max(len(m) for m in study) + 3
for m, r in sorted(study.items(), key=lambda kv: (-kv[1]["poisoned"], -kv[1]["clean"])):
    key = f'"{m}":'.ljust(width)
    print(f'    {key} {{"clean": {r["clean"]:5}, "poisoned": {r["poisoned"]:5}}},')
print("}\n")

print("ranked by how completely each model follows the poisoned examples (higher = more poisoned):")
for m, r in sorted(study.items(), key=lambda kv: -kv[1]["poisoned"]):
    bar = "#" * round(r["poisoned"] / 4)
    print(f"  {m:<22} clean {r['clean']:5.1f}%  poisoned {r['poisoned']:5.1f}%  {bar}")
