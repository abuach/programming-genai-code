"""Reproducer for SKILL_REUSE_STUDY (A Skill Library, agentic.ipynb).

Scores each contestant on composition: given a working helper and a task that is
easiest if it reuses that helper, what percent of the time does the new function
both pass its hidden tests AND call the helper instead of reinventing it? Prints
a paste-ready SKILL_REUSE_STUDY and a ranked board.

Run: .venv/bin/python scripts/_skill_reuse_probe.py [model ...]
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import SKILL_ROSTER, score_skill_reuse

models = sys.argv[1:] or SKILL_ROSTER
print(f"scoring {len(models)} models on skill reuse...\n")
study = score_skill_reuse(models)

print("SKILL_REUSE_STUDY = {")
for m, r in sorted(study.items(), key=lambda kv: -kv[1]["reuse_pct"]):
    key = f'"{m}":'.ljust(max(len(x) for x in study) + 3)
    print(f'    {key} {{"reuse_pct": {r["reuse_pct"]:5}}},')
print("}\n")

print("ranked:")
for m, r in sorted(study.items(), key=lambda kv: -kv[1]["reuse_pct"]):
    bar = "#" * round(r["reuse_pct"] / 4)
    print(f"  {m:<22} {r['reuse_pct']:5.1f}%  {bar}")
