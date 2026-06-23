"""Reproducer for SAFETY_JUDGE_STUDY (moderation judge screening, responsible.ipynb).

Each candidate model sits in the judge chair of `moderate` and rules SAFE or
UNSAFE on the fixed SAFETY_JUDGE_SET (gold verdicts by construction: four
clearly safe clinical replies, four quietly dangerous ones that no keyword
rule catches). Scores percent of unsafe drafts flagged and safe drafts
cleared. Prints a paste-ready SAFETY_JUDGE_STUDY plus each judge's raw calls.

Run: .venv/bin/python scripts/_resp_judge_probe.py [model ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.agent import AGENT_BENCH_MODELS
from genai.security import SAFETY_JUDGE_SET, judge_verdict, score_safety_judge

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"screening {len(models)} judges on {len(SAFETY_JUDGE_SET)} drafts...\n")
study = score_safety_judge(models)

print("SAFETY_JUDGE_STUDY = {")
width = max(len(m) for m in study) + 3
for m, r in sorted(study.items(), key=lambda kv: (-kv[1]["flags"], -kv[1]["clears"])):
    key = f'"{m}":'.ljust(width)
    print(f'    {key} {{"flags": {r["flags"]:5}, "clears": {r["clears"]:5}}},')
print("}\n")

print("raw verdicts (x = wrong call):")
for m in models:
    marks = []
    for text, gold in SAFETY_JUDGE_SET:
        v = judge_verdict(text, m)
        marks.append(v[0] if v == gold else "x")
    print(f"  {m:<22} {' '.join(marks)}")
