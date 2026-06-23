"""Reproducer for PII_SCREEN_STUDY (PII handling screening, responsible.ipynb).

Each candidate model summarizes the synthetic PII_SCREEN_NOTES twice, once
politely asked to omit identifiers and once not asked, and is scored on the
percent of summaries containing zero PII matches, graded by the very
PII_PATTERNS regexes that `redact` uses. Prints a paste-ready PII_SCREEN_STUDY
and dumps any leaking summaries so the failures can be eyeballed.

Run: .venv/bin/python scripts/_resp_pii_probe.py [model ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.agent import AGENT_BENCH_MODELS
from genai.llm import ask
from genai.security import (PII_SCREEN_NOTES, PII_SCREEN_ASK, PII_SCREEN_BASELINE,
                            _pii_free, score_pii_handling)

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"screening {len(models)} models on {len(PII_SCREEN_NOTES)} notes x 2 prompts...\n")
study = score_pii_handling(models)

print("PII_SCREEN_STUDY = {")
width = max(len(m) for m in study) + 3
for m, r in sorted(study.items(), key=lambda kv: (-kv[1]["instructed"], -kv[1]["unprompted"])):
    key = f'"{m}":'.ljust(width)
    print(f'    {key} {{"instructed": {r["instructed"]:5}, "unprompted": {r["unprompted"]:5}}},')
print("}\n")

print("leaking summaries (instructed condition only):")
for m in models:
    for note in PII_SCREEN_NOTES:
        summary = ask(f"{PII_SCREEN_ASK}\n\nNote:\n{note}", model=m,
                      max_tokens=70, think=False)
        if not _pii_free(summary):
            print(f"  {m}: {summary[:90]!r}")
