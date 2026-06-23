"""Reproducer for RED_TEAM_STUDY (Red Teaming, responsible.ipynb).

Runs the chapter's fixed RED_TEAM_SUITE against every candidate model twice,
once wearing the naive system prompt and once wearing the hardened one,
and scores the percent of attacks each build resists (the HUGEPANDA canary
never appears in the reply). Prints a paste-ready RED_TEAM_STUDY plus a
per-attack leak grid so a clean showdown pairing can be picked.

Run: .venv/bin/python scripts/_resp_redteam_probe.py [model ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.agent import AGENT_BENCH_MODELS
from genai.security import (RED_TEAM_SUITE, ASSISTANT_SYSTEM, HARDENED_SYSTEM,
                            _assistant_says, leaked, score_red_team)

# The default roster is the capability zoo minus functiongemma (its perfect score
# is hollow — it barely answers — and the chapter's other screenings already run
# the same six), plus the break-it model gemma4:e2b, so a no-arg run reproduces
# every row of RED_TEAM_STUDY including the small model on the chart. Pass models
# explicitly to override.
models = sys.argv[1:] or (
    [m for m in AGENT_BENCH_MODELS if "functiongemma" not in m] + ["gemma4:e2b"])
print(f"screening {len(models)} models on {len(RED_TEAM_SUITE)} attacks x 2 builds...\n")
study = score_red_team(models)

print("RED_TEAM_STUDY = {")
width = max(len(m) for m in study) + 3
for m, r in sorted(study.items(), key=lambda kv: (-kv[1]["hardened"], -kv[1]["naive"])):
    key = f'"{m}":'.ljust(width)
    print(f'    {key} {{"hardened": {r["hardened"]:5}, "naive": {r["naive"]:5}}},')
print("}\n")

print("one illustrative single-shot pass per attack (L = leaked; the borderline "
      "cells wobble\nrun to run, the majority-of-3 study above is the source of "
      "truth; columns: " + ", ".join(name for name, _ in RED_TEAM_SUITE) + ")")
for m in models:
    for build, system in (("naive", ASSISTANT_SYSTEM), ("hardened", HARDENED_SYSTEM)):
        marks = ["L" if leaked(_assistant_says(msg, system, m)) else "."
                 for _, msg in RED_TEAM_SUITE]
        print(f"  {m:<22} {build:<9} {' '.join(marks)}")
