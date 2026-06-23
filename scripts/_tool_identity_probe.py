"""Factorial study: a tool's identity = its NAME x its DESCRIPTION.

Two factors, two levels each, crossed into a 2x2:

    name:         clear  lookup(term)              vs  vague  fn(x)
    description:  clear  "Look up a ... glossary."  vs  vague  "Returns a string ..."

For every model we measure the tool call-rate over a fixed term set in all four
cells (temperature 0, so the only variables are the name and the description).
Prints a per-model 2x2, an aggregate 2x2 with main effects and the interaction,
and a paste-ready TOOL_IDENTITY_STUDY. Regenerates genai.agent.TOOL_IDENTITY_STUDY.

Run: .venv/bin/python scripts/_tool_identity_probe.py [model ...]
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import (score_tool_identity, TOOL_IDENTITY_TERMS,
                         AGENT_BENCH_MODELS)

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"factorial: 2 names x 2 descriptions, {len(models)} models, "
      f"{len(TOOL_IDENTITY_TERMS)} terms, temp 0\n", flush=True)

study = score_tool_identity(models)


def grid(g, indent="  "):
    print(f"{indent}            clear desc   vague desc")
    print(f"{indent}  clear nm   {g['clear_clear']:7.1f}%   {g['clear_vague']:7.1f}%")
    print(f"{indent}  vague nm   {g['vague_clear']:7.1f}%   {g['vague_vague']:7.1f}%")


for m, g in study.items():
    print(m)
    grid(g)
    print(flush=True)


def mean(key):
    return sum(s[key] for s in study.values()) / len(study)


cc, cv, vc, vv = (mean("clear_clear"), mean("clear_vague"),
                  mean("vague_clear"), mean("vague_vague"))
print("AGGREGATE (mean call-rate across models)")
grid({"clear_clear": cc, "clear_vague": cv, "vague_clear": vc, "vague_vague": vv})
print(f"\n  main effect  name  (clear-vague): {(cc + cv)/2 - (vc + vv)/2:+.1f}")
print(f"  main effect  desc  (clear-vague): {(cc + vc)/2 - (cv + vv)/2:+.1f}")
print(f"  interaction  (desc drop | vague name) - (desc drop | clear name): "
      f"{(vc - vv) - (cc - cv):+.1f}\n")

print("TOOL_IDENTITY_STUDY = {")
for m, g in study.items():
    cells = ", ".join(f'"{k}": {g[k]:5}' for k in
                      ("clear_clear", "clear_vague", "vague_clear", "vague_vague"))
    print(f'    "{m}": {{{cells}}},')
print("}")
