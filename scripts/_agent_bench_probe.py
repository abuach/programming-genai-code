"""Reproducer for AGENT_BENCH (the Audition, agentic.ipynb).

Scores every contestant in AGENT_BENCH_MODELS on the eight-prompt tool-calling
suite with the live grader (right tool + right args, partial credit) and prints
a paste-ready AGENT_BENCH dict plus a ranked board. This is the committed
harness behind the chapter's first bake-off: re-run it to refresh the numbers or
to add a model to the roster.

Run: .venv/bin/python scripts/_agent_bench_probe.py [model ...]
"""
import sys
sys.path.insert(0, "chapters")
from genai.agent import AGENT_BENCH_MODELS, score_tool_bench

models = sys.argv[1:] or AGENT_BENCH_MODELS
print(f"scoring {len(models)} models on the tool-calling suite...\n")
bench = score_tool_bench(models)

width = max(len(m) for m in bench) + 1
print("AGENT_BENCH = {")
for m, r in bench.items():
    key = f'"{m}":'.ljust(width + 2)
    print(f'    {key} {{"score_pct": {r["score_pct"]:5}, '
          f'"latency_s": {r["latency_s"]:.2f}}},')
print("}\n")

print("ranked:")
for m, r in sorted(bench.items(), key=lambda kv: -kv[1]["score_pct"]):
    bar = "#" * round(r["score_pct"] / 4)
    print(f"  {m:<22} {r['score_pct']:5.1f}%  {r['latency_s']:5.2f}s  {bar}")
