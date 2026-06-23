"""Reproducer for REFLEXION_STUDY (Escaping the Maze, agentic.ipynb).

Runs the gotcha suite for a roster of models, each two ways: reflexion (verbal
lessons carried across attempts) and blind retry (resample, no memory), over K
trials. Captures each model's FINAL pass-rate both ways, so the chart can show
which models actually benefit from reflection and which ceiling out (already ace
it) or stay stuck (too weak to write a useful lesson). Reuses the gotcha suite +
run_condition from _reflexion_gotcha / _reflexion_proto.

Run: .venv/bin/python scripts/_reflexion_bakeoff.py [repeats] [model ...]
Pass model names after the repeat count to score only those (e.g. to add one
new contestant's row without re-running the whole field).
"""
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "chapters")
import _reflexion_proto as P
import _reflexion_gotcha  # noqa: F401  (its import sets P.SUITE = GOTCHA_SUITE)

ROSTER = ["mistral:latest", "qwen3:latest", "gpt-oss:20b",
          "qwen2.5-coder:latest", "gemma4:latest", "phi4:14b", "lfm2:24b",
          "glm-4.7-flash:latest", "llama3.1:latest", "codellama:latest",
          "openhermes:v2.5", "ministral-3:latest", "deepseek-coder:latest",
          "tinyllama:1.1b", "qwen3.5:latest", "functiongemma:latest"]
K = 4
repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if len(sys.argv) > 2:                 # optional explicit model list
    ROSTER = sys.argv[2:]
P.temp = 0.7

print(f"reflexion bake-off  K={K} repeats={repeats} tasks={len(P.SUITE)}\n",
      flush=True)
study = {}
for model in ROSTER:
    refl = P.run_condition(True, model, K, 0.7, repeats)
    blind = P.run_condition(False, model, K, 0.7, repeats)
    study[model] = {"blind": round(100 * blind[-1], 1),
                    "reflexion": round(100 * refl[-1], 1)}
    lift = study[model]["reflexion"] - study[model]["blind"]
    print(f"  {model:<22} blind {study[model]['blind']:5.1f}%  "
          f"reflexion {study[model]['reflexion']:5.1f}%  (lift {lift:+.1f})",
          flush=True)

print("\nREFLEXION_STUDY = {")
for m, r in study.items():
    key = f'"{m}":'.ljust(max(len(x) for x in study) + 3)
    print(f'    {key} {{"blind": {r["blind"]:5}, "reflexion": {r["reflexion"]:5}}},')
print("}")
