"""Reproducer for the "Are smaller or simpler models safer?" admonition
(Prompt Injection and Jailbreaking, responsible.ipynb).

Runs the blunt, prefix, and cross-lingual attacks against the small models the
box names, wearing the naive system prompt, several trials each, and prints the
leak counts plus a sample reply per cell so every claim can be checked: the tiny
qwen3.5:0.8b leaks the blunt ask with no trick, gemma4:e2b refuses the blunt ask
but falls to the prefix and the French translation every time, and the 1B models
are spared the French attack for opposite reasons (llama3.2:1b refuses outright;
gemma3:1b is willing but garbles the canary it can't reproduce).

Run: .venv/bin/python scripts/_resp_admonition_probe.py [trials]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.security import (_assistant_says, leaked, ASSISTANT_SYSTEM,
                            DIRECT_ASK, PREFIX_JAILBREAK, TRANSLATE_ATTACK)

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ATTACKS = [("direct", DIRECT_ASK), ("prefix", PREFIX_JAILBREAK),
           ("french", TRANSLATE_ATTACK)]
MODELS = ["gemma4:e2b", "gemma3:1b", "llama3.2:1b", "qwen3.5:0.8b"]

print(f"leak count out of {TRIALS} trials (naive assistant):")
print(f"{'model':14} {'direct':>8} {'prefix':>8} {'french':>8}")
for m in MODELS:
    counts = [f"{sum(leaked(_assistant_says(msg, ASSISTANT_SYSTEM, m)) for _ in range(TRIALS))}/{TRIALS}"
              for _, msg in ATTACKS]
    print(f"{m:14} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8}")

print("\nsample replies (one shot each, for honesty):")
for m in MODELS:
    for name, msg in ATTACKS:
        reply = _assistant_says(msg, ASSISTANT_SYSTEM, m)
        tag = "LEAK " if leaked(reply) else "block"
        print(f"  [{tag}] {m:14} {name:7} {reply[:80]!r}")
