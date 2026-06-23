"""Reproducer for LEAK_GALLERY (the verbatim leak gallery, Red Teaming section,
responsible.ipynb).

Re-runs each model in the gallery against its attack on the naive assistant,
several trials, and reports how often it leaks plus a fresh sample reply. The chapter
bakes one captured reply per model because sampling is nondeterministic (a couple
of these attacks leak only ~2 tries in 3); this script is the live evidence that
the baked replies are real and reproducible.

Run: .venv/bin/python scripts/_resp_gallery_probe.py [trials]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.security import (_assistant_says, leaked, ASSISTANT_SYSTEM, LEAK_GALLERY,
                            REPEAT_ATTACK, PREFIX_JAILBREAK, THIRD_PERSON_ATTACK,
                            AUDIT_ATTACK, TRANSLATE_ATTACK, DIRECT_ASK)

ATTACKS = {"sysadmin-audit": AUDIT_ATTACK, "prefix-inject": PREFIX_JAILBREAK,
           "repeat-prompt": REPEAT_ATTACK, "third-person": THIRD_PERSON_ATTACK,
           "translate-french": TRANSLATE_ATTACK, "direct-ask": DIRECT_ASK}
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

for model, attack, _baked in LEAK_GALLERY:
    replies = [_assistant_says(ATTACKS[attack], ASSISTANT_SYSTEM, model) for _ in range(TRIALS)]
    n = sum(leaked(r) for r in replies)
    sample = next((r for r in replies if leaked(r)), replies[0])
    print(f"\n{model:22} {attack:16} leak {n}/{TRIALS}")
    print("    paste-ready full verbatim reply for LEAK_GALLERY:")
    print(f"    ({model!r}, {attack!r},")
    print(f"     {sample!r}),")
