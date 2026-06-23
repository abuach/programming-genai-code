"""Probe for tokens.ipynb edits: the strawberry big-vs-small contrast and a
chopped-digit arithmetic stumble. Run a few trials so the framing matches what
the models actually do (no invented numbers).

    .venv/bin/python scripts/_tokens_probe.py
"""
import sys
sys.path.insert(0, "chapters")
from genai import ask, tokenize

SMALL, BIG = "llama3.2:1b", "gemma4:latest"

print("=" * 60)
print("STRAWBERRY: how many R's? (true answer: 3)")
print("=" * 60)
q = "How many letter R's are in 'strawberry'? Reply with only the number."
for model in [SMALL, BIG]:
    answers = [ask(q, model=model, max_tokens=8).strip() for _ in range(5)]
    print(f"{model:16} -> {answers}")
print("both ever see:", tokenize("strawberry"))

print()
print("=" * 60)
print("ARITHMETIC: chopped digits, exact product")
print("=" * 60)
for a, b in [(3947, 6281), (1234, 5678), (888, 999)]:
    truth = a * b
    for model in [SMALL, BIG]:
        reply = ask(f"What is {a} * {b}? Reply with only the number.",
                    model=model, max_tokens=16).strip()
        ok = str(truth) in reply.replace(",", "")
        print(f"{a}*{b}={truth:<10} {model:16} -> {reply!r:20} {'OK' if ok else 'WRONG'}")
    print(f"   {a} tokenizes as {tokenize(str(a))}, {b} as {tokenize(str(b))}")
