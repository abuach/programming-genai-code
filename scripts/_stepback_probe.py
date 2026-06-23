"""THROWAWAY probe: find an honest substrate for a Step-Back prompting demo.

Step-back prompting {cite}`zheng2023stepback`: instead of answering a hard
question cold, the model first "steps back" to the general principle, law, or
formula that governs the problem (abstraction), then applies that principle to
answer. It should BEAT a cold answer when the model KNOWS the right principle
but fumbles applying it directly (a snap answer trips on the intuitive-but-wrong
shortcut), and it CANNOT help when the model does not know the principle either.

This probe measures, per (model, problem):
  - direct@M : fraction of cold snap answers that land on the accepted value
  - step@M   : fraction of (state-the-principle then apply) answers that land

At M=1 (triage) it also prints the principle the model produced and both
answers, so we can eyeball where the honest gap lives before baking anything.

Run: .venv/bin/python scripts/_stepback_probe.py [M] [model ...]
"""
import sys, re, time
sys.path.insert(0, "chapters")
import ollama
from genai.llm import SERVER

_client = ollama.Client(host=SERVER)

# Principle-governed problems with a one- or two-step solve and a clean numeric
# answer. The snap shortcut is the wrong intuition (linear when the law is a
# square root, "add" when percentages multiply, and so on). Each entry:
#   (label, question, accepted_value, tolerance)
PROBLEMS = [
    ("Combined", "A gas occupies 6 litres at 1 atm and 300 K. It is heated to 600 K "
                 "and its pressure rises to 3 atm. What is its new volume, in litres?",
     4.0, 0.01),
    ("Submerged", "A wooden block floats in water with 90 percent of its volume below "
                  "the surface. Water has a density of 1000 kg/m3. What is the density "
                  "of the block, in kg/m3?", 900.0, 1.0),
    ("Markup", "A jacket is marked down by 20 percent for a sale. The next week the "
               "sale price is raised by 20 percent. The final price is what percent of "
               "the original price?", 96.0, 0.5),
    ("Inverse", "A lamp lights a page 1 metre away with a certain brightness. If the "
                "page is moved to 4 metres away, the brightness becomes what fraction "
                "of the original?", 1/16, 0.005),
    ("Half", "A medicine has a half-life of 4 hours in the blood. After 12 hours, what "
             "fraction of the original dose remains?", 0.125, 0.005),
    ("Force", "A force gives a 2 kg cart an acceleration of 12 m/s2. The same force is "
              "applied to an 8 kg cart. What is its acceleration, in m/s2?", 3.0, 0.01),
    ("Resist", "Two 6-ohm resistors are wired in parallel, and that pair is wired in "
               "series with a third 6-ohm resistor. What is the total resistance, in "
               "ohms?", 9.0, 0.01),
    ("Pendulum", "A pendulum has a period of 3 seconds. Its length is increased to 4 "
                 "times its original length. What is the new period, in seconds?",
     6.0, 0.05),
    ("Pressure", "A sealed rigid tank of gas is at 2 atm and 250 K. It is cooled to "
                 "125 K. What is the new pressure, in atm?", 1.0, 0.01),
    ("Triple", "A bacterial colony triples every hour. It starts at 100 cells. How "
               "many cells are there after 3 hours?", 2700.0, 1.0),
]

# Cold snap: no room to reason, commit one number. This is the "asked cold"
# baseline -- the intuitive System-1 answer, lure and all.
_DIRECT = " Answer with only a single number. No words, no working, no units."
# Step back: first abstract to the governing law, then reason FROM it. The
# Answer: sentinel makes the final value parseable after the working.
_PRINCIPLE = (" Do not solve it yet. In one sentence, name the general principle, "
              "law, or formula that governs this problem.")
_APPLY = ("\n\nUse that principle to solve the problem. Explain in two or three "
          "short sentences, plain words and simple arithmetic, no equations or "
          "LaTeX. Then end with a line exactly like 'Answer: <number>'.")

_NUM = re.compile(r"(-?\d+\s*/\s*\d+|-?\d+\.?\d*)")


def _value(text: str):
    """The committed final number (after 'Answer:' if present) as a float."""
    tail = re.split(r"answer\s*[:=]", text, flags=re.I)
    found = _NUM.findall(tail[-1])
    if not found:
        return None
    tok = found[-1].replace(" ", "")
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b) if float(b) else None
    return float(tok)


def _ok(text, target, tol):
    v = _value(text)
    return v is not None and abs(v - target) <= tol


def _ask(prompt, model, temp, max_tokens):
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens, "temperature": temp})
    return (resp["message"]["content"] or "").strip()


def direct(question, model, temp=0.0):
    return _ask(question + _DIRECT, model, temp, 16)


def step_back(question, model, temp=0.0):
    principle = _ask(question + _PRINCIPLE, model, temp, 90)
    apply = _ask(f"General principle: {principle}\n\nProblem: {question}{_APPLY}",
                 model, temp, 200)
    return principle, apply


def triage(model):
    print(f"\n===== {model}  (triage M=1, temp=0) =====", flush=True)
    for label, q, tgt, tol in PROBLEMS:
        d = direct(q, model)
        principle, a = step_back(q, model)
        print(f"\n## {label}  (accept {tgt})")
        print(f"  DIRECT     [{'OK ' if _ok(d, tgt, tol) else 'XX '}] {d!r}")
        print(f"  PRINCIPLE  {principle[:160]!r}")
        print(f"  STEP-BACK  [{'OK ' if _ok(a, tgt, tol) else 'XX '}] {a!r}")


# The curated subset baked into genai.prompting.STEP_BACK_STUDY: one problem per
# honest regime (snap-already-right ceiling, snap-wrong-principle-fixes win,
# snap-wrong-principle-still-fumbles floor), chosen from the triage above.
BAKE = ["Half", "Pendulum", "Inverse", "Triple", "Pressure"]


def bake(model, M, temp=0.7):
    print(f"\n===== {model}  (M={M} samples/problem, temp={temp}) =====", flush=True)
    for label, q, tgt, tol in [p for p in PROBLEMS if p[0] in BAKE]:
        t0 = time.perf_counter()
        d = sum(_ok(direct(q, model, temp), tgt, tol) for _ in range(M)) / M
        s = sum(_ok(step_back(q, model, temp)[1], tgt, tol) for _ in range(M)) / M
        dt = time.perf_counter() - t0
        print(f"{label:9s} direct={d:4.0%}  step-back={s:4.0%}  "
              f"(gap {s-d:+4.0%})  {dt:4.1f}s", flush=True)


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    models = sys.argv[2:] or ["gemma4:latest"]
    for mdl in models:
        if M == 1:
            triage(mdl)
        else:
            bake(mdl, M)
