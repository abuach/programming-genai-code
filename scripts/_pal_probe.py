"""THROWAWAY probe: find an honest substrate for a Program-Aided LM (PAL) demo.

PAL {cite}`gao2022pal` (and the closely related Program-of-Thoughts
{cite}`chen2022pot`): instead of reasoning toward a number in prose, the model
writes a short Python program whose execution yields the answer, so the brittle
step (multi-step arithmetic) is offloaded from the model to a real interpreter.
It should BEAT prose chain-of-thought when the problem is easy to translate to
code but the arithmetic is gnarly enough that the model slips doing it in words,
and it CANNOT help when the model mis-models the problem (a wrong program runs
to a confidently wrong number) or the question is not really computational.

This probe measures, per (model, problem):
  - cot@M : fraction of prose chain-of-thought answers that land on the value
  - pal@M : fraction of (write-a-program then run it) answers that land

At M=1 (triage) it also prints the prose answer and the actual generated
program with the number the sandbox computed, so we can eyeball where the honest
gap lives before baking anything.

Run: .venv/bin/python scripts/_pal_probe.py [M] [model ...]
"""
import sys, re, ast, io, time, builtins, contextlib
sys.path.insert(0, "chapters")
import ollama
from genai.llm import SERVER

_client = ollama.Client(host=SERVER)

# Multi-step word problems with UNCOMMON numbers (off the memorized beaten path),
# each trivial to express as a few lines of arithmetic but awkward to grind out in
# the head. "Snail" is the honest-limit candidate: the naive program (7 / (4-3))
# runs to a confident 7, but the snail climbs out on day 4 before it can slide, so
# a mis-modeled program is wrong with full conviction. Each: (label, q, value, tol).
PROBLEMS = [
    ("Pages", "On the first day a student reads 3 pages. Each day after, she reads "
              "4 more pages than the day before. How many pages has she read in "
              "total after 30 days?", 1830, 0.5),
    ("Seats", "A theatre has 24 rows. The first row has 18 seats and each row after "
              "has 3 more seats than the one before. How many seats are there in "
              "total?", 1260, 0.5),
    ("Compound", "A town of 1200 people grows by 8 percent every year. How many "
                 "people are there after 10 years, rounded to the nearest whole "
                 "person?", 2591, 0.5),
    ("Iterate", "Start with the number 1. Double it and then add 3, and repeat that "
                "whole step 8 times in total. What number do you end with?", 1021, 0.5),
    ("Posts", "A straight fence is 84 metres long with a post every 7 metres, "
              "including a post at each end. How many posts are there?", 13, 0.5),
    ("Cuts", "It takes 4 minutes to saw through a log. A 30 metre log is cut into 6 "
             "equal pieces. How many minutes does the sawing take in total?", 20, 0.5),
    ("Wages", "A worker earns 18.50 dollars an hour and works 37 hours. Then 22 "
              "percent of the gross pay is withheld for taxes. What is the take-home "
              "pay in dollars?", 533.91, 0.05),
    ("Snail", "A snail is at the bottom of a 7 metre well. It climbs 4 metres each "
              "day and slides back 3 metres each night. How many days does it take "
              "to reach the top?", 4, 0.5),
]

# Prose chain of thought: reason in words, then commit to a parseable final line.
_COT = (" Think step by step in words, then end with a line exactly like "
        "'Answer: <number>'.")
# PAL: don't do the arithmetic, write a program that does. Reason in comments,
# stay import-free so it runs in the locked-down sandbox, and leave the result in
# a known variable so we can read it back.
_PAL = ("\n\nDo not do the arithmetic yourself. Write a short Python program that "
        "computes the answer. Reason in comments, use plain arithmetic and built-in "
        "functions only with no imports, and assign the final numeric answer to a "
        "variable named answer. Reply with only the program in one code block.")

# Built-ins a numeric program legitimately needs; everything that could touch the
# file system, network, or interpreter internals (open, eval, __import__, ...) is
# deliberately left out, so a hostile or buggy program cannot escape the sandbox.
_SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "min", "max", "round", "sum", "len", "int", "float", "range", "pow",
    "sorted", "enumerate", "zip", "map", "filter", "list", "dict", "set", "tuple",
    "bool", "str", "divmod", "print", "reversed", "all", "any", "True", "False",
    "None") if hasattr(builtins, n)}
_BANNED_CALLS = {"eval", "exec", "open", "compile", "input", "__import__",
                 "globals", "locals", "vars", "getattr", "setattr", "delattr"}


def run_program(code, result="answer"):
    """Run model-written code in a throwaway, screened namespace; return `result`."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed in the sandbox")
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in _BANNED_CALLS:
            raise ValueError(f"{node.func.id}() is not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed in the sandbox")
    ns = {"__builtins__": _SAFE_BUILTINS}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, "<pal>", "exec"), ns)
    if result not in ns:
        raise ValueError(f"the program never assigned `{result}`")
    return ns[result]


def _extract_code(reply):
    """Pull the program out of a chatty reply: first ```fenced``` block if any."""
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip()


_NUM = re.compile(r"(-?\d+\s*/\s*\d+|-?\d+\.?\d*)")


def _value(text):
    """The committed final number (after 'Answer:' if present) as a float."""
    tail = re.split(r"answer\s*[:=]", text, flags=re.I)[-1].replace(",", "").replace("$", "")
    found = _NUM.findall(tail)
    if not found:
        return None
    tok = found[-1].replace(" ", "")
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b) if float(b) else None
    return float(tok)


def _ask(prompt, model, temp, max_tokens):
    resp = _client.chat(model=model, think=False,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens, "temperature": temp})
    return (resp["message"]["content"] or "").strip()


def cot(question, model, temp=0.0):
    text = _ask(question + _COT, model, temp, 320)
    return text, _value(text)


def pal(question, model, temp=0.0):
    code = _extract_code(_ask(question + _PAL, model, temp, 320))
    try:
        return code, float(run_program(code))
    except Exception as e:
        return code, None


def _ok(v, target, tol):
    return v is not None and abs(v - target) <= tol


def triage(model):
    print(f"\n===== {model}  (triage M=1, temp=0) =====", flush=True)
    for label, q, tgt, tol in PROBLEMS:
        ctext, cv = cot(q, model)
        code, pv = pal(q, model)
        print(f"\n## {label}  (accept {tgt})")
        print(f"  COT  [{'OK ' if _ok(cv, tgt, tol) else 'XX '}] -> {cv}")
        print(f"  PAL  [{'OK ' if _ok(pv, tgt, tol) else 'XX '}] -> {pv}")
        print("  PROGRAM:")
        for ln in code.splitlines():
            print(f"      {ln}")


# The five baked into genai.pal.PAL_STUDY: three prose-hostile wins (a series, a
# second series, and compounding) plus two problems whose single tricky step the
# model already handles in its head, where PAL is redundant. Chosen from triage.
BAKE = ["Pages", "Seats", "Compound", "Snail", "Cuts"]


def bake(model, M, temp=0.7, only=BAKE):
    print(f"\n===== {model}  (M={M}/problem, temp={temp}) =====", flush=True)
    for label, q, tgt, tol in PROBLEMS:
        if only and label not in only:
            continue
        t0 = time.perf_counter()
        c = sum(_ok(cot(q, model, temp)[1], tgt, tol) for _ in range(M)) / M
        p = sum(_ok(pal(q, model, temp)[1], tgt, tol) for _ in range(M)) / M
        dt = time.perf_counter() - t0
        print(f"{label:9s} cot={c:4.0%}  pal={p:4.0%}  (gap {p-c:+4.0%})  {dt:5.1f}s",
              flush=True)


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    models = sys.argv[2:] or ["qwen2.5-coder:latest"]
    for mdl in models:
        if M == 1:
            triage(mdl)
        else:
            bake(mdl, M)
