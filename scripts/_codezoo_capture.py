"""Capture the metacoding code-zoo scorecard: every local code model graded
across a suite of small tasks, grouped into category columns, so capability
shows up as a profile with real gradient instead of a wall of 0s and 100s.

Each square of the scorecard is a CATEGORY holding three small tasks (two or
three hidden cases each). A model that solves two of the three colours the
square two-thirds green, which is the whole point: fewer samples per task, more
tasks, so the cell can land anywhere between fail and ace.

  Instruction columns (chat: write a whole function from a spec)
      strings : reverse_words, roman_to_int (subtractive trap), is_palindrome
      math    : is_prime, factorial, digit_sum
      lists   : flatten (deep nesting), median (even-length), chunk
  Fill-in-the-Middle columns (raw, each family's own sentinel tokens)
      fill-loop  : gcd, sum_squares, count_vowels   (hole in a loop body)
      fill-build : dedup, char_count, running_max    (hole that grows a result)

FIM only works for FIM-trained families; Magicoder (Evol-Instruct) has no FIM
tokens, so its FIM cells are N/A, not 0. FIM completions are clipped to the hole
(lines at or deeper than the hole's indent), the way an editor keeps only the
insertion. DeepCoder is dropped from the book scorecard (it scored near zero on
everything); it is omitted here too.

Writes chapters/genai/code_zoo.json (verbatim outputs for baking) and prints the
grid.  Run:  .venv/bin/python scripts/_codezoo_capture.py
"""
import io
import json
import re
from contextlib import redirect_stdout
import ollama

client = ollama.Client(host="http://localhost:11434")

MODELS = [  # (label, ollama id, fim_format | None)
    ("Qwen2.5-Coder",  "qwen2.5-coder:latest",  "qwen"),
    ("CodeQwen",       "codeqwen:7b",           "qwen"),
    ("OpenCoder",      "opencoder:8b",          "star"),
    ("Magicoder",      "magicoder:7b",          None),
    ("DeepSeek-Coder", "deepseek-coder:latest", "deepseek"),
    ("CodeLlama",      "codellama:latest",      "codellama"),
    ("Stable-Code",    "stable-code:3b",        "star"),
    ("StarCoder2",     "starcoder2:latest",     "star"),
]

W = "Only code, no explanation."

# (key, column, fn, prompt, cases)
INSTR = [
    ("reverse_words", "strings", "reverse_words",
     f"Write a Python function reverse_words(s) that reverses the order of the "
     f"words in a string. {W}",
     [(("hello world",), "world hello"), (("a b c",), "c b a")]),
    ("roman", "strings", "roman_to_int",
     f"Write a Python function roman_to_int(s) that converts a Roman numeral "
     f"string to an integer. {W}",
     [(("III",), 3), (("IX",), 9), (("MCMXCIV",), 1994)]),
    ("palindrome", "strings", "is_palindrome",
     f"Write a Python function is_palindrome(s) that returns True if s is a "
     f"palindrome, ignoring case, spaces and punctuation. {W}",
     [(("A man, a plan, a canal: Panama",), True), (("race a car",), False)]),
    ("is_prime", "math", "is_prime",
     f"Write a Python function is_prime(n) that returns True if n is prime. {W}",
     [((2,), True), ((7,), True), ((9,), False), ((1,), False)]),
    ("factorial", "math", "factorial",
     f"Write a Python function factorial(n) that returns n factorial. {W}",
     [((0,), 1), ((5,), 120)]),
    ("digit_sum", "math", "digit_sum",
     f"Write a Python function digit_sum(n) that returns the sum of the decimal "
     f"digits of a non-negative integer n. {W}",
     [((123,), 6), ((9009,), 18)]),
    ("flatten", "lists", "flatten",
     f"Write a Python function flatten(lst) that flattens an arbitrarily nested "
     f"list into one flat list. {W}",
     [(([[1, 2], [3, [4]]],), [1, 2, 3, 4]), (([1, [2, [3, [4]]]],), [1, 2, 3, 4]),
      (([],), [])]),
    ("median", "lists", "median",
     f"Write a Python function median(nums) that returns the median of a list of "
     f"numbers. {W}",
     [(([3, 1, 2],), 2), (([1, 2, 3, 4],), 2.5)]),
    ("chunk", "lists", "chunk",
     f"Write a Python function chunk(lst, n) that splits lst into consecutive "
     f"sublists of length n (the last may be shorter). {W}",
     [(([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]), (([1, 2, 3], 1), [[1], [2], [3]])]),
]

# (key, column, fn, prefix, suffix, cases)
FIM = [
    ("gcd", "fill-loop", "gcd",
     "def gcd(a, b):\n    while b:\n        ", "\n    return a\n",
     [((48, 18), 6), ((7, 5), 1), ((1071, 462), 21)]),
    ("sum_squares", "fill-loop", "sum_squares",
     "def sum_squares(nums):\n    total = 0\n    for x in nums:\n        ",
     "\n    return total\n",
     [(([1, 2, 3],), 14), (([],), 0), (([4],), 16)]),
    ("count_vowels", "fill-loop", "count_vowels",
     "def count_vowels(s):\n    n = 0\n    for ch in s:\n        if ch in 'aeiou':\n            ",
     "\n    return n\n",
     [(("hello",), 2), (("xyz",), 0), (("aeiou",), 5)]),
    ("dedup", "fill-build", "dedup",
     "def dedup(xs):\n    seen = set()\n    out = []\n    for x in xs:\n"
     "        if x not in seen:\n            ", "\n    return out\n",
     [(([1, 1, 2, 3, 3, 2],), [1, 2, 3]), (([],), []), ((["a", "b", "a"],), ["a", "b"])]),
    ("char_count", "fill-build", "char_count",
     "def char_count(s):\n    counts = {}\n    for ch in s:\n        ",
     "\n    return counts\n",
     [(("aab",), {"a": 2, "b": 1}), (("",), {})]),
    ("running_max", "fill-build", "running_max",
     "def running_max(nums):\n    out = []\n    best = float('-inf')\n    for x in nums:\n"
     "        best = max(best, x)\n        ", "\n    return out\n",
     [(([1, 3, 2],), [1, 3, 3]), (([5, 4],), [5, 5])]),
]

FIM_TMPL = {
    "star":      lambda p, s: f"<fim_prefix>{p}<fim_suffix>{s}<fim_middle>",
    "qwen":      lambda p, s: f"<|fim_prefix|>{p}<|fim_suffix|>{s}<|fim_middle|>",
    "codellama": lambda p, s: f"<PRE> {p} <SUF>{s} <MID>",
    "deepseek":  lambda p, s: f"<｜fim▁begin｜>{p}<｜fim▁hole｜>{s}<｜fim▁end｜>",
}
FIM_STOPS = ["<|endoftext|>", "<|fim_pad|>", "<fim_pad>", "<file_sep>",
             "<|file_sep|>", "<EOT>", "<|EOT|>", "</s>", "<｜end▁of▁sentence｜>"]


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def hole_indent(prefix):
    return len(prefix) - len(prefix.rstrip(" "))


def clip_middle(middle, indent):
    lines = middle.split("\n")
    kept = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "":
            kept.append(ln)
        elif len(ln) - len(ln.lstrip(" ")) >= indent:
            kept.append(ln)
        else:
            break
    return "\n".join(kept).rstrip()


def _eq(got, want):
    try:
        return abs(got - want) < 1e-9 if isinstance(want, float) else got == want
    except Exception:
        return False


def run_cases(src, fn, cases):
    try:
        ns = {}
        with redirect_stdout(io.StringIO()):
            exec(src, ns)
        f = ns[fn]
        with redirect_stdout(io.StringIO()):
            passed = sum(_eq(f(*a), w) for a, w in cases)
    except Exception as e:
        return 0, type(e).__name__
    return passed, "all" if passed == len(cases) else "partial"


def gen_instruct(model, prompt):
    return client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0.0, "num_predict": 384}
                       )["message"]["content"]


def gen_fim(model, fmt, prefix, suffix):
    return client.generate(model=model, prompt=FIM_TMPL[fmt](prefix, suffix), raw=True,
                           options={"temperature": 0.0, "num_predict": 48,
                                    "stop": FIM_STOPS})["response"]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # generated code may raise SyntaxWarnings

    # captures: the gradeable artifact per cell (function source / clipped middle /
    #           None) -> baked into code_zoo.json for the book.
    # archive : the FULL story per cell, including the raw model output for every
    #           failure, so nothing a model actually produced is thrown away
    #           -> code_zoo_raw.json, for honest walkthroughs and inspection.
    captures, archive = {}, {}
    for label, model, fmt in MODELS:
        captures[label], archive[label] = {}, {}
        marks = []
        for key, col, fn, prompt, cases in INSTR:
            raw = gen_instruct(model, prompt)
            code = extract_code(raw)
            captures[label][key] = code
            p, note = run_cases(code, fn, cases)
            archive[label][key] = {"mode": "instruction", "col": col, "raw": raw,
                                   "out": code, "passed": p, "total": len(cases),
                                   "note": note}
            marks.append(f"{key[:4]}:{p}/{len(cases)}")
        for key, col, fn, prefix, suffix, cases in FIM:
            if fmt is None:
                captures[label][key] = None
                archive[label][key] = {"mode": "fim", "col": col, "raw": None,
                                       "out": None, "passed": None,
                                       "total": len(cases), "note": "no FIM"}
                marks.append(f"{key[:4]}:NA")
                continue
            raw = gen_fim(model, fmt, prefix, suffix)
            mid = clip_middle(raw, hole_indent(prefix))
            captures[label][key] = mid
            p, note = run_cases(prefix + mid + suffix, fn, cases)
            archive[label][key] = {"mode": "fim", "col": col, "raw": raw,
                                   "out": mid, "passed": p, "total": len(cases),
                                   "note": note}
            marks.append(f"{key[:4]}:{p}/{len(cases)}")
        print(f"{label:15} " + " ".join(marks), flush=True)

    json.dump(captures, open("chapters/genai/code_zoo.json", "w"),
              indent=1, ensure_ascii=False)
    json.dump(archive, open("chapters/genai/code_zoo_raw.json", "w"),
              indent=1, ensure_ascii=False)
    print("\nwrote chapters/genai/code_zoo.json + code_zoo_raw.json")
