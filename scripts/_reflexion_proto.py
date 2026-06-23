"""THROWAWAY prototype: does Reflexion (verbal lessons in memory) actually beat
blind retry on local models, with honestly-computed numbers? Not wired into the
book. Run: .venv/bin/python scripts/_reflexion_proto.py [model]"""
import re, sys, signal, statistics, io, contextlib
sys.path.insert(0, "chapters")
from genai.llm import ask

# ── A suite of tiny tasks, each with a classic edge-case trap and a real check ──
# Each task: (fn_name, spec, cases) where cases = list of (args_tuple, expected).
SUITE = [
    ("spreadsheet_col",
     "Write spreadsheet_col(n): convert a 1-based column number to its spreadsheet "
     "letters (bijective base-26), so 1->'A', 26->'Z', 27->'AA', 702->'ZZ'.",
     [((1,), "A"), ((26,), "Z"), ((27,), "AA"), ((53,), "BA"),
      ((702,), "ZZ"), ((703,), "AAA")]),
    ("col_to_num",
     "Write col_to_num(s): convert spreadsheet column letters to a 1-based number "
     "(inverse of bijective base-26), so 'A'->1, 'Z'->26, 'AA'->27, 'ZZ'->702.",
     [(("A",), 1), (("Z",), 26), (("AA",), 27), (("BA",), 53), (("ZZ",), 702)]),
    ("to_roman",
     "Write to_roman(n) for 1..3999: return the Roman numeral in subtractive form "
     "(4->'IV', 9->'IX', 40->'XL', 90->'XC', 400->'CD', 900->'CM').",
     [((4,), "IV"), ((9,), "IX"), ((40,), "XL"), ((90,), "XC"), ((400,), "CD"),
      ((900,), "CM"), ((1994,), "MCMXCIV"), ((3888,), "MMMDCCCLXXXVIII")]),
    ("is_valid_ipv4",
     "Write is_valid_ipv4(s): True only if s is four dot-separated decimal octets, "
     "each 0-255 with NO leading zeros (so '0' is ok but '01' is not).",
     [(("192.168.1.1",), True), (("255.255.255.255",), True), (("0.0.0.0",), True),
      (("256.1.1.1",), False), (("01.2.3.4",), False), (("1.2.3",), False),
      (("1.2.3.04",), False)]),
    ("rle_decode",
     "Write rle_decode(s): decode run-length text where each run is one non-digit "
     "char followed by its count, and counts can be multi-digit, so 'a3b12'->'aaa'"
     "+'b'*12.",
     [(("a3",), "aaa"), (("a3b2",), "aaabb"), (("a12",), "a"*12),
      (("a3b12c1",), "aaa"+"b"*12+"c")]),
    ("caesar",
     "Write caesar(s, k): Caesar-shift letters by k preserving case and wrapping "
     "within the alphabet; leave every non-letter unchanged.",
     [(("abc", 1), "bcd"), (("xyz", 3), "abc"), (("Zebra", 1), "Afcsb"),
      (("Hello, World!", 5), "Mjqqt, Btwqi!")]),
    ("parse_duration",
     "Write parse_duration(s) -> total minutes: support '1h30m'->90, '2h'->120, "
     "'45m'->45, a bare number like '90'->90 (minutes), and ''->0.",
     [(("1h30m",), 90), (("2h",), 120), (("45m",), 45), (("90",), 90), (("",), 0)]),
    ("expand_ranges",
     "Write expand_ranges(s): expand a comma list of numbers and a-b ranges into a "
     "sorted int list, so '1-3,5,7-8' -> [1,2,3,5,7,8].",
     [(("1-3,5,7-8",), [1,2,3,5,7,8]), (("1",), [1]), (("1-1",), [1]),
      (("10-12,1",), [1,10,11,12])]),
]


class _TO(Exception):
    pass


def _alarm(sig, frame):
    raise _TO()


def extract_fn(text, name):
    """Pull the function source out of a model reply (fenced block if present)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    # keep from the first 'def name' or 'import'/'def' onward
    i = body.find("def " + name)
    if i == -1:
        i = body.find("import ")
    return body[i:] if i != -1 else body


def check(code, name, cases):
    """Exec candidate, run every case. Return (ok, detail_of_first_failure)."""
    ns = {}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(3)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec("import math, re\n" + code, ns)
            fn = ns.get(name)
            if not callable(fn):
                return False, f"{name} was not defined"
            for args, exp in cases:
                got = fn(*args)
                if got != exp:
                    a = ", ".join(map(repr, args))
                    return False, f"{name}({a}) returned {got!r}, expected {exp!r}"
        return True, "all cases passed"
    except _TO:
        return False, "timed out (likely infinite loop)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)


def attempt(spec, name, lessons, model, temp):
    hint = ""
    if lessons:
        hint = "\nLessons from your past failed attempts:\n" + \
               "\n".join(f"- {l}" for l in lessons)
    # reasoning models (gpt-oss) think before they write, so give them room
    budget = 512 if "gpt-oss" in model else 0
    code = ask(f"{spec}\nReturn ONLY the function in a python code block.{hint}",
               model=model, system="You are a careful Python programmer.",
               max_tokens=220, thinking_budget=budget, options={"temperature": temp})
    return extract_fn(code, name)


def reflect(spec, detail, model):
    budget = 512 if "gpt-oss" in model else 0
    return ask(f"Your code for this task failed.\nTask: {spec}\nFailure: {detail}\n"
               "In ONE short sentence, state the mistake and what to do differently.",
               model=model, system="You are a careful Python programmer.",
               max_tokens=60, thinking_budget=budget,
               options={"temperature": temp}).strip().replace("\n", " ")


def run_condition(use_memory, model, K, temp, repeats, log=False):
    """Return cumulative fraction of tasks solved by trial k, k=1..K (averaged)."""
    tag = "REFLEX" if use_memory else "BLIND "
    per_repeat = []
    for r in range(repeats):
        solved_at = []
        for name, spec, cases in SUITE:
            lessons, first = [], None
            for t in range(1, K + 1):
                code = attempt(spec, name, lessons if use_memory else [], model, temp)
                ok, detail = check(code, name, cases)
                if ok:
                    first = t
                    break
                if use_memory:
                    lessons.append(reflect(spec, detail, model))
            solved_at.append(first)
            if log:
                print(f"  [{tag} r{r}] {name:14} solved@{first}", flush=True)
        cum = [sum(1 for s in solved_at if s is not None and s <= k) / len(SUITE)
               for k in range(1, K + 1)]
        per_repeat.append(cum)
    return [statistics.mean(col) for col in zip(*per_repeat)]


if __name__ == "__main__":
    global temp
    model   = sys.argv[1] if len(sys.argv) > 1 else "deepseek-coder:latest"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    K, temp = 4, 0.7
    print(f"model={model}  K={K}  temp={temp}  repeats={repeats}  tasks={len(SUITE)}",
          flush=True)
    refl  = run_condition(True,  model, K, temp, repeats, log=True)
    print("reflexion   by trial:", [f"{x:.0%}" for x in refl], flush=True)
    blind = run_condition(False, model, K, temp, repeats, log=True)
    print("blind retry by trial:", [f"{x:.0%}" for x in blind], flush=True)
