"""Probe: find checkable problems where the three gemma4 tiers truly differ.

The routing demo needs a real difficulty gradient across the resolver tiers
  easy   -> gemma4:e2b   (tiny)
  medium -> gemma4:latest (mid)
  hard   -> gemma4:26b   (big)
so that routing DOWN buys time at no quality cost and routing UP buys quality.

This script runs a candidate pool of checkable problems (numeric, text, and code
graded by execution) on all three tiers at temperature 0 and reports, per
problem, which tiers solve it. The buckets fall out of the data:
  EASY   = all three solve            (routing down to e2b is free)
  MEDIUM = e2b fails, latest+26b solve (latest is the right tier)
  HARD   = only 26b solves            (routing up to 26b is the only way)

Run: uv run python scripts/_routing_tier_probe.py
"""
import re
import time

import ollama

client = ollama.Client(host="http://localhost:11434")

TIERS = [("easy", "gemma4:e2b"), ("medium", "gemma4:latest"), ("hard", "gemma4:26b")]

BARE = " Reply with ONLY the final answer, no explanation, no units."
CODE = " Reply with ONLY a Python code block, nothing else."


def ask(prompt: str, model: str, max_tokens: int = 256) -> tuple:
    """One deterministic forward pass; return (text, latency_s)."""
    t0 = time.perf_counter()
    resp = client.chat(model=model, think=False,
                       messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0, "num_predict": max_tokens})
    return (resp["message"]["content"] or "").strip(), time.perf_counter() - t0


# ── checkers ──────────────────────────────────────────────────────────────────
def _nums(text: str) -> list:
    return re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))


def check_num(reply: str, gold: str) -> bool:
    """Gold value appears among the numbers in the reply (last few)."""
    return gold in _nums(reply)[-4:]


def check_text(reply: str, gold) -> bool:
    golds = gold if isinstance(gold, list) else [gold]
    return any(g.lower() in reply.lower() for g in golds)


def check_code(reply: str, spec: dict) -> bool:
    """Extract a code block, exec it, run the named function against cases."""
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.S)
    code = m.group(1) if m else reply
    ns = {}
    try:
        exec(code, ns)                                   # noqa: S102 (probe sandbox)
        fn = ns[spec["fn"]]
        return all(fn(*args) == out for args, out in spec["cases"])
    except Exception:
        return False


CHECK = {"num": check_num, "text": check_text, "code": check_code}
SUFFIX = {"num": BARE, "text": BARE, "code": CODE}


# ── candidate pool (guess = my prior; the data decides the real bucket) ───────
CANDIDATES = [
    # --- EASY: lookups, formatting, one-step arithmetic (all three solve) ---
    dict(id="metres",   guess="easy", kind="num",  gold="3000",
         q="How many metres are in 3 kilometres?"),
    dict(id="gpu",      guess="easy", kind="text", gold="graphics processing unit",
         q="What does the abbreviation GPU stand for?"),
    dict(id="round",    guess="easy", kind="num",  gold="3.14",
         q="Round 3.14159 to two decimal places."),
    dict(id="percent",  guess="easy", kind="num",  gold="30",
         q="What is 15 percent of 200?"),
    dict(id="iseven",   guess="easy", kind="code",
         q="Write a Python function is_even(n) that returns True when n is even.",
         spec=dict(fn="is_even", cases=[((4,), True), ((7,), False), ((0,), True)])),
    dict(id="seq",      guess="easy", kind="num", gold="42",
         q="What is the next number in the sequence 2, 6, 12, 20, 30, ... ?"),
    dict(id="leap",     guess="easy", kind="code",
         q="Write a Python function is_leap(y) that returns True if y is a leap "
           "year in the Gregorian calendar.",
         spec=dict(fn="is_leap", cases=[((2000,), True), ((1900,), False), ((2024,), True),
                                        ((2023,), False), ((1600,), True), ((1700,), False)])),

    # --- MEDIUM: e2b fails, latest + 26b solve ---
    dict(id="bat",      guess="medium", kind="num", gold="0.05",
         q="A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 "
           "dollar more than the ball. How much does the ball cost in dollars?"),
    dict(id="coins",    guess="medium", kind="num", gold="13",
         q="How many distinct ways can you make 25 cents using any number of "
           "pennies, nickels, dimes, and quarters?"),
    dict(id="trailz",   guess="medium", kind="code",
         q="Write a Python function trailing_zeros(n) returning the number of "
           "trailing zeros in n factorial.",
         spec=dict(fn="trailing_zeros", cases=[((10,), 2), ((25,), 6), ((100,), 24), ((5,), 1)])),
    dict(id="clock",    guess="medium", kind="num", gold="7.5",
         q="What is the angle in degrees between the hour and minute hands of a "
           "clock at exactly 3:15?"),
    dict(id="modpow",   guess="medium", kind="num", gold="1",
         q="What is the remainder when 7 to the power 100 is divided by 5?"),
    dict(id="petlogic", guess="medium", kind="text", gold=["cat"],
         q="Alice, Bob, and Carol each own exactly one pet: a cat, a dog, or a "
           "fish. Alice owns neither the cat nor the dog. Bob owns the dog. Which "
           "pet does Carol own? Answer with just the pet."),

    # --- HARD: only 26b solves (multi-step that the 9B model fumbles snap) ---
    dict(id="well",     guess="hard", kind="num", gold="4",
         q="A snail is at the bottom of a 7-metre well. It climbs 4 metres each "
           "day and slides back 3 metres each night. How many days to reach the top?"),
    dict(id="sum3or5",  guess="hard", kind="num", gold="2418",
         q="What is the sum of all integers from 1 to 100 that are divisible by "
           "3 or by 5?"),
    dict(id="father",   guess="hard", kind="num", gold="36",
         q="A father is 3 times as old as his son. In 12 years the father will "
           "be twice as old as the son. How old is the father now?"),
    dict(id="mixture",  guess="hard", kind="num", gold="6",
         q="How many litres of pure water must be added to 10 litres of a 40 "
           "percent acid solution to dilute it to 25 percent acid?"),
    dict(id="roman",    guess="hard", kind="code",
         q="Write a Python function roman_to_int(s) converting a Roman numeral "
           "string to an integer, handling subtractive cases like IV and IX.",
         spec=dict(fn="roman_to_int", cases=[(("IV",), 4), (("IX",), 9),
                                             (("MCMXCIV",), 1994), (("LVIII",), 58)])),
    dict(id="calc",     guess="hard", kind="code",
         q="Write a Python function calc(s) that evaluates a string arithmetic "
           "expression containing +, -, *, / and parentheses, returning the number.",
         spec=dict(fn="calc", cases=[(("2+3*4",), 14), (("(2+3)*4",), 20), (("10/2-3",), 2)])),
]


def main():
    print(f"{'problem':9} {'guess':7} | {'e2b':^5} {'latest':^6} {'26b':^5} | "
          f"{'e2b_s':>6} {'lat_s':>6} {'26b_s':>6}  -> bucket")
    print("-" * 78)
    rows = []
    for c in CANDIDATES:
        marks, secs = {}, {}
        for tier, model in TIERS:
            reply, dt = ask(c["q"] + SUFFIX[c["kind"]], model)
            gold = c.get("gold") or c.get("spec")
            ok = CHECK[c["kind"]](reply, gold)
            marks[tier], secs[tier] = ok, dt
        passes = tuple(marks[t] for t, _ in TIERS)
        bucket = {(True, True, True): "EASY",
                  (False, True, True): "MEDIUM",
                  (False, False, True): "HARD"}.get(passes, "mixed:" + "".join(
                      "1" if p else "0" for p in passes))
        g = lambda b: "  ok " if b else "  .  "
        print(f"{c['id']:9} {c['guess']:7} | {g(marks['easy'])} {g(marks['medium']):^6} "
              f"{g(marks['hard'])} | {secs['easy']:6.1f} {secs['medium']:6.1f} "
              f"{secs['hard']:6.1f}  -> {bucket}")
        rows.append((c["id"], c["guess"], bucket, secs))
    print("-" * 78)
    for want in ("EASY", "MEDIUM", "HARD"):
        hits = [r[0] for r in rows if r[2] == want]
        print(f"{want:7}: {', '.join(hits) if hits else '(none)'}")
    avg = lambda t: sum(r[3][t] for r in rows) / len(rows)
    print(f"\nmean latency  e2b={avg('easy'):.1f}s  latest={avg('medium'):.1f}s  26b={avg('hard'):.1f}s")


if __name__ == "__main__":
    main()
