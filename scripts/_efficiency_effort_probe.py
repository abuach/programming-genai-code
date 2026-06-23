"""Probe: gpt-oss:20b's reasoning-effort dial (low/medium/high) as an EFFICIENCY
cost knob. For each effort level and task, measure the token bill (eval_count,
reasoning + answer), the load-free generation time (eval_duration), the split
between hidden reasoning and visible answer, and whether the answer is correct.

The question this answers before any notebook build: on tasks gpt-oss already
solves, does higher effort buy anything but a bigger bill? And is there any task
where higher effort actually flips a wrong answer to a right one?

Run on an idle GPU (gpt-oss is the model that contends; see efficiency_sparsity
demo's contention note). Token counts are load-independent, so they're the
honest cost metric; wall-clock is reported too but only the warm runs matter.

    uv run python scripts/_efficiency_effort_probe.py
"""
import subprocess
import sys
import time
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "chapters"))
from genai.thinking import NOVEL_PROBLEMS, _grade  # noqa: E402

client = ollama.Client(host="http://localhost:11434")
MODEL = "gpt-oss:20b"
EFFORTS = ["low", "medium", "high"]
SNAP = " Reply with only the final answer."

# Easy/known-answer tasks (expect solved at every effort -> pure cost), then the
# book's calibrated NOVEL_PROBLEMS (harder -> might separate by effort).
EASY = [
    ("Capital", "What is the capital of Australia?", ["canberra"]),
    ("Bakery", "A bakery sells 3 dozen muffins in the morning and 18 more in the "
               "afternoon. How many muffins total each day?", ["54"]),
]
TASKS = [(n, q, g, "easy") for n, q, g in EASY] + \
        [(n, q, g, "hard") for n, q, g in NOVEL_PROBLEMS]

# ── Frontier battery: known reasoning traps with a strong wrong intuition, to
# hunt for a task where low effort fails but high effort succeeds (effort as a
# real lever). Computational golds are recomputed here so they're verifiable.
_digit1 = sum(str(n).count("1") for n in range(1, 101))            # 21
_neither = sum(1 for n in range(1, 101) if n % 2 and n % 3)        # 33
_coins50 = sum(1 for q in range(3) for d in range(6) for n in range(11)
               if 25 * q + 10 * d + 5 * n <= 50)                   # 49
_units7 = pow(7, 2024, 10)                                         # 1
FRONTIER = [
    ("Dice6", "Two fair six-sided dice are rolled. Given that at least one shows "
              "a 6, what is the probability that both show a 6? Give a fraction.",
     ["1/11"]),
    ("BoyTuesday", "A family has two children. At least one is a boy born on a "
                   "Tuesday. What is the probability both children are boys? "
                   "Give a fraction.", ["13/27"]),
    ("Horse", "A man buys a horse for $60, sells it for $70, buys it back for $80, "
              "and sells it again for $90. What is his total profit in dollars?",
     ["$20", "20 dollar", "= 20", "is 20"]),
    ("Clock", "What is the smaller angle in degrees between the hour and minute "
              "hands of a clock at exactly 3:15?", ["7.5"]),
    ("Widgets", "If 5 machines take 5 minutes to make 5 widgets, how many minutes "
                "do 100 machines take to make 100 widgets?", ["5 min", "is 5", "= 5"]),
    ("Digit1", "How many times does the digit 1 appear when you write out every "
               "integer from 1 to 100 inclusive?", [str(_digit1)]),
    ("Neither", "How many integers from 1 to 100 inclusive are divisible by "
                "neither 2 nor 3?", [str(_neither)]),
    ("Coins50", "How many ways can you make exactly 50 cents using any number of "
                "pennies, nickels, dimes, and quarters?", [str(_coins50)]),
    ("Units7", "What is the units digit of 7 raised to the power 2024?", [str(_units7)]),
    ("Monty", "Monty Hall, 3 doors: you pick door 1, the host opens door 3 to "
              "show a goat. If you switch to door 2, what is your probability of "
              "winning the car? Give a fraction.", ["2/3"]),
]


def grade(name, tier, answer, gold):
    if tier == "easy":
        return any(g.lower() in answer.lower() for g in gold)
    return _grade(answer, gold)


def run(effort, question):
    t0 = time.perf_counter()
    r = client.chat(model=MODEL, messages=[{"role": "user", "content": question}],
                    think=effort, options={"num_predict": 8000, "temperature": 0})
    wall = time.perf_counter() - t0
    m = r["message"]
    return {
        "answer": (m.get("content") or "").strip(),
        "think": (m.get("thinking") or "").strip(),
        "eval_count": r.get("eval_count", 0),
        "gen_s": r.get("eval_duration", 0) / 1e9,
        "wall_s": wall,
    }


def main():
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    foreign = [ln.split()[0] for ln in ps.splitlines()[1:]
               if ln.strip() and not ln.startswith(MODEL)]
    if foreign:
        print(f"WARNING: other models resident ({', '.join(foreign)}); GPU shared.\n")

    run("low", "Hello")  # warm-up so the first real run isn't a cold load
    rows = []
    for effort in EFFORTS:
        print(f"\n=== effort = {effort} ===", flush=True)
        for name, q, gold, tier in TASKS:
            m = run(effort, q + (SNAP if tier == "hard" else ""))
            ok = grade(name, tier, m["answer"], gold)
            rows.append({"effort": effort, "task": name, "tier": tier, "ok": ok, **m})
            print(f"  {name:11}({tier}) {'OK ' if ok else 'XX '} "
                  f"bill={m['eval_count']:5} tok  reason~{len(m['think']):5}c "
                  f"ans~{len(m['answer']):4}c  gen={m['gen_s']:5.1f}s  "
                  f"-> {m['answer'][:42]!r}", flush=True)
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)

    print("\n" + "=" * 64 + "\nSUMMARY per effort (mean token bill, mean gen time, accuracy)")
    for effort in EFFORTS:
        e = [r for r in rows if r["effort"] == effort]
        acc = sum(r["ok"] for r in e)
        tok = sum(r["eval_count"] for r in e) / len(e)
        gen = sum(r["gen_s"] for r in e) / len(e)
        print(f"  {effort:6} acc {acc}/{len(e)}  mean bill {tok:6.0f} tok  "
              f"mean gen {gen:5.1f}s", flush=True)

    print("\nPer-task token bill low -> medium -> high:")
    for name, q, gold, tier in TASKS:
        bills = [next(r["eval_count"] for r in rows
                      if r["effort"] == ef and r["task"] == name) for ef in EFFORTS]
        oks = [next(r["ok"] for r in rows if r["effort"] == ef and r["task"] == name)
               for ef in EFFORTS]
        flips = "" if len(set(oks)) == 1 else "  <-- EFFORT CHANGES CORRECTNESS"
        mult = bills[2] / bills[0] if bills[0] else 0
        print(f"  {name:11}({tier}) {bills[0]:5} -> {bills[1]:5} -> {bills[2]:5} tok "
              f"({mult:.1f}x)  correct {oks}{flips}", flush=True)

    # Bakeable constant for the notebook: per-effort means + one example task.
    import json
    n = len(TASKS)

    def per_effort(field, fn):
        return [fn([r[field] for r in rows if r["effort"] == ef]) for ef in EFFORTS]

    ex_q = next(q for nm, q, g in NOVEL_PROBLEMS if nm == "Well")
    ex = {ef: next(r for r in rows if r["effort"] == ef and r["task"] == "Well")
          for ef in EFFORTS}
    study = {
        "efforts": EFFORTS,
        "bill": [round(v) for v in per_effort("eval_count", lambda xs: sum(xs) / len(xs))],
        "gen_s": [round(v, 1) for v in per_effort("gen_s", lambda xs: sum(xs) / len(xs))],
        "correct": per_effort("ok", sum),
        "n_tasks": n,
        "example": {
            "task": "snail-in-the-well", "question": ex_q,
            "answer": ex["low"]["answer"],
            "bill": [ex[ef]["eval_count"] for ef in EFFORTS],
            "gen_s": [round(ex[ef]["gen_s"], 1) for ef in EFFORTS],
        },
    }
    print("\nEFFORT_STUDY = " + json.dumps(study, indent=4))


def hunt_frontier():
    """Run the frontier battery at low vs high effort; flag any task where low
    fails but high succeeds (effort earning its cost). Prints full answers so
    each verdict can be eyeballed, since these answers are fractions and traps."""
    run("low", "Hello")  # warm-up
    flips = []
    for name, q, gold in FRONTIER:
        res = {}
        for effort in ("low", "high"):
            m = run(effort, q + " Reply with only the final answer.")
            ans = m["answer"]
            norm = ans.lower().replace(" ", "").replace("\\", "").replace("frac{", "")\
                      .replace("}{", "/").replace("}", "").replace("$", "")
            ok = any(g.lower().replace(" ", "").replace("$", "") in norm for g in gold)
            res[effort] = (ok, m["eval_count"], ans)
            print(f"  {name:11} {effort:4} {'OK ' if ok else 'XX '} "
                  f"bill={m['eval_count']:5}  gold={gold[0]:7}  -> {ans[:60]!r}",
                  flush=True)
        if not res["low"][0] and res["high"][0]:
            flips.append(name)
        print(flush=True)
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)
    print("=" * 64)
    print(f"FRONTIER FLIPS (low fails, high succeeds): {flips or 'NONE'}")


if __name__ == "__main__":
    if "frontier" in sys.argv:
        hunt_frontier()
    else:
        main()
