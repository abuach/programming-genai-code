"""Capture the baked routing tables for genai/agent.py (temperature 0).

Runs every live model call ONCE and prints Python literals to paste over the
placeholders in agent.py:
  ROUTING_SOLVE   - per task, each tier's (correct, latency, real answer)
  ROUTING_PICKS   - gemma4 as the router: the tier it labels each task
  ROUTER_STUDY    - every cast model scored as a router by solves + time
  ROUTING_SAVINGS - the 12-task batch answered three ways

Everything downstream (figures, show_routing, show_savings, the season card)
reads these, so the notebook never calls a model live.

Run: uv run python scripts/_routing_capture.py   (~15 min; 26b is slow)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))

from genai.agent import (ROUTING_TASKS, TIER_MODEL, TIER_ORDER, AGENT_BENCH_MODELS,
                         resolve, route_tier)
from genai.llm import ask as _ask

ROUTER_FIG = ["gemma4:latest", "qwen3:latest", "qwen3.5:latest",
              "mistral:latest", "ministral-3:latest", "llama3.1:latest"]


def warm(model):
    _ask("hi", model=model, max_tokens=2, system="")


def clip(text, n=80):
    return " ".join(text.split())[:n]


def capture_solve():
    """resolve every task at every tier -> {id: {tier: {ok, s, a}}}."""
    print("# resolving 12 tasks x 3 tiers ...", file=sys.stderr)
    for m in TIER_MODEL.values():
        warm(m)
    solve = {}
    for task in ROUTING_TASKS:
        solve[task["id"]] = {}
        for tier in TIER_ORDER:
            ok, dt, ans = resolve(task, tier)
            solve[task["id"]][tier] = {"ok": ok, "s": dt, "a": clip(ans)}
            print(f"#   {task['id']:9} {tier:6} {'ok' if ok else '..'} {dt:5.1f}s",
                  file=sys.stderr)
    return solve


def score_router(model, solve):
    """Classify every task with `model`, resolve via the table; return picks + score."""
    warm(model)
    picks, n_ok, total_s = {}, 0, 0.0
    for task in ROUTING_TASKS:
        t0 = time.perf_counter()
        tier = route_tier(task["q"], classifier=model)
        entry = solve[task["id"]][tier]
        picks[task["id"]] = tier
        n_ok += int(entry["ok"])
        total_s += (time.perf_counter() - t0) + entry["s"]
    return picks, {"solve_pct": round(100 * n_ok / len(ROUTING_TASKS), 1),
                   "time_s": round(total_s, 1), "n_correct": n_ok}


def agg(solve, tier):
    """All tasks resolved at one fixed tier: (solved, total_time)."""
    solved = sum(solve[t["id"]][tier]["ok"] for t in ROUTING_TASKS)
    secs = round(sum(solve[t["id"]][tier]["s"] for t in ROUTING_TASKS), 1)
    return solved, secs


def emit(solve, picks, study, savings):
    n = len(ROUTING_TASKS)
    print("\n\n# ===== paste the four literals below into genai/agent.py =====\n")

    print("ROUTING_SOLVE = {")
    for t in ROUTING_TASKS:
        cells = ", ".join(
            f'"{tier}": {{"ok": {solve[t["id"]][tier]["ok"]}, '
            f'"s": {solve[t["id"]][tier]["s"]}, "a": {solve[t["id"]][tier]["a"]!r}}}'
            for tier in TIER_ORDER)
        print(f'    "{t["id"]}": {{{cells}}},')
    print("}\n")

    print("ROUTING_PICKS = {")
    print("    " + ", ".join(f'"{tid}": "{tier}"' for tid, tier in picks.items()))
    print("}\n")

    print("ROUTER_STUDY = {")
    for m, v in study.items():
        print(f'    "{m}": {{"solve_pct": {v["solve_pct"]}, '
              f'"time_s": {v["time_s"]}, "n_correct": {v["n_correct"]}}},')
    print("}\n")

    print("ROUTING_SAVINGS = {")
    for key in ("all_cheap", "all_strong", "routed"):
        s = savings[key]
        print(f'    "{key}": {{"solved": {s["solved"]}, "total": {n}, '
              f'"time_s": {s["time_s"]}}},')
    print("}")


def main():
    solve = capture_solve()

    print("# scoring routers ...", file=sys.stderr)
    study, picks_by_model = {}, {}
    for model in AGENT_BENCH_MODELS:
        picks, score = score_router(model, solve)
        study[model] = score
        picks_by_model[model] = picks
        print(f"#   {model:22} solved {score['n_correct']}/{len(ROUTING_TASKS)} "
              f"in {score['time_s']}s", file=sys.stderr)

    gemma_picks = picks_by_model["gemma4:latest"]
    cheap_solved, cheap_s = agg(solve, "easy")
    strong_solved, strong_s = agg(solve, "hard")
    savings = {
        "all_cheap":  {"solved": cheap_solved,  "time_s": cheap_s},
        "all_strong": {"solved": strong_solved, "time_s": strong_s},
        "routed":     {"solved": study["gemma4:latest"]["n_correct"],
                       "time_s": study["gemma4:latest"]["time_s"]},
    }
    # keep the figure subset ordered the way the chart wants it
    ordered = {m: study[m] for m in ROUTER_FIG if m in study}
    ordered.update({m: study[m] for m in study if m not in ordered})
    emit(solve, gemma_picks, ordered, savings)


if __name__ == "__main__":
    main()
