"""Capture every cast model's tier picks (temperature 0) for the dispatch contest.
Emits ROUTER_PICKS {model: {task_id: tier}} and an updated ROUTER_STUDY with
correct/under/over (vs the gold tier) merged onto the existing solve_pct/time_s.
Reuses route_tier only — no resolve calls, so the baked latency table is untouched.

Run: uv run python scripts/_router_picks_capture.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))

from genai.agent import route_tier, ROUTING_TASKS, AGENT_BENCH_MODELS, ROUTER_STUDY

ORDER = {"easy": 0, "medium": 1, "hard": 2}
GOLD = {t["id"]: ORDER[t["tier"]] for t in ROUTING_TASKS}


def split(picks):
    u = c = o = 0
    for tid, tier in picks.items():
        d = ORDER[tier] - GOLD[tid]
        u, c, o = (u + 1, c, o) if d < 0 else (u, c + 1, o) if d == 0 else (u, c, o + 1)
    return u, c, o


def main():
    picks_by_model, stats = {}, {}
    for model in AGENT_BENCH_MODELS:
        picks = {t["id"]: route_tier(t["q"], classifier=model) for t in ROUTING_TASKS}
        u, c, o = split(picks)
        picks_by_model[model] = picks
        stats[model] = (u, c, o)
        print(f"#   {model:22} under={u} correct={c} over={o}", file=sys.stderr)

    n = len(ROUTING_TASKS)
    print("\n# ===== paste ROUTER_PICKS into genai/agent.py =====\n")
    print("ROUTER_PICKS = {")
    for m, picks in picks_by_model.items():
        cells = ", ".join(f'"{tid}": "{tier}"' for tid, tier in picks.items())
        print(f'    "{m}": {{{cells}}},')
    print("}\n")

    print("# ===== paste ROUTER_STUDY (correct/under/over merged) =====\n")
    print("ROUTER_STUDY = {")
    for m in picks_by_model:
        u, c, o = stats[m]
        old = ROUTER_STUDY.get(m, {})
        print(f'    "{m}": {{"correct": {c}, "under": {u}, "over": {o}, '
              f'"correct_pct": {round(100 * c / n, 1)}, '
              f'"solve_pct": {old.get("solve_pct", round(100 * (c + o) / n, 1))}, '
              f'"time_s": {old.get("time_s", 0.0)}}},')
    print("}")


if __name__ == "__main__":
    main()
