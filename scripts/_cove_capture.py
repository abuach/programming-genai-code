"""Capture the frozen CoVe demo outputs and render the chart, once.

Mirrors how the PAL / Self-Consistency demos were filled: run the real model and
re-roll until each transcript is representative, then save the stdout so the insert
script can embed it as a frozen cell. We want two contrasting real runs:

  - Portugal (the WIN): the draft slips in Morocco, the factored check rejects it and
    keeps Spain, so precision climbs 50% -> 100% with recall intact.
  - Brazil (the LIMIT): the draft is fully correct, but the factored check wrongly
    insists Brazil does not border Venezuela/Guyana/Suriname/Bolivia and drops them,
    so precision stays 100% while recall collapses -- the boundary needing retrieval.

Also renders plot_cove(COVE_STUDY) to images/augmentation/cove.png through nbclient
with the working directory set to chapters/, so the image lands beside the chapter's
other figures and the output dict matches the sibling charts.

Run once:  .venv/bin/python scripts/_cove_capture.py
"""
import io
import sys
import json
import contextlib
sys.path.insert(0, "chapters")
from genai import cove, show_cove, list_precision

DEMO_OUT = "scripts/_cove_demo_out.txt"     # Portugal win
LIMIT_OUT = "scripts/_cove_limit_out.txt"   # Brazil limit
PLOT_OUT = "scripts/_cove_plot_out.json"


def _render(result):
    """Render one precomputed CoVe run to text (no fresh model call)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        show_cove(result=result)
    return buf.getvalue()


def _dropped_correct(r):
    kept = set(r["revised"])
    return [it for it in r["draft"] if it in r["gold"] and it not in kept]


def _morocco_reason(r):
    return next((reason for it, ok, reason in r["checks"] if it == "morocco"), "")


def capture_win(tries=30):
    """Portugal roll where the draft over-reaches to Morocco, the check drops it, and
    the result is the clean 50% -> 100% precision win. The dialogue layout runs two
    lines per check, so a clean two-name draft lands around eight lines. Prefer a roll
    whose Morocco reply names the sea/strait between them (more vivid), but fall back to
    any clean win so we never loop forever."""
    fallback = None
    for i in range(tries):
        r = cove("Portugal")
        text = _render(r)
        n = len(text.rstrip("\n").split("\n"))
        win = ("morocco" in r["draft"] and "morocco" not in r["revised"]
               and "spain" in r["revised"] and list_precision(r["draft"], r["gold"]) < 1.0)
        if win and n <= 12:
            vivid = any(w in _morocco_reason(r).lower()
                        for w in ("separat", "strait", "sea", "gibraltar", "water"))
            print(f"[win] roll {i+1}: draft={r['draft']} -> {r['revised']}, "
                  f"{n} lines{'  vivid OK' if vivid else '  (plain)'}")
            if vivid:
                return text
            fallback = fallback or text
        else:
            print(f"[win] roll {i+1}: draft={r['draft']}  (reroll)")
    if fallback:
        return fallback
    raise SystemExit("no representative Portugal win; widen tries")


# The canonical collapse the walkthrough and the chart caption describe: a fully
# correct draft whose factored check denies the five northern/western neighbours and
# keeps only the two southern ones, so recall falls from seven of ten to two.
_CANON_DROP = {"colombia", "venezuela", "guyana", "suriname", "bolivia"}
_CANON_KEEP = {"paraguay", "argentina"}


def capture_limit(tries=60):
    """Brazil roll where the draft is fully correct but the factored check wrongly
    denies five real neighbours and keeps only Paraguay and Argentina, the recall
    collapse the walkthrough names country by country. The dialogue layout runs two
    lines per check, so a seven-name draft lands around twenty lines (the calling cell
    is tagged no-trim to keep them). Rank every qualifying roll and return the best: the
    canonical five-drop (Colombia, Venezuela, Guyana, Suriname, Bolivia gone, the two
    southern neighbours kept) on an all-sovereign draft, falling back through clean
    three-plus-drops to any qualifying roll so the capture never loops forever."""
    best = None                 # (rank, text); a smaller rank tuple is more representative
    for i in range(tries):
        r = cove("Brazil")
        text = _render(r)
        n = len(text.rstrip("\n").split("\n"))
        bad = _dropped_correct(r)
        if list_precision(r["draft"], r["gold"]) != 1.0 or len(bad) < 3 or n > 24:
            print(f"[limit] roll {i+1}: dropped_correct={bad} {n}ln  (reroll)")
            continue
        clean = "french guiana" not in r["draft"]
        canon = set(r["revised"]) == _CANON_KEEP and _CANON_DROP <= set(bad)
        rank = (0 if canon else 1, 0 if clean else 1, -len(bad))
        print(f"[limit] roll {i+1}: dropped correct {bad}, {n} lines"
              f"{'  CANON' if canon else ''}{'  clean' if clean else '  (territory)'}")
        if best is None or rank < best[0]:
            best = (rank, text)
        if canon and clean:
            return text
    if best:
        return best[1]
    raise SystemExit("no representative Brazil limit; widen tries")


def render_plot():
    """Execute the chart cell through nbclient with the working directory set to
    chapters/, so plot_cove saves under chapters/images/augmentation/ and the output
    dict (image/png + text/plain, empty metadata) matches the chapter's other charts."""
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("from genai import plot_cove, COVE_STUDY"),
        nbformat.v4.new_code_cell("plot_cove(COVE_STUDY)"),
    ]
    NotebookClient(nb, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": "chapters"}}).execute()
    return nb.cells[1].outputs


if __name__ == "__main__":
    win = capture_win()
    open(DEMO_OUT, "w").write(win)
    print("\n----- PORTUGAL WIN -----\n" + win)

    limit = capture_limit()
    open(LIMIT_OUT, "w").write(limit)
    print("\n----- BRAZIL LIMIT -----\n" + limit)

    outputs = render_plot()   # also writes chapters/images/augmentation/cove.png
    json.dump(outputs, open(PLOT_OUT, "w"))
    print(f"\nchart outputs -> {PLOT_OUT}")
