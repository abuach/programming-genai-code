"""Capture the frozen PAL demo outputs and render the chart, once.

Renders the two frozen PAL panels for the Compound problem from committed caches
(scripts/_pal_cot_capture.json, _pal_demo_capture.json): the prose panel where the
model sets the calculation up correctly then commits a confidently wrong number,
and the program panel where the same model's code runs to the right answer. The
clean wrong commit is rare for a capable coding model (it usually nails the
arithmetic or sprawls past its token budget without landing), so it is cached
rather than re-rolled; delete a cache to draw and re-cache a fresh roll. Also
renders plot_pal(PAL_STUDY) to images/prompting/pal.png for the chart cell.

Run once:  .venv/bin/python scripts/_pal_capture.py
"""
import io
import re
import sys
import json
import contextlib
from pathlib import Path
sys.path.insert(0, "chapters")
from genai.pal import (cot_answer, pal_solve, _show_prose_turn, _show_program_turn,
                       _compact, _problem, _close, _fmt)

LABEL = "Compound"
COT_OUT = "scripts/_pal_cot_out.txt"
DEMO_OUT = "scripts/_pal_demo_out.txt"
PLOT_OUT = "scripts/_pal_plot_out.json"
COT_CACHE = Path("scripts/_pal_cot_capture.json")
DEMO_CACHE = Path("scripts/_pal_demo_capture.json")
_ANS = re.compile(r"answer\s*[:=]\s*\$?-?[\d,]+", re.I)


def _render_prose(text, val):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _show_prose_turn(LABEL, "qwen2.5-coder", text, val)
    return buf.getvalue()


def capture_prose(tries=40):
    """Render the prose panel: the model sets the calculation up correctly and then
    commits a WRONG number. That clean wrong commit is rare, so the first good roll
    is cached and replayed; delete COT_CACHE to draw a fresh one."""
    if COT_CACHE.exists():
        d = json.loads(COT_CACHE.read_text())
        print(f"[prose] cached: lands on {_fmt(d['val'])}")
        return _render_prose(d["message"]["content"], d["val"])
    _, q, value, tol, _ = _problem(LABEL)
    for i in range(tries):
        text, val = cot_answer(q)
        last = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
        clean_wrong = (val is not None and not _close(val, value, tol)
                       and bool(_ANS.search(last)) and text.lstrip()[:1].isalpha())
        if clean_wrong:
            json.dump({"label": LABEL, "message": {"thinking": "", "content": text}, "val": val},
                      open(COT_CACHE, "w"), ensure_ascii=False, indent=1)
            print(f"[prose] roll {i+1}: clean wrong commit -> {_fmt(val)}  ✓")
            return _render_prose(text, val)
        print(f"[prose] roll {i+1}: reroll")
    raise SystemExit("no clean wrong-commit prose roll; widen tries or pick another problem")


def _render_program(code, val):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _show_program_turn(LABEL, code, val)
    return buf.getvalue()


def capture_program(tries=24):
    """Render the program panel: the SAME model writes a short program the
    interpreter runs to the RIGHT answer, narrow enough not to wrap the PDF box
    (~70 monospace chars). Cached for determinism; the cell is tagged no-trim, so a
    line or two past ten is fine."""
    if DEMO_CACHE.exists():
        d = json.loads(DEMO_CACHE.read_text())
        print(f"[program] cached: interpreter -> {_fmt(d['val'])}")
        return _render_program(d["code"], d["val"])
    _, q, value, tol, _ = _problem(LABEL)
    for i in range(tries):
        code, val = pal_solve(q)
        lines = _compact(code)
        wide = max((len(l) for l in lines), default=99)
        if _close(val, value, tol) and len(lines) <= 10 and wide <= 70:
            json.dump({"label": LABEL, "code": code, "val": val},
                      open(DEMO_CACHE, "w"), ensure_ascii=False, indent=1)
            print(f"[program] roll {i+1}: correct, {len(lines)} lines, widest {wide}  ✓")
            return _render_program(code, val)
        print(f"[program] roll {i+1}: reroll")
    raise SystemExit("no representative program roll; widen tries or pick another problem")


def render_plot():
    """Execute the chart cell through nbclient with retina inline rendering, so its
    output dict matches the sibling charts exactly (text/plain + image/png +
    width/height metadata). Returns the outputs list for the cell."""
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "%config InlineBackend.figure_format = 'retina'\n"
            "import sys; sys.path.insert(0, 'chapters')\n"
            "from genai import plot_pal, PAL_STUDY"),
        nbformat.v4.new_code_cell("plot_pal(PAL_STUDY)"),
    ]
    NotebookClient(nb, timeout=120).execute()
    return nb.cells[1].outputs


if __name__ == "__main__":
    cot = capture_prose()
    open(COT_OUT, "w").write(cot)
    print("\n----- PROSE PANEL -----\n" + cot)

    demo = capture_program()
    open(DEMO_OUT, "w").write(demo)
    print("\n----- PROGRAM PANEL -----\n" + demo)

    outputs = render_plot()   # also writes images/prompting/pal.png via _save
    json.dump(outputs, open(PLOT_OUT, "w"))
    print(f"\nchart outputs -> {PLOT_OUT}  (image/png + retina metadata)")
