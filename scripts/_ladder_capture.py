"""Reproducer + capture for the "Matching Model to Task" ladder demo
(efficiency.ipynb, cell 81d2ca51).

Runs the task ladder for real (1B vs 3B llama3.2 on three GPU-themed
questions of rising difficulty), captures the printed speeds and the
plot_task_ladder figure, and embeds both back into the frozen notebook
cell. Timing drifts run to run; the embedded output is one honest capture.
The stable shape the prose leans on: the 1B model is the faster tier at
every rung, and neither model's per-token speed moves much as the task
gets harder, because speed is a property of the model, not the question.

Usage: .venv/bin/python scripts/_ladder_capture.py
"""
import base64
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"
sys.path.insert(0, str(CHAPTERS))

NB_PATH = CHAPTERS / "efficiency.ipynb"
CELL_ID = "81d2ca51"
RENDER_DPI = 130

# Exact source of the notebook cell; the capture executes this verbatim.
CELL_SRC = '''\
##system
# One task ladder, two model sizes: speed belongs to the model.
ladder = [("simple",   "What does GPU stand for?"),
          ("moderate", "Explain in one sentence why AI runs on GPUs."),
          ("complex",  "Compare running AI locally versus in the cloud.")]
rows = []
for task, prompt in ladder:
    small = time_call(prompt, model="llama3.2:1b")
    large = time_call(prompt, model="llama3.2:latest")
    rows.append({"task": task, "small": small["tokens_per_sec"],
                 "large": large["tokens_per_sec"]})
    print(f"{task:9s} 1B {small['tokens_per_sec']:3.0f} t/s   "
          f"3B {large['tokens_per_sec']:3.0f} t/s")
plot_task_ladder(rows)'''


def main():
    import os
    os.chdir(CHAPTERS)  # so the figure lands in chapters/images/efficiency/
    from genai import time_call, plot_task_ladder

    # Warm both models so the first timed call isn't paying a load cost.
    for m in ("llama3.2:1b", "llama3.2:latest"):
        time_call("hi", model=m)

    plt.close("all")
    ns = {"time_call": time_call, "plot_task_ladder": plot_task_ladder}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(CELL_SRC, "<ladder cell>", "exec"), ns)
    text = stdout.getvalue()
    print(text, end="")

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI,
                bbox_inches="tight", facecolor="white")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    size = f"{int(fig.get_figwidth() * 100)}x{int(fig.get_figheight() * 100)}"
    plt.close(fig)

    raw = NB_PATH.read_text(encoding="utf-8")
    nb = json.loads(raw)
    cell = next(c for c in nb["cells"] if c.get("id") == CELL_ID)
    cell["source"] = CELL_SRC.splitlines(keepends=True)
    cell["outputs"] = [
        {"name": "stdout", "output_type": "stream",
         "text": text.splitlines(keepends=True)},
        {"data": {"image/png": b64,
                  "text/plain": [f"<Figure size {size} with 1 Axes>"]},
         "metadata": {}, "output_type": "display_data"},
    ]

    # Byte-stable write: preserve the file's unicode escaping and trailing
    # newline so untouched cells stay byte-for-byte identical.
    new = json.dumps(nb, indent=1, ensure_ascii=raw.isascii())
    if raw.endswith("\n"):
        new += "\n"
    if new != raw:
        NB_PATH.write_text(new, encoding="utf-8")
        print(f"embedded capture into {NB_PATH.name} cell {CELL_ID}")
    else:
        print("notebook unchanged")


if __name__ == "__main__":
    main()
