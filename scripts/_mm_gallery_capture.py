"""Render and embed the mm.ipynb 'what diffusion does well' gallery montage.

The mm-gallery-code cell is frozen (execute_notebook.py skips it), so this
regenerates its embedded figure in place. It executes the cell's own source
(the three verbatim FLUX prompts + seeds live there) with plt.show() patched to
capture the montage, then writes the PNG back byte-stably. Seed-pinned, so the
montage is reproducible. Idempotent: locates the cell by id and re-renders.

Each FLUX image is ~2 min on MPS after the pipeline loads, so budget ~10 min.

Run:  uv run python scripts/_mm_gallery_capture.py
"""
import base64
import io
import json
import sys

sys.path.insert(0, "chapters")  # so `from genai import gallery` resolves

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

NB = "chapters/mm.ipynb"
CELL_ID = "mm-gallery-code"


def render(src):
    """Exec a cell's source, capturing the figure it hands to plt.show()."""
    cap = {}

    def grab(*_a, **_k):
        fig = plt.gcf()
        cap["nominal"] = tuple(int(round(d * fig.dpi)) for d in fig.get_size_inches())
        cap["naxes"] = len(fig.axes)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight")
        cap["png"] = buf.getvalue()
        plt.close(fig)

    orig, plt.show = plt.show, grab
    try:
        exec(compile(src, "<cell>", "exec"), {"__name__": "__cell__"})
    finally:
        plt.show = orig
    return cap


def embed(cell, cap, preview):
    w, h = Image.open(io.BytesIO(cap["png"])).size
    nw, nh = cap["nominal"]
    Image.open(io.BytesIO(cap["png"])).save(preview)
    cell["outputs"] = [{
        "data": {
            "image/png": base64.b64encode(cap["png"]).decode("ascii"),
            "text/plain": [f"<Figure size {nw}x{nh} with {cap['naxes']} Axes>"],
        },
        "metadata": {"image/png": {"height": h, "width": w}},
        "output_type": "display_data",
    }]
    cell["execution_count"] = 1
    print(f"  {cell['id']}: {w}x{h}  ->  {preview}")


def main():
    raw = open(NB, "rb").read()
    nb = json.loads(raw)
    cell = next(c for c in nb["cells"] if c.get("id") == CELL_ID)
    print(f"{NB}  {CELL_ID}: rendering montage with FLUX.1-schnell ...")
    embed(cell, render("".join(cell["source"])), f"/tmp/_{CELL_ID}.png")
    text = json.dumps(nb, ensure_ascii=raw.isascii(), indent=1)
    if raw.endswith(b"\n"):
        text += "\n"
    open(NB, "w").write(text)
    print(f"  wrote {NB}")


if __name__ == "__main__":
    main()
