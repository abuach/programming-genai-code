"""Reproducer for the four FLUX chapter-opener images:

  * Foundations  found-orrery-img  -- orrery vs cosmos ("all models are wrong")
  * Foundations  a56b47a8          -- the Jigsaw/LEGO panel (FLUX, modernized)
  * Prompting    prompting-socrates-img -- a marble bust of Socrates
  * Multimodal   mm-galaxy-eye-img -- a constellation that vaguely forms an eye

Each cell is frozen, so execute_notebook.py skips it; this script regenerates
the embedded image bytes in place. It executes each cell's own source (the
verbatim FLUX prompt lives there) with plt.show() patched to capture the figure,
then writes the PNG back byte-stably. Idempotent: locates cells by id, re-renders.

Run:  uv run python scripts/_opener_images.py
"""
import base64
import io
import json
import sys

sys.path.insert(0, "chapters")  # so `from genai.imagegen import ...` resolves

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

TARGETS = {
    "chapters/intro.ipynb": ["found-orrery-img", "a56b47a8"],
    "chapters/prompting.ipynb": ["prompting-socrates-img"],
    "chapters/mm.ipynb": ["mm-galaxy-eye-img"],
}


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
    for path, ids in TARGETS.items():
        raw = open(path, "rb").read()
        nb = json.loads(raw)
        by_id = {c.get("id"): c for c in nb["cells"]}
        for cid in ids:
            cell = by_id[cid]
            print(f"{path}  {cid}: rendering with FLUX.1-schnell ...")
            cap = render("".join(cell["source"]))
            embed(cell, cap, f"/tmp/_opener_{cid}.png")
        text = json.dumps(nb, ensure_ascii=raw.isascii(), indent=1)
        if raw.endswith(b"\n"):
            text += "\n"
        open(path, "w").write(text)
        print(f"  wrote {path}\n")


if __name__ == "__main__":
    main()
