#!/usr/bin/env python3
"""Re-render self-contained genai.viz diagram cells and re-embed their PNG.

After editing a plotting function in ``chapters/genai/viz.py``, the diagram
images embedded in the notebooks are stale. Re-executing the whole notebook
would re-run its (slow, non-deterministic) LLM cells. The schematic diagram
cells, though, are self-contained — typically just::

    ##system
    from genai.viz import plot_rag_architecture
    plot_rag_architecture()

so we can run only those, capture the figure, and write a fresh base64 PNG back
into the cell's outputs. This mirrors how the Jupyter inline backend embeds a
matplotlib figure (display_data with image/png + a "<Figure ...>" text rep).

Usage:
    # re-render every cell that calls one of the named functions
    python scripts/rerender_diagrams.py chapters/augmentation.ipynb \
        plot_rag_architecture

    # re-render every self-contained viz.plot_* cell in the notebook
    python scripts/rerender_diagrams.py chapters/responsible.ipynb --all
"""
import base64
import io
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402

# Source dpi for the embedded raster. The PDF scales every figure to a fixed
# 0.95\linewidth regardless of pixel size, so dpi only controls print sharpness;
# 130 keeps small label text crisp without bloating the notebook JSON.
RENDER_DPI = 130

CALL_RE = re.compile(r"\b(plot_[A-Za-z0-9_]+)\s*\(")


def png_display_output(fig, execution_count):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI,
                bbox_inches="tight", facecolor="white")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    # width/height in the text rep is cosmetic; pull true px from the PNG header.
    w = int.from_bytes(raw[16:20], "big")
    h = int.from_bytes(raw[20:24], "big")
    return nbformat.v4.new_output(
        "display_data",
        data={"image/png": b64,
              "text/plain": [f"<Figure size {w}x{h} with 1 Axes>"]},
        metadata={},
    )


def cell_target_fns(src, wanted):
    fns = set(CALL_RE.findall(src))
    if wanted is None:          # --all: any viz plot call
        return fns
    return fns & set(wanted)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    nb_path = Path(sys.argv[1]).resolve()
    rest = sys.argv[2:]
    wanted = None if rest == ["--all"] else rest

    chapters_dir = nb_path.parent
    sys.path.insert(0, str(chapters_dir))
    os.chdir(chapters_dir)            # so genai imports + image save paths resolve

    nb = nbformat.read(str(nb_path), as_version=4)

    # Keep execution counts monotonic and unique across the cells we touch.
    used = [c.get("execution_count") for c in nb.cells
            if c.get("cell_type") == "code" and c.get("execution_count")]
    next_ec = (max(used) + 1) if used else 1

    n = 0
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if not cell_target_fns(src, wanted):
            continue

        plt.close("all")
        ns = {}
        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)
        except Exception as e:                       # noqa: BLE001
            print(f"  [skip] cell {idx}: {type(e).__name__}: {e}")
            continue

        fig = plt.gcf()
        if not fig.get_axes():
            print(f"  [skip] cell {idx}: produced no figure")
            continue

        ec = cell.get("execution_count") or next_ec
        if cell.get("execution_count") is None:
            next_ec += 1
        cell["outputs"] = [png_display_output(fig, ec)]
        cell["execution_count"] = ec
        plt.close(fig)
        n += 1
        fn = ", ".join(sorted(cell_target_fns(src, wanted)))
        print(f"  [ec={ec}] re-rendered cell {idx}: {fn}")

    nbformat.write(nb, str(nb_path))
    print(f"done: re-rendered {n} cell(s) in {nb_path.name}")


if __name__ == "__main__":
    main()
