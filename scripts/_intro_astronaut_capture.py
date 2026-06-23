"""Replace the intro.ipynb image marker with a FLUX.1-schnell frontispiece.

The marker cell (id 63afe172) holds just the prompt text
`astronaut riding a dinosaur through the desert into the sunset`. We swap it
for a frozen FLUX image cell + a {cite} caption cell, modeled byte-for-byte on
the existing `intro-engineer-img` / `intro-engineer-caption` pair: matplotlib
render at figsize=(6, 6) -> 600x600 PNG, display_data output, freeze tag.

Self-contained: plain json (never nbformat, which reformats the whole file),
edits only the marker, writes back preserving the file's encoding so unrelated
cells stay byte-identical.

Run:  uv run python scripts/_intro_astronaut_capture.py
"""
import base64
import io
import json
import sys

sys.path.insert(0, "chapters")  # so `from genai import ...` resolves from repo root

NB = "chapters/intro.ipynb"
MARKER_ID = "63afe172"
PROMPT = "astronaut riding a dinosaur through the desert into the sunset"
SEED = 0

# Source mirrors intro-engineer-img exactly (same imports, figsize, axis off).
SOURCE = [
    "from genai.imagegen import generate_image\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "astronaut = generate_image(\n",
    f'    "{PROMPT}",\n',
    '    model="flux", seed=0, size=512,\n',
    ")\n",
    "fig, ax = plt.subplots(1, 1, figsize=(6, 6))\n",
    "ax.imshow(astronaut)\n",
    "ax.axis('off')\n",
    "plt.tight_layout(pad=0)\n",
    "plt.show()",
]

CAPTION = [
    "*Image generated with FLUX.1-schnell {cite}`flux2024schnell`. "
    f'Prompt: "{PROMPT}".*'
]


def main():
    raw = open(NB, "rb").read()
    ensure_ascii = raw.isascii()
    trailing_nl = raw.endswith(b"\n")
    nb = json.loads(raw)
    cells = nb["cells"]

    idx = next(i for i, c in enumerate(cells) if c.get("id") == MARKER_ID)
    print(f"marker cell {MARKER_ID} at index {idx}")

    # Generate the FLUX image (seed-pinned) and render it through the exact same
    # matplotlib path the cell code uses, so the saved PNG matches a live run.
    print("\nLoading FLUX.1-schnell pipeline and generating (seed=0)...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from genai.imagegen import generate_image

    img = generate_image(PROMPT, model="flux", seed=SEED, size=512)
    print(f"  generated {img.size[0]}x{img.size[1]} {img.mode} image")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")  # dpi defaults to figure dpi (100) -> 600x600
    plt.close(fig)
    png = buf.getvalue()
    b64 = base64.b64encode(png).decode("ascii")

    from PIL import Image
    w, h = Image.open(io.BytesIO(png)).size
    Image.open(io.BytesIO(png)).save("/tmp/_flux_astronaut_dino.png")  # preview only
    print(f"  rendered {w}x{h} png ({len(png)} bytes) | preview: "
          "/tmp/_flux_astronaut_dino.png")

    output = {
        "data": {
            "image/png": b64,
            "text/plain": ["<Figure size 600x600 with 1 Axes>"],
        },
        "metadata": {"image/png": {"height": h, "width": w}},
        "output_type": "display_data",
    }
    code_cell = {
        "cell_type": "code",
        "execution_count": 1,
        "id": "intro-astronaut-img",
        "metadata": {"tags": ["freeze"]},
        "outputs": [output],
        "source": SOURCE,
    }
    caption_cell = {
        "cell_type": "markdown",
        "id": "intro-astronaut-caption",
        "metadata": {},
        "source": CAPTION,
    }

    cells[idx:idx + 1] = [code_cell, caption_cell]
    print(f"  replaced marker with intro-astronaut-img + intro-astronaut-caption")

    text = json.dumps(nb, ensure_ascii=ensure_ascii, indent=1)
    if trailing_nl:
        text += "\n"
    with open(NB, "w") as f:
        f.write(text)
    print(f"\nWrote {NB} (ensure_ascii={ensure_ascii}, trailing_nl={trailing_nl})")


if __name__ == "__main__":
    sys.exit(main())
