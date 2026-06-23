"""Generate the static figures for the mm.ipynb text/spatial edits.
Deterministic, no model calls. Run from repo root with PYTHONPATH=chapters."""
import io, sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from genai import mnist_digits

MM = "chapters/images/mm"


def shapes_row():
    colors = ["red", "green", "blue", "orange", "purple"]
    W, H, r = 1100, 240, 72
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    step = W // (len(colors) + 1)
    for i, c in enumerate(colors):
        cx, cy = step * (i + 1), H // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c, outline="black", width=4)
    img.save(f"{MM}/shapes_row.png")
    print("wrote shapes_row.png")


# gemma4's real per-digit reading of mnist_digits(25, seed=7) from the captured
# run that the notebook embeds: 22/25, missing 5->7, 8->4, 9->7. Overridden at
# runtime by chapters/.mm_capture.json when present (re-run scripts/_mm_capture.py).
GEMMA4_PREDS = [0, 6, 1, 7, 7, 6, 4, 1, 3, 7, 7, 4, 3, 0, 3, 4,
                5, 3, 9, 7, 8, 4, 0, 7, 6]


def mnist_grid():
    digits = mnist_digits(n=25, seed=7, root="chapters/.cache_mnist")
    preds = GEMMA4_PREDS
    try:                                   # prefer the authoritative captured run
        import json
        cap = json.load(open("chapters/.mm_capture.json"))
        preds = [int(p) if str(p).isdigit() else -1 for p in cap["gemma4_preds"]]
    except FileNotFoundError:
        pass
    fig, axes = plt.subplots(5, 5, figsize=(7.2, 7.8))
    for ax, (b, label), pred in zip(axes.flat, digits, preds):
        ax.imshow(Image.open(io.BytesIO(b)))
        ok = pred == label
        for s in ax.spines.values():
            s.set_color("#2e7d32" if ok else "#c62828"); s.set_linewidth(3.5)
        ax.set_title(f"read {pred}" if not ok else f"{label}",
                     fontsize=12, color="#222" if ok else "#c62828",
                     fontweight="normal" if ok else "bold")
        ax.set_xticks([]); ax.set_yticks([])
    hits = sum(p == l for (_, l), p in zip(digits, preds))
    fig.suptitle(f"gemma4 reads MNIST: {hits}/25 correct", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{MM}/mnist_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote mnist_grid.png ({hits}/25)")


def text_img(text, font_path, idx=0, size=64, pad=34):
    font = ImageFont.truetype(font_path, size, index=idx)
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), "white")
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill="black")
    return img


def cursive():
    phrase = "the model reads my handwriting"
    snell = text_img(phrase, "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc", size=64)
    zapf = text_img(phrase, "/System/Library/Fonts/Supplemental/Zapfino.ttf", size=52)
    snell.save(f"{MM}/text_cursive.png")
    zapf.save(f"{MM}/text_ornate.png")
    # combined figure: the two styles stacked on one white canvas
    pad, gap = 30, 24
    W = max(snell.width, zapf.width) + 2 * pad
    H = snell.height + zapf.height + gap + 2 * pad
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(snell, ((W - snell.width) // 2, pad))
    canvas.paste(zapf, ((W - zapf.width) // 2, pad + snell.height + gap))
    canvas.save(f"{MM}/text_handwriting.png")
    print("wrote text_cursive.png, text_ornate.png, text_handwriting.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["shapes_row", "mnist_grid", "cursive"]
    for w in which:
        globals()[w]()
