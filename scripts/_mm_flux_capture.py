"""Regenerate the mm.ipynb astronaut image with FLUX.1-schnell and apply the
accompanying text fixes (SDXL-Turbo -> FLUX naming + 4 prose typos).

Self-contained: loads the notebook with plain json (never nbformat, which
reformats the whole file), edits only the targeted cells, and writes back
preserving the file's existing encoding so unrelated cells stay byte-identical.

Run:  uv run python scripts/_mm_flux_capture.py
"""
import base64
import io
import json
import sys

sys.path.insert(0, "chapters")  # so `from genai import ...` resolves from repo root

NB = "chapters/mm.ipynb"
PROMPT = ("an astronaut riding a horse on the moon, "
          "photorealistic, dramatic lighting")

# (cell index, old substring, new substring) -- each must hit exactly once.
TEXT_PATCHES = [
    # --- FLUX naming / step-count / citation ---
    (62, "here, SD-Turbo, has been distilled so aggressively that it needs "
         "only one or two of those steps",
         "here, FLUX.1-schnell, has been distilled so aggressively that it "
         "needs only a handful of those steps"),
    (62, "which is why it can draw on a laptop in a few seconds "
         "{cite}`sauer2023adversarial`",
         "which is why it can draw on a laptop in a few seconds "
         "{cite}`flux2024schnell`"),
    (64, "generated with SDXL-Turbo {cite}`sauer2023adversarial`",
         "generated with FLUX.1-schnell {cite}`flux2024schnell`"),
    (77, "like the SD-Turbo model from the generation section",
         "like the FLUX.1-schnell model from the generation section"),
    # --- typos ---
    (2, "This one of the most immediately useful",
        "This is one of the most immediately useful"),
    (17, "going on:T the model", "going on: the model"),
    (25, "*how many* amonst other things", "*how many* amongst other things"),
    (25, "generally increase with the size of the mode.",
         "generally increases with the size of the model."),
    # --- image cell: switch backend ---
    (63, 'model="sdxl-turbo"', 'model="flux"'),
]


def patch_source(cell, old, new):
    """Replace `old` with `new` in the single source line that contains it,
    leaving the rest of the source array byte-identical."""
    hits = 0
    for i, line in enumerate(cell["source"]):
        if old in line:
            cell["source"][i] = line.replace(old, new)
            hits += line.count(old)
    return hits


def main():
    raw = open(NB, "rb").read()
    ensure_ascii = raw.isascii()
    trailing_nl = raw.endswith(b"\n")
    nb = json.loads(raw)
    cells = nb["cells"]

    # 1) Apply every text patch, asserting each lands exactly once.
    for idx, old, new in TEXT_PATCHES:
        n = patch_source(cells[idx], old, new)
        assert n == 1, f"cell {idx}: expected 1 hit for {old!r}, got {n}"
        print(f"  patched cell {idx}: {old[:42]!r} -> {new[:42]!r}")

    # 2) Generate the FLUX image (seed-pinned, deterministic).
    print("\nLoading FLUX.1-schnell pipeline and generating (seed=0)...")
    from genai import generate_image
    img = generate_image(PROMPT, model="flux", seed=0, size=512)
    w, h = img.size
    print(f"  generated {w}x{h} {img.mode} image")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    b64 = base64.b64encode(png).decode("ascii")
    img.save("/tmp/_flux_astronaut.png")  # for visual verification only
    print(f"  png bytes: {len(png)}  | preview: /tmp/_flux_astronaut.png")

    # 3) Embed into cell 63's existing display_data, changing ONLY the bytes.
    out = next(o for o in cells[63]["outputs"]
               if o.get("output_type") == "display_data")
    existing = out["data"]["image/png"]
    out["data"]["image/png"] = [b64] if isinstance(existing, list) else b64
    # update the inline repr + metadata to match the regenerated image size
    out["data"]["text/plain"] = [f"<PIL.Image.Image image mode={img.mode} size={w}x{h}>"]
    out["metadata"]["image/png"] = {"height": h, "width": w}
    assert cells[63]["execution_count"] is not None

    # 4) Write back, preserving the file's original JSON style.
    text = json.dumps(nb, ensure_ascii=ensure_ascii, indent=1)
    if trailing_nl:
        text += "\n"
    with open(NB, "w") as f:
        f.write(text)
    print(f"\nWrote {NB} (ensure_ascii={ensure_ascii}, trailing_nl={trailing_nl})")


if __name__ == "__main__":
    sys.exit(main())
