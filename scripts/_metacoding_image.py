"""Regenerate the metacoding opener image with FLUX.1-schnell (book default)
and switch the caption from SDXL-Turbo to FLUX.

Mirrors scripts/_mm_flux_capture.py: plain-json load, targeted byte-stable
edits, FLUX regen embedded into the frozen opener cell (cell 3). The cell stays
frozen; only its image bytes + size metadata change.

Run:  uv run python scripts/_metacoding_image.py
"""
import base64
import io
import json
import sys

sys.path.insert(0, "chapters")  # so `from genai import ...` resolves

NB = "chapters/metacoding.ipynb"
PROMPT = (
    "photorealistic technical pencil drawing of machines building other machines, "
    "robotic arms on a factory floor assembling robot bodies, "
    "intricate mechanical detail, cinematic industrial lighting, monochrome"
)

# (cell index, old, new) -- each must hit exactly once.
TEXT_PATCHES = [
    (3, 'model="sdxl-turbo"', 'model="flux"'),
    (3, "size=1024", "size=512"),  # FLUX at 1024 OOMs MPS; 512 is the book default
    (4, "Image generated with SDXL-Turbo {cite}`sauer2023adversarial`",
        "Image generated with FLUX.1-schnell {cite}`flux2024schnell`"),
]


def patch_source(cell, old, new):
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

    # sanity: cells 3/4 are the image + caption
    assert "generate_image" in "".join(cells[3]["source"]), "cell 3 not image"
    assert "Image generated" in "".join(cells[4]["source"]), "cell 4 not caption"

    for idx, old, new in TEXT_PATCHES:
        n = patch_source(cells[idx], old, new)
        assert n == 1, f"cell {idx}: expected 1 hit for {old!r}, got {n}"
        print(f"  patched cell {idx}: {old[:40]!r} -> {new[:40]!r}")

    print("\nLoading FLUX.1-schnell and generating (seed=7, size=512)...")
    from genai import generate_image
    img = generate_image(PROMPT, model="flux", seed=7, size=512)
    w, h = img.size
    print(f"  generated {w}x{h} {img.mode} image")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    b64 = base64.b64encode(png).decode("ascii")
    img.save("/tmp/_flux_metacoding.png")  # for visual verification
    print(f"  png bytes: {len(png)}  | preview: /tmp/_flux_metacoding.png")

    out = next(o for o in cells[3]["outputs"]
               if o.get("output_type") == "display_data")
    existing = out["data"]["image/png"]
    out["data"]["image/png"] = [b64] if isinstance(existing, list) else b64
    out["data"]["text/plain"] = [
        f"<PIL.Image.Image image mode={img.mode} size={w}x{h}>"]
    out.setdefault("metadata", {})["image/png"] = {"height": h, "width": w}
    assert cells[3]["execution_count"] is not None

    text = json.dumps(nb, ensure_ascii=ensure_ascii, indent=1)
    if trailing_nl:
        text += "\n"
    with open(NB, "w") as f:
        f.write(text)
    print(f"\nWrote {NB} "
          f"(ensure_ascii={ensure_ascii}, trailing_nl={trailing_nl})")


if __name__ == "__main__":
    sys.exit(main())
