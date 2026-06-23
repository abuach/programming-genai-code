"""Can FLUX.1-schnell draw a true Penrose/Escher impossible staircase?

The round-3 "recursive_stairs" attempt came back as a handsome but ORDINARY
spiral: FLUX rendered the *look* of endlessness, not the logical impossibility,
because an impossible object is a relational/global property (the same weakness
behind horse-riding-astronaut). This probes harder: explicit "Penrose / impossible
/ isometric / Escher" cues across two phrasings and a few seeds. If none produce a
genuine impossible loop, that is itself the finding.

Seed-pinned + FLUX.1-schnell so any keeper is reproducible. ~2 min/image on MPS
after the pipeline loads; skip-existing so re-runs are cheap.

Run:  uv run python scripts/_penrose_probe.py
"""
import os
import sys
import time

sys.path.insert(0, "chapters")

OUT = "/tmp/penrose"

P1 = ("The Penrose impossible staircase, a closed square loop of four stone "
      "stair-flights that appears to climb forever with no top or bottom, "
      "isometric optical illusion, dramatic raking light, ultra-detailed")
P2 = ("An impossible staircase in the style of M.C. Escher's lithograph "
      "Ascending and Descending, tiny figures walking a rectangular loop of "
      "stairs that never rises or falls, isometric perspective, moody "
      "monochrome, dramatic shadows")

# (slug, prompt, seed)
ATTEMPTS = [
    ("penrose_p1_s1", P1, 1),
    ("penrose_p1_s2", P1, 2),
    ("penrose_p1_s3", P1, 3),
    ("penrose_p2_s1", P2, 1),
    ("penrose_p2_s2", P2, 2),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    from genai import generate_image
    for slug, prompt, seed in ATTEMPTS:
        path = f"{OUT}/{slug}.png"
        if os.path.exists(path):
            print(f"  {slug:16} exists, skipping", flush=True)
            continue
        t0 = time.time()
        img = generate_image(prompt, model="flux", seed=seed, size=512)
        img.save(path)
        print(f"  {slug:16} seed={seed}  {time.time()-t0:5.1f}s  -> {path}", flush=True)


if __name__ == "__main__":
    main()
