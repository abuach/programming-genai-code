"""Exploration: generate a batch of abstract "what diffusion does well" images.

These probe the prompt formula
    [natural phenomenon] + [physical material] + [mathematical structure] + [cinematic lighting]
for a possible "what diffusion does well" beat in mm.ipynb's Generating Images
section (which currently only shows the relationship-flip *failure*). We render a
spread, eyeball the real outputs, and keep the best two or three.

All seed-pinned (seed=0) + FLUX.1-schnell so any adopted image is reproducible.

Run:  uv run python scripts/_diffusion_gallery_probe.py
"""
import sys
import time

sys.path.insert(0, "chapters")  # so `from genai import ...` resolves from repo root

OUT = "/tmp/diffusion_gallery"

# (slug, prompt, seed). The first two are the user's verbatim formula examples.
# fungal_glass at seed=0 collapsed to pure static (schnell non-convergence), so
# fungal_glass_s1 retries the same prompt at seed=1; the original is kept on disk
# as the failure example. ~8 min/image on MPS, so re-runs skip finished files.
PROMPTS = [
    ("fungal_glass",
     "Bioluminescent fungal network made of translucent glass fibers, "
     "hyperbolic geometry, volumetric lighting, atmospheric fog, "
     "ultra-detailed macro photography", 0),
    ("fungal_glass_s1",
     "Bioluminescent fungal network made of translucent glass fibers, "
     "hyperbolic geometry, volumetric lighting, atmospheric fog, "
     "ultra-detailed macro photography", 1),
    ("neural_metal",
     "Flowing liquid metal forming a neural network, emergent topology, "
     "iridescent reflections, dark background, cinematic contrast", 0),
    ("brain_coral",
     "A human brain made of growing coral reefs, bioluminescent translucent "
     "polyps, volumetric lighting, ultra-detailed macro photography, "
     "dark background", 0),
    ("concept_landscape",
     "A vast landscape made entirely from interconnected glowing concepts, "
     "neural pathways morphing into forests, rivers of light, atmospheric fog, "
     "cinematic lighting", 0),
    ("silk_galaxy",
     "A galaxy woven from silk threads, iridescent filaments, emergent spiral "
     "structure, dark background, cinematic contrast, ultra-detailed", 0),
    ("crystal_wave",
     "An ocean wave frozen into crystalline geometry, faceted glass, refracted "
     "light, volumetric lighting, ultra-detailed macro photography", 0),
    # --- round 2: new categories (architecture+biology, physics+fluid, geometry) ---
    ("cathedral_mycelium",
     "A vast cathedral grown from living mycelium, branching fungal architecture, "
     "translucent membranes between stone arches, volumetric god rays, "
     "atmospheric fog, cinematic lighting, ultra-detailed", 0),
    ("fluid_turbulence",
     "Colored fluids frozen mid-turbulence, swirling ink and milk in water, "
     "iridescent fractal eddies, dark background, cinematic contrast, "
     "ultra-detailed macro photography", 0),
    ("crystal_forest",
     "A forest of translucent crystals growing in impossible geometries, faceted "
     "quartz trees, refracted light, volumetric lighting, atmospheric fog, "
     "ultra-detailed", 0),
    # --- round 3: blooming intelligence + architectural dreams ---
    ("neural_flower",
     "A neural network blooming like a flower, glowing synaptic petals unfurling, "
     "iridescent filaments, dark background, volumetric lighting, ultra-detailed "
     "macro photography", 0),
    ("infinite_trees",
     "Infinite fractal structures growing like trees, branching architecture "
     "repeating at every scale, glowing filigree, atmospheric fog, cinematic "
     "lighting, ultra-detailed", 0),
    ("crystal_city",
     "A city suspended within a giant crystal lattice, faceted glass towers "
     "embedded in geometric crystal, refracted light, volumetric lighting, dark "
     "background, ultra-detailed", 0),
    ("dissolving_buildings",
     "Skyscrapers dissolving into clouds, towers fading into swirling mist at "
     "their peaks, soft volumetric light, atmospheric fog, cinematic, "
     "ultra-detailed", 0),
    ("recursive_stairs",
     "An impossible recursive staircase with no beginning or end, Escher-like "
     "interlocking stone steps, looping architecture, dramatic shadows, "
     "volumetric lighting, ultra-detailed", 0),
]


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    from genai import generate_image

    for slug, prompt, seed in PROMPTS:
        path = f"{OUT}/{slug}.png"
        if os.path.exists(path):
            print(f"  {slug:18} exists, skipping", flush=True)
            continue
        t0 = time.time()
        img = generate_image(prompt, model="flux", seed=seed, size=512)
        img.save(path)
        print(f"  {slug:18} {img.size}  seed={seed}  {time.time()-t0:5.1f}s  -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
