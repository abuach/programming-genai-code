"""Generate a grid collage of differently-styled snake game screenshots using SD-Turbo."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chapters'))

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from genai.imagegen import generate_image

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images', 'prompting')
os.makedirs(OUT_DIR, exist_ok=True)

STYLES = [
    ("Retro Arcade",   "retro 1980s arcade snake game screenshot, green phosphor snake on black CRT screen, visible scanlines, pixel grid, classic high score display, authentic arcade cabinet look"),
    ("Neon Cyberpunk", "neon synthwave snake game screenshot, glowing electric blue and magenta snake on dark city grid, neon light reflections, 80s retro-futuristic aesthetic, authentic game UI"),
    ("Snake.io",       "snake.io multiplayer browser game screenshot, colorful glowing worm snakes competing on dark background, multiple players visible, smooth rounded neon snakes, modern web game"),
    ("Nokia LCD",      "Nokia 3310 snake game on monochrome LCD screen, grainy black pixel snake on grey-green backlit display, tiny mobile phone screen, classic 1998 mobile game, realistic photo"),
    ("Atari Vector",   "Atari vector graphics snake game, glowing thin vector line snake on black CRT phosphor screen, retro 1979 arcade machine, bright white-green vector lines, authentic scanlines"),
    ("DOS Terminal",   "MS-DOS text mode snake game screenshot, monochrome green characters on black screen, ASCII box-drawing borders, blinking cursor, authentic 1980s IBM PC terminal game"),
]

SIZE = 512
COLS, ROWS = 3, 2

print(f"Generating {len(STYLES)} images with SD-Turbo...")
images = []
for i, (label, prompt) in enumerate(STYLES):
    print(f"  [{i+1}/{len(STYLES)}] {label}...")
    img = generate_image(prompt, steps=2, seed=42 + i, size=SIZE)
    images.append((label, img))

# Build grid collage with matplotlib
PAD = 6
LABEL_H = 22
fig_w = COLS * SIZE + (COLS + 1) * PAD
fig_h = ROWS * (SIZE + LABEL_H) + (ROWS + 1) * PAD
dpi = 96
fig, axes = plt.subplots(ROWS, COLS, figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi)
fig.patch.set_facecolor('#1a1a2e')
plt.subplots_adjust(left=PAD/fig_w, right=1-PAD/fig_w,
                    top=1-PAD/fig_h, bottom=PAD/fig_h,
                    wspace=PAD/(SIZE+PAD), hspace=(PAD+LABEL_H)/(SIZE+LABEL_H+PAD))

for idx, (ax, (label, img)) in enumerate(zip(axes.flat, images)):
    ax.imshow(img)
    ax.set_title(label, fontsize=10, color='#e0e0e0', pad=4,
                 fontfamily='monospace', fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(False)

out_path = os.path.join(OUT_DIR, 'snake_collage.png')
fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
