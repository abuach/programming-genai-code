"""Regenerate chapters/images/mm/terminal.png (Figure: Python traceback in a terminal).

The text content must stay exactly as-is: the frozen ask_image cells in mm.ipynb
diagnose this specific traceback (KeyError 'key', app.py line 17, process_data),
and the prose around them describes those diagnoses.

Note: images/mm/terminal_small.png is the original low-res render (860x480, tiny
default PIL font). It is kept ON PURPOSE as act 1 of the legibility contrast demo
(gemma3 misreads 'fn' as 'irn' on it) — do not regenerate or "fix" it.
"""
from PIL import Image, ImageDraw, ImageFont

OUT = "chapters/images/mm/terminal.png"
FONT = "/System/Library/Fonts/Menlo.ttc"

BG = "#262b3b"          # terminal body
BAR = "#1b1f2c"         # title bar
TEXT = "#d4d8e2"        # normal output
RED = "#f2655c"         # error lines
TITLE = "#9aa0b0"       # title bar label
LIGHTS = ["#ff5f57", "#febc2e", "#28c840"]

# (text, color) per line; None text = blank line
LINES = [
    ("$ python3 app.py", TEXT),
    (None, TEXT),
    ("Traceback (most recent call last):", RED),
    ('  File "app.py", line 42, in <module>', TEXT),
    ("    result = process_data(df, config)", TEXT),
    ('  File "app.py", line 17, in process_data', TEXT),
    ("    return df.groupby(config['key']).agg(config['fn'])", TEXT),
    ("KeyError: 'key'", RED),
    (None, TEXT),
    ("$ _", TEXT),
]

S = 2                   # supersample factor for crisp PDF rendering
FS = 32 * S             # font size
PAD = 40 * S            # body padding
BAR_H = 56 * S
LINE_H = int(FS * 1.55)
RADIUS = 18 * S

font = ImageFont.truetype(FONT, FS)
title_font = ImageFont.truetype(FONT, int(FS * 0.72))

longest = max((t for t, _ in LINES if t), key=len)
text_w = int(font.getlength(longest))
W = text_w + 2 * PAD
H = BAR_H + 2 * PAD + LINE_H * len(LINES)

img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([0, 0, W - 1, H - 1], radius=RADIUS, fill=BG)
d.rounded_rectangle([0, 0, W - 1, BAR_H + RADIUS], radius=RADIUS, fill=BAR)
d.rectangle([0, BAR_H, W - 1, BAR_H + RADIUS], fill=BG)

for i, c in enumerate(LIGHTS):
    cx = PAD // 2 + 18 * S + i * 30 * S
    r = 9 * S
    d.ellipse([cx - r, BAR_H // 2 - r, cx + r, BAR_H // 2 + r], fill=c)

title = "python3 — 80×24"
tw = title_font.getlength(title)
d.text(((W - tw) // 2, (BAR_H - title_font.size) // 2), title, font=title_font, fill=TITLE)

y = BAR_H + PAD
for text, color in LINES:
    if text:
        d.text((PAD, y), text, font=font, fill=color)
    y += LINE_H

img.save(OUT)
print(f"wrote {OUT}: {W}x{H}")
