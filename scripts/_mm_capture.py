"""Authoritative capture for the mm.ipynb edits: runs the FINAL cell code once
(cwd=chapters, like the notebook) and writes chapters/.mm_capture.json with the
exact stdout to embed in each frozen cell, plus gemma4's per-digit MNIST
predictions so the scorecard figure matches the embedded score.
Run from repo root with PYTHONPATH=chapters."""
import os, io, json, re
from contextlib import redirect_stdout
from functools import partial

os.chdir("chapters")                      # match the notebook's working directory
from genai import ask_image, mnist_digits

out = {}

# ---- spatial: exec the exact cell source, capture stdout ----
SPATIAL = '''
from genai import ask_image
from functools import partial
img = open("images/mm/shapes_row.png", "rb").read()
qs = ["the circle immediately to the right of the green one",
      "the fourth circle from the left"]
for model in ["gemma4:latest", "llava:latest"]:
    answer = partial(ask_image, model=model)
    reply = [answer(img, f"What color is {q}? One word.").strip() for q in qs]
    print(f"{model:14}  right of green: {reply[0]:8}  fourth: {reply[1]}")
'''
buf = io.StringIO()
with redirect_stdout(buf):
    exec(SPATIAL, {})
out["spatial_out"] = buf.getvalue()

# ---- MNIST: instrumented single pass per model (records gemma4 preds) ----
read_digit = lambda t: (re.findall(r"\d", t) or ["?"])[0]
digits = mnist_digits(n=25, seed=7)
prompt = "What single handwritten digit is shown? Reply with only the digit."
lines = []
for model in ["gemma4:latest", "llava:latest"]:
    ask = partial(ask_image, model=model)
    preds = [read_digit(ask(b, prompt)) for b, _ in digits]
    hits = sum(p == str(lbl) for p, (_, lbl) in zip(preds, digits))
    lines.append(f"{model:14} read {hits}/25 correctly ({100 * hits // 25}%)")
    if model.startswith("gemma4"):
        out["gemma4_preds"] = preds
        out["gemma4_hits"] = hits
out["mnist_out"] = "\n".join(lines) + "\n"
out["labels"] = [lbl for _, lbl in digits]

# ---- cursive: exec the exact cell source ----
CURSIVE = '''
from genai import ask_image
from functools import partial
read = partial(ask_image, model="gemma4:latest")
prompt = "Transcribe the handwriting in this image exactly. Reply with only the words."
for style in ["cursive", "ornate"]:
    img = open(f"images/mm/text_{style}.png", "rb").read()
    print(f"{style:8} → {read(img, prompt)!r}")
'''
buf = io.StringIO()
with redirect_stdout(buf):
    exec(CURSIVE, {})
out["cursive_out"] = buf.getvalue()

with open(".mm_capture.json", "w") as f:
    json.dump(out, f, indent=2)
print("=== SPATIAL ===\n" + out["spatial_out"])
print("=== MNIST ===\n" + out["mnist_out"])
print("gemma4 preds:", out["gemma4_preds"])
print("=== CURSIVE ===\n" + out["cursive_out"])
