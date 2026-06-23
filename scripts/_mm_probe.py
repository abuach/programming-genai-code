"""Reproducer for mm.ipynb edits: chess count (gemma4:latest vs llava) and
cursive/ornate reading (gemma4:latest vs llava). Run from chapters/."""
from functools import partial
from genai import ask_image

# --- Chess piece count: swap gemma4:e2b -> gemma4:latest, keep llava contrast.
q = "How many chess pieces total are visible in this image? Just the number."
img = open("images/mm/chess.jpg", "rb").read()
print("=== chess count (true answer: 6) ===")
for model in ["gemma4:latest", "llava:latest"]:
    reply = partial(ask_image, model=model)(img, q).strip()
    print(f"{model:18} -> {reply[:90]!r}")

# --- Cursive + ornate reading: gemma4:latest vs llava.
prompt = "Transcribe the handwriting in this image exactly. Reply with only the words."
print("\n=== handwriting reading (truth: 'the model reads my handwriting') ===")
for model in ["gemma4:latest", "llava:latest"]:
    read = partial(ask_image, model=model)
    for style in ["cursive", "ornate"]:
        b = open(f"images/mm/text_{style}.png", "rb").read()
        print(f"{model:14} {style:8} -> {read(b, prompt)!r}")
