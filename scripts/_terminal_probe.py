"""Reproducer for the mm.ipynb error-screenshot legibility contrast (Reading Error Screenshots).

Two acts, same prompt, same model (gemma3 via ask_image):
- terminal_small.png (860x480, ~11px text): the model can't resolve the pixels and
  fills the smudge with a plausible guess — it invents a key ('irn' for 'fn') and
  diagnoses an error that isn't on the screen.
- terminal.png (legible re-render via gen_terminal_png.py, same text content):
  every character lands (KeyError 'key', the groupby call), though it still pins
  the error on line 42, the outer call site, instead of line 17 in process_data.

Run: .venv/bin/python scripts/_terminal_probe.py
"""
import sys
sys.path.insert(0, "chapters")
from genai.vision import ask_image

M = 6
PROMPT = ("This is a terminal screenshot of a Python error. In 3-4 "
          "sentences: what error occurred, which line caused it, and "
          "what is the most likely fix?")

for name in ("terminal_small.png", "terminal.png"):
    img = open(f"chapters/images/mm/{name}", "rb").read()
    runs = [ask_image(img, PROMPT) for _ in range(M)]
    print(f"=== {name}")
    print(f"  invents 'irn':                  {sum('irn' in r for r in runs)}/{M}")
    print(f"  reads KeyError: 'key':          {sum(chr(39)+'key'+chr(39) in r for r in runs)}/{M}")
    print(f"  traces line 17 in process_data: {sum('17' in r and 'process_data' in r for r in runs)}/{M}")
    print(f"  points at line 42 (call site):  {sum('42' in r for r in runs)}/{M}")
    print(f"--- full transcript of run 1 (embed this) ---")
    print(runs[0])
    print()
