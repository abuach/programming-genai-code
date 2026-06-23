"""Probe: can a modern vision-language model read a terminal screenshot the
older one only guesses at?

The Multimodal chapter's "Reading Error Screenshots" demo lands one lesson with
the same model on two renders: a legible screenshot is read correctly, an
illegible one is not, so resolution is part of the prompt. This probe supplies
the other half. Hand the *same legible* screenshot (images/mm/terminal.png, a
Python KeyError traceback) to three vision models and ask each to transcribe it:

  - llava:latest    : the chapter's incumbent vision foil
  - qwen3-vl:8b      : Alibaba's stronger VLM (OCR across 32 languages, screen UI)
  - llava-phi3:3.8b  : the small tier

Finding (nondeterministic, vision sampling): qwen3-vl returns the real traceback
line for line, llava confabulates an unrelated pygame program that is nowhere on
the screen, and the small llava-phi3 reads the text but stutters on a token. The
captured outputs are baked into genai.vision.TERMINAL_TRANSCRIPTS and shown in a
frozen notebook cell; rerun this to regenerate them.

Vision calls are slow (~70s each) and the result drifts run to run; we stop each
model after use to keep memory free on this Mac. Run:
    uv run python scripts/_qwen3vl_probe.py
"""
import sys, time, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "chapters"))
from genai.vision import ask_image

IMG = (REPO / "chapters" / "images" / "mm" / "terminal.png").read_bytes()
PROMPT = "Transcribe the text in this terminal screenshot exactly."
MODELS = ["llava:latest", "qwen3-vl:8b", "llava-phi3:3.8b"]


def stop(model):
    subprocess.run(["ollama", "stop", model], capture_output=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    for model in MODELS:
        try:
            tic = time.perf_counter()
            out = ask_image(IMG, PROMPT, model=model)
            dt = time.perf_counter() - tic
            print("=" * 72)
            print(f"{model}   ({dt:.1f}s)")
            print("=" * 72)
            print(out)
            print()
        except Exception as e:
            print(f"{model}: ERROR {type(e).__name__}: {e}")
        stop(model)
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min")
