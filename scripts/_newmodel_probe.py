"""Probe the 18 newly-downloaded models with the book's own harnesses to find
where each one stands out (positively or negatively), before proposing a home.

Batteries reuse existing apparatus so the numbers are comparable to the book:
  - code        : grade_is_prime (the metacoding code-zoo scorer, 12 cases)
  - reasoning   : NOVEL_PROBLEMS + _grade (the thinking chapter, snap vs think)
  - embedding   : GAP_PAIRS related-vs-unrelated gap (semantics chapter)
  - multilingual: aya vs the incumbent gemma4 on low-resource translation (NEW angle)
  - vision      : ask_image transcription on the mm chapter's terminal.png
  - general     : one-sentence explanation + strict format-following

Deterministic where possible (temperature 0, single run). Each model is stopped
after use to keep memory free on this Mac. Run:
    uv run python scripts/_newmodel_probe.py
"""
import os, re, sys, time, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CH   = REPO / "chapters"
sys.path.insert(0, str(CH))

import ollama
from genai.thinking import NOVEL_PROBLEMS, _grade
from genai.code import grade_is_prime
from genai.embed import embed, similarity, GAP_PAIRS
from genai.vision import ask_image

client = ollama.Client(host="http://localhost:11434")
SNAP = " Reply with ONLY the final number, no explanation, no units."


def run(model, prompt, think=False, budget=0, max_tokens=80, system=None):
    """One deterministic pass; return (text, seconds, had_thinking_trace)."""
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": prompt})
    opts = {"temperature": 0, "num_predict": max_tokens + budget}
    t0 = time.perf_counter()
    try:
        resp = client.chat(model=model, messages=msgs, think=think, options=opts)
    except Exception:
        resp = client.chat(model=model, messages=msgs, options=opts)
    dt = time.perf_counter() - t0
    msg = resp["message"]
    return (msg.get("content") or "").strip(), dt, bool(msg.get("thinking"))


def stop(model):
    subprocess.run(["ollama", "stop", model], capture_output=True)


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def hdr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


# ── 1. CODE ZOO (is_prime, 12 cases) ──────────────────────────────────────────
def battery_code():
    hdr("CODE  — is_prime, graded by grade_is_prime (incumbent qwen2.5-coder = 12/12)")
    prompt = ("Write a concise Python function is_prime(n) that returns True if n "
              "is prime. Only code, no explanation.")
    for model in ["codeqwen:7b", "magicoder:7b", "opencoder:8b", "stable-code:3b"]:
        try:
            raw, dt, _ = run(model, prompt, max_tokens=320)
            passed, total, note = grade_is_prime(extract_code(raw))
            print(f"{model:18} {passed:2}/{total}  {dt:5.1f}s  {note}", flush=True)
        except Exception as e:
            print(f"{model:18} ERROR {type(e).__name__}: {e}", flush=True)
        stop(model)


# ── 2. REASONING (NOVEL_PROBLEMS, snap vs think) ──────────────────────────────
def battery_reasoning():
    hdr("REASONING — NOVEL_PROBLEMS snap(think=False) vs think(think=True), temp 0")
    models = ["lfm2.5-thinking:1.2b", "orca2:7b", "phi4:14b",
              "phi4-reasoning:14b", "gpt-oss:20b"]
    for model in models:
        snap_hits = think_hits = 0
        traces = False
        t_total = 0.0
        for label, q, gold in NOVEL_PROBLEMS:
            try:
                s, ds, _ = run(model, q + SNAP, think=False, max_tokens=60)
                snap_hits += _grade(s, gold)
            except Exception as e:
                s = f"ERR {type(e).__name__}"
            try:
                t, dt, ht = run(model, q + SNAP, think=True, budget=2500, max_tokens=60)
                think_hits += _grade(t, gold); traces |= ht; t_total += dt
            except Exception as e:
                t = f"ERR {type(e).__name__}"
            print(f"  {model:20} {label:10} snap={s[:18]!r:22} think={t[:18]!r}", flush=True)
        tag = "native-think" if traces else "no-think-trace"
        print(f"{model:20} snap {snap_hits}/4  think {think_hits}/4  "
              f"({tag}, think total {t_total:.0f}s)", flush=True)
        stop(model)


# ── 3. EMBEDDING (related-vs-unrelated gap, vs nomic) ─────────────────────────
def battery_embedding():
    hdr("EMBEDDING — GAP_PAIRS related vs unrelated cosine (bigger gap = better)")
    for model in ["nomic-embed-text", "embeddinggemma:300m"]:
        try:
            rel, unrel = [], []
            for _, a, b, c in GAP_PAIRS:
                rel.append(similarity(embed(a, model), embed(b, model)))
                unrel.append(similarity(embed(a, model), embed(c, model)))
            mr, mu = sum(rel) / len(rel), sum(unrel) / len(unrel)
            print(f"{model:22} related={mr:.3f}  unrelated={mu:.3f}  gap={mr-mu:.3f}",
                  flush=True)
        except Exception as e:
            print(f"{model:22} ERROR {type(e).__name__}: {e}", flush=True)
        stop(model)


# ── 4. MULTILINGUAL (aya vs incumbent gemma4) ─────────────────────────────────
def battery_multilingual():
    hdr("MULTILINGUAL — translate one sentence; aya:8b vs incumbent gemma4:latest")
    src = "Translate exactly into {lang}, reply with only the translation: " \
          "'The harvest was good this year, so the village celebrated.'"
    langs = ["French", "Swahili", "Yoruba", "Tagalog"]
    for model in ["gemma4:latest", "aya:8b"]:
        for lang in langs:
            try:
                out, dt, _ = run(model, src.format(lang=lang), max_tokens=80, system=None)
                print(f"  {model:14} {lang:9} -> {out[:70]!r}", flush=True)
            except Exception as e:
                print(f"  {model:14} {lang:9} ERROR {type(e).__name__}", flush=True)
        stop(model)


# ── 5. VISION (transcribe terminal.png) ───────────────────────────────────────
def battery_vision():
    hdr("VISION — transcribe images/mm/terminal.png; qwen3-vl & llava-phi3 vs llava")
    img = (CH / "images" / "mm" / "terminal.png").read_bytes()
    for model in ["llava:latest", "qwen3-vl:8b", "llava-phi3:3.8b"]:
        try:
            t0 = time.perf_counter()
            out = ask_image(img, "Transcribe the text in this image exactly.", model=model)
            print(f"  {model:16} {time.perf_counter()-t0:5.1f}s -> {out[:120]!r}", flush=True)
        except Exception as e:
            print(f"  {model:16} ERROR {type(e).__name__}: {e}", flush=True)
        stop(model)


# ── 6. GENERAL / TINY (explain + strict format) ───────────────────────────────
def battery_general():
    hdr("GENERAL — one-sentence explain + strict-format follow; speed in seconds")
    explain = "Explain what a hash table is in one sentence."
    fmt = "List exactly three fruits, comma-separated, lowercase, nothing else."
    for model in ["tinyllama:1.1b", "gemma3n:e4b", "openhermes:v2.5",
                  "lfm2:24b", "glm-4.7-flash:latest"]:
        try:
            e, de, _ = run(model, explain, max_tokens=80)
            f, df, _ = run(model, fmt, max_tokens=40)
            print(f"  {model:22} {de:4.1f}s explain -> {e[:70]!r}", flush=True)
            print(f"  {model:22}      fmt     -> {f[:60]!r}", flush=True)
        except Exception as ex:
            print(f"  {model:22} ERROR {type(ex).__name__}: {ex}", flush=True)
        stop(model)


if __name__ == "__main__":
    t0 = time.perf_counter()
    battery_code()
    battery_embedding()
    battery_multilingual()
    battery_reasoning()
    battery_vision()
    battery_general()
    print(f"\nDONE in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
