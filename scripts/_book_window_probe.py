"""Reproducer for the "does the book fit in the window?" beat in the
Augmentation chapter.

Measures the whole book corpus (prose + code + output snapshots) three ways
and compares it to gemma4's 131,072-token context window:

  * cl100k_base / o200k_base  -- fast, deterministic tiktoken BPE (the number
    the notebook cell prints via genai.tokens.count_tokens);
  * gemma4 itself             -- the on-model count, read from prompt_eval_count
    over raw chunks (slow; this is the figure the prose cites).

Run from the repo root:
    PYTHONPATH=chapters .venv/bin/python scripts/_book_window_probe.py
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))
from genai.book import load_book_corpus, load_code_corpus, load_output_corpus  # noqa: E402

WINDOW = 131_072          # gemma4's trained context length (ollama show gemma4)
DEFAULT = 4_096           # Ollama's default num_ctx
MODEL, NUM_CTX, CHUNK = "gemma4:latest", 16_384, 24_000


def book_text():
    book = load_book_corpus() + load_code_corpus() + load_output_corpus()
    return [p["text"] for p in book]


def tiktoken_count(texts, encoding):
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return sum(len(enc.encode(t, disallowed_special=())) for t in texts)


def gemma_count(texts):
    """Sum prompt_eval_count over raw chunks small enough not to be truncated."""
    blob, total = "\n\n".join(texts), 0
    for i in range(0, len(blob), CHUNK):
        body = json.dumps({
            "model": MODEL, "prompt": blob[i:i + CHUNK], "stream": False,
            "raw": True,
            "options": {"num_predict": 1, "num_ctx": NUM_CTX, "temperature": 0},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate", body,
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            total += json.loads(r.read()).get("prompt_eval_count", 0)
    return total


if __name__ == "__main__":
    texts = book_text()
    rows = [
        ("cl100k_base (GPT-4)", tiktoken_count(texts, "cl100k_base")),
        ("o200k_base (GPT-4o)", tiktoken_count(texts, "o200k_base")),
        ("gemma4 (on-model)", gemma_count(texts)),
    ]
    print(f"{'tokenizer':22} {'tokens':>9}  vs window  vs default")
    for name, n in rows:
        verdict = "OVER" if n > WINDOW else "under"
        print(f"{name:22} {n:>9,}  {verdict} {abs(n - WINDOW):>6,}  {n / DEFAULT:>5.0f}x")
