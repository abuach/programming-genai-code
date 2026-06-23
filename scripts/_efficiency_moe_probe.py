"""Reproducer for the sparsity / mixture-of-experts figure in efficiency.ipynb.

The chapter shows two ways to pay less per token: quantization (fewer bits) and
routing (a smaller model for easy traffic). This probe measures the third lever,
sparsity: a model that stores many parameters but activates only a few per token.

It clocks decode speed (tokens/sec) and disk footprint for a spectrum of models
that spend different numbers of parameters per token:

  tinyllama:1.1b   dense floor          1.1B total, all 1.1B active
  llama3.2:latest  dense workhorse      3.2B total, all 3.2B active
  gemma3n:e4b      on-device, selective 6.9B total, ~4B effective per token
  lfm2:24b         mixture-of-experts   23.8B total, ~2B active per token

Active-parameter counts are architecture facts from each model's card
(LFM2-24B-A2B activates ~2B of its 23.8B; Gemma 3n E4B exposes ~4B effective
parameters via per-layer embeddings; the two dense models activate everything).
Total parameter counts and disk sizes are read live from the Ollama API, and
generation speed is measured once per model with ``time_call``. Each model is
stopped afterward to free memory on a 32GB Mac, and because every model is
clocked exactly once there is no repeated-prompt KV-cache collapse to worry
about (see efficiency_padding_demo). The probe also captures each model's reply
to a strict-format request, where the dense floor breaks while the heavier
models comply.

    uv run python scripts/_efficiency_moe_probe.py

Prints a SPARSITY_STUDY dict ready to paste into genai/perf.py.
"""
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "chapters"))

from genai.perf import time_call  # noqa: E402

OLLAMA_CHAT = "http://localhost:11434/api/chat"
OLLAMA_TAGS = "http://localhost:11434/api/tags"

# Architecture facts (params in billions): `total` is each model's card figure
# (cross-checked against the Ollama API below, which rounds TinyLlama's 1.1B to
# 1B); `active` is the per-token count from the card. Dense models activate
# everything, so total == active.
SPEC = [
    {"model": "tinyllama:1.1b",  "kind": "dense floor",        "total": 1.1, "active": 1.1},
    {"model": "llama3.2:latest", "kind": "dense workhorse",    "total": 3.2, "active": 3.2},
    {"model": "gemma3n:e4b",     "kind": "selective (PLE)",    "total": 6.9, "active": 4.0},
    {"model": "lfm2:24b",        "kind": "mixture-of-experts", "total": 23.8, "active": 2.0},
]

SPEED_PROMPT = "Explain what a hash table is in two sentences."
FORMAT_PROMPT = "List exactly three fruits, comma-separated, lowercase, nothing else."


def disk_gb():
    """Map every local model tag to its on-disk size in gigabytes."""
    tags = requests.get(OLLAMA_TAGS).json()["models"]
    return {m["name"]: m["size"] / 1e9 for m in tags}


def total_b(model):
    """Total parameter count (billions) as Ollama reports it."""
    out = subprocess.run(["ollama", "show", model], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "parameters" in line:
            val = line.split()[-1].rstrip("B")
            return float(val)
    return None


def format_reply(model):
    """The model's real answer to the strict-format request (temperature 0)."""
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": FORMAT_PROMPT}],
               "options": {"temperature": 0}}
    r = requests.post(OLLAMA_CHAT, json=payload).json()
    return (r["message"]["content"] or "").strip()


def stop(model):
    subprocess.run(["ollama", "stop", model], capture_output=True)


def resident():
    """Names of models Ollama currently holds in memory."""
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    return [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]


def fastest_tps(model, runs=3):
    """Best decode rate over a few runs. Contention only ever slows a call, so
    the fastest observed rate is the cleanest estimate of uncontended speed."""
    time_call(SPEED_PROMPT, model=model)  # warm-up / load
    best, toks = 0.0, 0
    for _ in range(runs):
        m = time_call(SPEED_PROMPT, model=model)
        if m["tokens_per_sec"] > best:
            best, toks = m["tokens_per_sec"], m["gen_tokens"]
    return best, toks


def main():
    foreign = [m for m in resident() if m not in {s["model"] for s in SPEC}]
    if foreign:
        print(f"WARNING: other models resident ({', '.join(foreign)}); they share "
              "the GPU and depress these numbers. Re-run on an idle machine.\n")
    sizes = disk_gb()
    study = []
    for spec in SPEC:
        model = spec["model"]
        tps, toks = fastest_tps(model)
        reply = format_reply(model)
        stop(model)
        ollama_total = total_b(model)  # rounded; the card figure is authoritative
        row = {
            "model": model,
            "kind": spec["kind"],
            "total_b": spec["total"],
            "active_b": spec["active"],
            "size_gb": round(sizes.get(model, 0.0), 1),
            "tps": tps,
            "gen_tokens": toks,
            "fmt_answer": reply,
        }
        study.append(row)
        note = "" if ollama_total == spec["total"] else f" (ollama reports {ollama_total})"
        print(f"{model:18} total {row['total_b']:5}B{note}  active {row['active_b']:4}B  "
              f"{row['size_gb']:5}GB  {row['tps']:6.1f} t/s  ({row['gen_tokens']} tok)",
              flush=True)
        print(f"{'':18} fmt -> {reply[:60]!r}", flush=True)

    print("\nSPARSITY_STUDY = " + json.dumps(study, indent=4))


if __name__ == "__main__":
    main()
