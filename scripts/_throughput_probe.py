"""Reproducer for the four-models throughput chart in efficiency.ipynb.

Asks the same one-sentence question of four models and prints the per-token
speed alongside the reading/generation split that the chart shows, then
regenerates images/efficiency/throughput.png. Generation time is derived as
gen_tokens / tokens_per_sec (pure writing time), so a cold model load never
pollutes the split, which is why the numbers are stable across warm and cold
runs.

    uv run python scripts/_throughput_probe.py

The captured 100-dpi PNG is written next to the saved chart for embedding back
into the frozen notebook cell.
"""
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "chapters"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from genai import compare_models, plot_throughput  # noqa: E402

MODELS = ["llama3.2:latest", "qwen3:4b", "gemma3:latest", "deepseek-r1:latest"]
QUESTION = "Explain recursion in one sentence."


def main():
    results = compare_models(QUESTION, MODELS)
    plot_throughput(results)  # writes images/efficiency/throughput.png

    for r in results:
        gen_s = r["gen_tokens"] / r["tokens_per_sec"] if r["tokens_per_sec"] else 0.0
        print(f"{r['model']:22s} {r['tokens_per_sec']:6.1f} t/s   "
              f"read {r['prompt_ms']:6.1f} ms   gen {gen_s:6.1f} s   "
              f"{r['gen_tokens']:>5} tok")

    # The notebook embeds figures at 100 dpi; capture one to match.
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=100, bbox_inches="tight")
    out = ROOT / "images/efficiency/throughput_cell.png"
    out.write_bytes(buf.getvalue())
    print(f"\ncell_png={out}")
    print("results_json=" + json.dumps(results))


if __name__ == "__main__":
    main()
