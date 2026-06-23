"""Reproducer for the bat-and-ball reasoning-trace demo (thinking.ipynb).

Runs the Cognitive Reflection Test question through the chapter's thinking model
(DEEP = qwen3:4b) with the reasoning trace exposed, then prints it via
genai.thinking.show_reasoning. The model call is nondeterministic, so the first
run caches the full response to _reasoning_capture.json and later runs replay it,
which keeps the frozen notebook cell stable while we tune how much of the answer
to show. Delete the capture to draw a fresh response.

    uv run python scripts/_reasoning_probe.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chapters"))

import ollama
from genai.thinking import show_reasoning

DEEP = "qwen3:4b"
CAPTURE = Path(__file__).with_name("_reasoning_capture.json")
question = (
    "A bat and a ball cost $1.10 in total. "
    "The bat costs $1.00 more than the ball. "
    "How much does the ball cost?"
)

if CAPTURE.exists():
    resp = json.loads(CAPTURE.read_text(encoding="utf-8"))
else:
    client = ollama.Client(host="http://localhost:11434")
    live = client.chat(
        model=DEEP, messages=[{"role": "user", "content": question}],
        think=True, options={"num_predict": 2000},
    )
    resp = {"message": {"thinking": live["message"].get("thinking") or "",
                        "content": live["message"]["content"]}}
    CAPTURE.write_text(json.dumps(resp, indent=1, ensure_ascii=False), encoding="utf-8")

show_reasoning(resp)
