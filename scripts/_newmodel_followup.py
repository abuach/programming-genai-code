"""Follow-up probes after _newmodel_probe.py, to nail down three things:

  1. codeqwen:7b scored 0/12 on is_prime — confirm whether that is real or a
     harness/format artifact by printing its raw output (honesty: don't label a
     capable coder as failing if the grader just couldn't find the function).
  2. aya:8b looked weak on Swahili/Yoruba/Tagalog — but those are NOT in Aya 23's
     language list. Re-probe on languages Aya 23 DOES cover (Turkish, Vietnamese,
     Persian, Indonesian) for the fair flip side, against gemma4.
  3. gpt-oss:20b headline feature is adjustable reasoning effort. Try low vs high
     on one hard problem and report answer + visible reasoning length.

Run: uv run python scripts/_newmodel_followup.py
"""
import re, sys, time, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "chapters"))
import ollama
from genai.code import grade_is_prime

client = ollama.Client(host="http://localhost:11434")


def stop(m): subprocess.run(["ollama", "stop", m], capture_output=True)


# 1. codeqwen raw output ──────────────────────────────────────────────────────
print("=" * 70, "\n1. codeqwen:7b raw is_prime output\n", "=" * 70)
prompt = ("Write a concise Python function is_prime(n) that returns True if n "
          "is prime. Only code, no explanation.")
raw = client.chat(model="codeqwen:7b",
                  messages=[{"role": "user", "content": prompt}],
                  options={"temperature": 0, "num_predict": 320})["message"]["content"]
print(repr(raw[:400]))
m = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
code = m.group(1).strip() if m else raw
print("graded:", grade_is_prime(code))
stop("codeqwen:7b")


# 2. aya on its OWN languages vs gemma4 ────────────────────────────────────────
print("\n" + "=" * 70, "\n2. aya:8b vs gemma4 on Aya-23 languages\n", "=" * 70)
src = ("Translate exactly into {lang}, reply with only the translation: "
       "'The harvest was good this year, so the village celebrated.'")
for model in ["gemma4:latest", "aya:8b"]:
    for lang in ["Turkish", "Vietnamese", "Persian", "Indonesian"]:
        out = client.chat(model=model,
                          messages=[{"role": "user", "content": src.format(lang=lang)}],
                          options={"temperature": 0, "num_predict": 80})["message"]["content"]
        print(f"  {model:14} {lang:11} -> {out.strip()[:70]!r}")
    stop(model)


# 3. gpt-oss reasoning effort low vs high ──────────────────────────────────────
print("\n" + "=" * 70, "\n3. gpt-oss:20b reasoning effort low vs high\n", "=" * 70)
q = ("A snail is at the bottom of a 7-metre well. It climbs 4 metres each day "
     "and slides back 3 metres each night. How many days to reach the top? "
     "Reply with ONLY the final number.")
for effort in ["low", "high"]:
    try:
        t0 = time.perf_counter()
        resp = client.chat(model="gpt-oss:20b",
                          messages=[{"role": "user", "content": q}],
                          think=effort, options={"num_predict": 3000})
        dt = time.perf_counter() - t0
        msg = resp["message"]
        think_len = len(msg.get("thinking") or "")
        print(f"  effort={effort:5} {dt:5.1f}s  answer={msg['content'].strip()[:20]!r}  "
              f"reasoning_chars={think_len}")
    except Exception as e:
        print(f"  effort={effort:5} ERROR {type(e).__name__}: {e}")
stop("gpt-oss:20b")
print("\nFOLLOWUP DONE")
