"""Probe Whisper capabilities for the mm.ipynb audio expansion.

Throwaway reproducer: checks segment- and word-level timestamps and the
detected language on the existing narration clip so the notebook demo uses
real, measured output (no invented values).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapters"))

from genai.audio import read_wav, _asr

WAV = "chapters/audio/narration.wav"

samples, sr = read_wav(WAV)
print(f"clip: {len(samples)/sr:.2f}s @ {sr} Hz, {len(samples)} samples\n")

asr = _asr()

# Segment-level timestamps
seg = asr({"array": samples, "sampling_rate": sr},
          return_timestamps=True,
          generate_kwargs={"language": "en", "task": "transcribe"})
print("=== SEGMENT TIMESTAMPS ===")
for ch in seg["chunks"]:
    print(ch["timestamp"], repr(ch["text"]))

# Word-level timestamps
word = asr({"array": samples, "sampling_rate": sr},
           return_timestamps="word",
           generate_kwargs={"language": "en", "task": "transcribe"})
print("\n=== WORD TIMESTAMPS ===")
for ch in word["chunks"]:
    s, e = ch["timestamp"]
    print(f"{s:5.2f}-{e:5.2f}  {ch['text']!r}")
print(f"\nword count: {len(word['chunks'])}")
