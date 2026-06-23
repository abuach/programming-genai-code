"""Reproducer for INDEX_SPEED_STUDY (genai/book.py): time naive linear-scan
retrieval against Chroma's HNSW index as the corpus grows.

Random nomic-width (768) vectors stand in for passages so the corpus can scale
past the book's own 421, and the query is embedded once up front, so only the
search structure is timed, not the embedding call both approaches share. The
naive scan is the DocumentStore's sort-by-similarity; Chroma is queried through
its HNSW index. Prints the study as a ready-to-paste list literal. Wall-clock
numbers drift run to run; the shape -- naive linear, Chroma flat -- is what
reproduces, and the demo bakes one captured run as INDEX_SPEED_STUDY.

Run: .venv/bin/python scripts/_index_speed_probe.py
"""
import sys
sys.path.insert(0, "chapters")
from genai.book import benchmark_index

rows = benchmark_index()
print("INDEX_SPEED_STUDY = [")
for r in rows:
    print(f"    {r!r},")
print("]")
for r in rows:
    print(f"# n={r['n']:>7,}  naive {r['naive_ms']:>7.1f} ms  "
          f"chroma {r['chroma_ms']:>5.2f} ms  speedup {r['speedup']:>5.1f}x",
          file=sys.stderr)
